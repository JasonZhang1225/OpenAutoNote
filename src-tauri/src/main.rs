#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::json;
use tauri::{Manager, WebviewWindow};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const TARGET_PORT: u16 = 8964;
const HEALTH_PATH: &str = "/health";
const MAX_WAIT_SECS: u64 = 180;

enum BackendChild {
    Sidecar(CommandChild),
    Process(std::process::Child),
    External,
}

struct BackendState {
    child: Mutex<Option<BackendChild>>,
}

fn target_port() -> u16 {
    std::env::var("OPENAUTONOTE_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TARGET_PORT)
}

fn backend_ready(port: u16) -> bool {
    match std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(400),
    ) {
        Ok(mut stream) => {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(400)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(400)));
            let req = format!(
                "GET {HEALTH_PATH} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
            );
            if stream.write_all(req.as_bytes()).is_err() {
                return false;
            }
            let mut buf = [0u8; 128];
            match stream.read(&mut buf) {
                Ok(n) if n > 0 => {
                    let head = String::from_utf8_lossy(&buf[..n]);
                    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
                }
                _ => false,
            }
        }
        Err(_) => false,
    }
}

fn kill_listener_on_port(port: u16) {
    #[cfg(unix)]
    {
        if let Ok(output) = Command::new("lsof")
            .args(["-ti", &format!("tcp:{port}")])
            .output()
        {
            for pid in String::from_utf8_lossy(&output.stdout).split_whitespace() {
                if pid.chars().all(|c| c.is_ascii_digit()) {
                    let _ = Command::new("kill").args(["-9", pid]).output();
                }
            }
        }
    }

    #[cfg(windows)]
    {
        let _ = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
                ),
            ])
            .output();
    }
}

fn python_core_main() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev = manifest.join("../python-core/main.py");
    if let Ok(p) = dev.canonicalize() {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidates = [
                dir.join("python-core/main.py"),
                dir.join("../python-core/main.py"),
                dir.join("../../python-core/main.py"),
                dir.join("../../../python-core/main.py"),
            ];
            for c in candidates {
                if let Ok(p) = c.canonicalize() {
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

fn find_python(script: &std::path::Path) -> Option<PathBuf> {
    if let Some(root) = script.parent() {
        #[cfg(windows)]
        let venvs = [
            root.join("venv/Scripts/python.exe"),
            root.join("oan2/Scripts/python.exe"),
        ];
        #[cfg(not(windows))]
        let venvs = [root.join("venv/bin/python"), root.join("oan2/bin/python")];
        for p in venvs {
            if p.is_file() {
                return Some(p);
            }
        }
    }
    for name in ["python3", "python"] {
        #[cfg(windows)]
        let which = "where";
        #[cfg(not(windows))]
        let which = "which";
        if let Ok(out) = Command::new(which).arg(name).output() {
            if out.status.success() {
                if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                    let p = PathBuf::from(line.trim());
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

fn spawn_python_fallback(port: u16) -> Result<std::process::Child, String> {
    let script = python_core_main().ok_or_else(|| {
        "找不到 python-core/main.py，且 sidecar 二进制 api-server 不可用。请先在 python-core 安装依赖后运行 python main.py。".to_string()
    })?;
    let python = find_python(&script).ok_or_else(|| {
        "系统 PATH 中没有 python3/python，无法回退启动后端。".to_string()
    })?;

    let port_s = port.to_string();
    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .args(["--port", &port_s, "--host", "127.0.0.1", "--watchdog"])
        .env("OPENAUTONOTE_WATCHDOG", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }

    cmd.spawn()
        .map_err(|e| format!("启动 Python 后端失败 ({python:?}): {e}"))
}

fn set_splash_status(splash: &Option<WebviewWindow>, title: &str, subtitle: &str, failed: bool) {
    let Some(splash) = splash else { return };
    let payload = json!({
        "title": title,
        "subtitle": subtitle,
        "failed": failed,
    });
    let js = format!(
        r#"
        (function(d) {{
          var t = document.getElementById('status-title');
          var s = document.getElementById('status-sub');
          var sp = document.getElementById('spinner');
          if (t) t.textContent = d.title;
          if (s) s.textContent = d.subtitle;
          if (sp) {{
            if (d.failed) sp.classList.add('error');
            else sp.classList.remove('error');
          }}
        }})({payload});
        "#
    );
    let _ = splash.eval(&js);
}

async fn show_main_and_close_splash(
    main: Option<WebviewWindow>,
    splash: Option<WebviewWindow>,
    port: u16,
) {
    if let Some(main) = main {
        let url = format!("http://127.0.0.1:{port}/");
        let url_js = serde_json::to_string(&url)
            .unwrap_or_else(|_| "\"http://127.0.0.1:8964/\"".into());
        let js = format!("window.location.replace({url_js});");
        for _ in 0..20 {
            if main.eval(&js).is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(splash) = splash {
        let _ = splash.close();
    }
}

fn kill_backend(state: &BackendState) {
    if let Ok(mut guard) = state.child.lock() {
        match guard.take() {
            Some(BackendChild::Sidecar(child)) => {
                let _ = child.kill();
            }
            Some(BackendChild::Process(mut child)) => {
                let _ = child.kill();
            }
            _ => {}
        }
    }
}

fn main() {
    let port = target_port();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendState {
            child: Mutex::new(None),
        })
        .setup(move |app| {
            let app_handle = app.handle().clone();
            let splash = app_handle.get_webview_window("splashscreen");
            let main_window = app_handle.get_webview_window("main");
            set_splash_status(
                &splash,
                "正在连接后端…",
                "Connecting to Python engine",
                false,
            );

            if !backend_ready(port) {
                kill_listener_on_port(port);
            }

            let mut spawn_error: Option<String> = None;
            let mut spawned = false;

            if backend_ready(port) {
                if let Ok(mut guard) = app.state::<BackendState>().child.lock() {
                    *guard = Some(BackendChild::External);
                }
                spawned = true;
            } else {
                match app.shell().sidecar("api-server") {
                    Ok(cmd) => match cmd
                        .env("OPENAUTONOTE_WATCHDOG", "1")
                        .args([
                            "--port",
                            &port.to_string(),
                            "--host",
                            "127.0.0.1",
                            "--watchdog",
                        ])
                        .spawn()
                    {
                        Ok((mut rx, child)) => {
                            if let Ok(mut guard) = app.state::<BackendState>().child.lock() {
                                *guard = Some(BackendChild::Sidecar(child));
                            }
                            spawned = true;
                            tauri::async_runtime::spawn(async move {
                                while let Some(event) = rx.recv().await {
                                    match event {
                                        CommandEvent::Stdout(line) => {
                                            println!("[PY] {}", String::from_utf8_lossy(&line));
                                        }
                                        CommandEvent::Stderr(line) => {
                                            eprintln!(
                                                "[PY ERR] {}",
                                                String::from_utf8_lossy(&line)
                                            );
                                        }
                                        CommandEvent::Terminated(payload) => {
                                            eprintln!("[PY] sidecar exited: {:?}", payload);
                                        }
                                        CommandEvent::Error(err) => {
                                            eprintln!("[PY ERR] {err}");
                                        }
                                        _ => {}
                                    }
                                }
                            });
                        }
                        Err(err) => {
                            spawn_error = Some(format!("sidecar spawn failed: {err}"));
                        }
                    },
                    Err(err) => {
                        spawn_error = Some(format!("sidecar missing: {err}"));
                    }
                }

                if !spawned {
                    match spawn_python_fallback(port) {
                        Ok(mut child) => {
                            if let Some(stdout) = child.stdout.take() {
                                std::thread::spawn(move || {
                                    use std::io::BufRead;
                                    let reader = std::io::BufReader::new(stdout);
                                    for line in reader.lines().flatten() {
                                        println!("[PY] {line}");
                                    }
                                });
                            }
                            if let Some(stderr) = child.stderr.take() {
                                std::thread::spawn(move || {
                                    use std::io::BufRead;
                                    let reader = std::io::BufReader::new(stderr);
                                    for line in reader.lines().flatten() {
                                        eprintln!("[PY ERR] {line}");
                                    }
                                });
                            }
                            if let Ok(mut guard) = app.state::<BackendState>().child.lock() {
                                *guard = Some(BackendChild::Process(child));
                            }
                            spawned = true;
                            spawn_error = None;
                        }
                        Err(err) => {
                            let combined = match spawn_error {
                                Some(prev) => format!("{prev}; fallback: {err}"),
                                None => err,
                            };
                            spawn_error = Some(combined);
                        }
                    }
                }
            }

            if !spawned {
                let msg = spawn_error.unwrap_or_else(|| "无法启动 Python 后端".into());
                eprintln!("[OpenAutoNote] {msg}");
                set_splash_status(&splash, "后端启动失败", &msg, true);
                if let Some(splash_win) = &splash {
                    let _ = splash_win.set_always_on_top(false);
                    let _ = splash_win.set_decorations(true);
                }
                let _ = main_window.as_ref().map(|w| w.show());
                return Ok(());
            }

            tauri::async_runtime::spawn(async move {
                let mut elapsed = 0u64;
                let mut ready = false;
                while elapsed <= MAX_WAIT_SECS {
                    if backend_ready(port) {
                        ready = true;
                        break;
                    }
                    if elapsed % 5 == 0 {
                        set_splash_status(
                            &splash,
                            "正在连接后端…",
                            &format!("等待 Python 引擎就绪 ({elapsed}s / {MAX_WAIT_SECS}s)"),
                            false,
                        );
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    elapsed += 1;
                }

                if ready {
                    show_main_and_close_splash(main_window, splash, port).await;
                } else {
                    let msg = format!(
                        "Python 后端在 {MAX_WAIT_SECS}s 内没有响应 http://127.0.0.1:{port}{HEALTH_PATH}。\n请确认已安装依赖并手动运行: python python-core/main.py"
                    );
                    eprintln!("[OpenAutoNote] {msg}");
                    set_splash_status(&splash, "连接超时", &msg, true);
                    if let Some(splash_win) = &splash {
                        let _ = splash_win.set_always_on_top(false);
                        let _ = splash_win.set_decorations(true);
                    }
                    if let Some(main) = main_window {
                        let _ = main.show();
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // Only the main window should tear down the app. Closing the
                // splash after a successful handshake used to call exit(0)
                // and looked like a launch crash (GitHub issue #1).
                if window.label() == "main" {
                    let state = window.state::<BackendState>();
                    kill_backend(&state);
                    window.app_handle().exit(0);
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

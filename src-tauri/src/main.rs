#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::process::Command;
use std::thread;
use std::time::Duration;

use tauri::{Manager, WebviewWindow};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

// Define the port we expect the Python server to listen on
const TARGET_PORT: u16 = 8964;

fn kill_zombie_sidecars() {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/IM", "api-server.exe"])
            .output();
    }

    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("pkill").args(["-f", "api-server"]).output();
    }

    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("pkill").args(["-f", "api-server"]).output();
    }
}

fn backend_ready() -> bool {
    if let Ok(mut stream) = std::net::TcpStream::connect(("127.0.0.1", TARGET_PORT)) {
        let _ = stream.write_all(
            b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        );
        let mut buf = [0u8; 64];
        if let Ok(n) = stream.read(&mut buf) {
            let head = String::from_utf8_lossy(&buf[..n]);
            return head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200");
        }
    }
    false
}

fn show_main_and_close_splash(main: Option<WebviewWindow>, splash: Option<WebviewWindow>) {
    if let Some(splash) = splash {
        let _ = splash.close();
    }
    if let Some(main) = main {
        let _ = main.show();
        let _ = main.set_focus();
    }
}

fn main() {
    kill_zombie_sidecars();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let splash = app_handle.get_webview_window("splashscreen");
            let main_window = app_handle.get_webview_window("main");

            // 1. Spawn the Sidecar
            let sidecar_command = app
                .shell()
                .sidecar("api-server")
                .expect("Failed to create sidecar command");

            let (mut rx, _child) = sidecar_command
                .args(["--port", &TARGET_PORT.to_string()])
                .spawn()
                .expect("Failed to spawn sidecar");

            // 2. Handle Sidecar Events (logging) in a separate thread
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line) = event {
                        let log = String::from_utf8_lossy(&line);
                        println!("[PY] {}", log);
                    } else if let CommandEvent::Stderr(line) = event {
                        let log = String::from_utf8_lossy(&line);
                        eprintln!("[PY ERR] {}", log);
                    }
                }
            });

            // 3. Wait for Python Server to be Ready, then swap splash -> main
            tauri::async_runtime::spawn(async move {
                let mut attempts = 0;
                let max_attempts = 60; // ~30s
                let mut ready = false;

                while attempts < max_attempts {
                    if backend_ready() {
                        ready = true;
                        break;
                    }
                    thread::sleep(Duration::from_millis(500));
                    attempts += 1;
                }

                if ready {
                    show_main_and_close_splash(main_window, splash);
                } else {
                    eprintln!("Failed to connect to Python backend after timeout.");
                    app_handle.exit(1);
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::CloseRequested { .. } => {
                #[cfg(not(target_os = "macos"))]
                {
                    window.app_handle().exit(0);
                }
                #[cfg(target_os = "macos")]
                {
                    window.app_handle().exit(0);
                }
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

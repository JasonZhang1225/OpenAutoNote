#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{WebviewUrl, WebviewWindowBuilder, Manager};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use std::time::Duration;
use std::thread;
use std::net::TcpStream;

// Define the port we expect the Python server to listen on
const TARGET_PORT: u16 = 8964;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            
            // 1. Spawn the Sidecar
            // Note: The binary name in tauri.conf.json must allow this command
            let sidecar_command = app.shell().sidecar("api-server")
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
                        println!("[PY] {}", log); // Forward to Rust stdout
                    } else if let CommandEvent::Stderr(line) = event {
                        let log = String::from_utf8_lossy(&line);
                        eprintln!("[PY ERR] {}", log);
                    }
                }
            });

            // 3. Wait for Python Server to be Ready
            tauri::async_runtime::spawn(async move {
                let mut attempts = 0;
                let max_attempts = 30; // 15 seconds roughly
                let mut ready = false;

                while attempts < max_attempts {
                    match TcpStream::connect(format!("127.0.0.1:{}", TARGET_PORT)) {
                        Ok(_) => {
                            ready = true;
                            break;
                        }
                        Err(_) => {
                            thread::sleep(Duration::from_millis(500));
                            attempts += 1;
                        }
                    }
                }

                if ready {
                    println!("Python server is ready! Creating main window...");
                    // 4. Create the Main Window pointing to localhost
                    // We execute this on the main thread via the dispatcher if needed,
                    // but WebviewWindowBuilder is thread-safe enough if we have the app_handle.
                    // Wait, WebviewWindowBuilder::new takes &AppHandle.
                    
                    let _ = WebviewWindowBuilder::new(
                        &app_handle,
                        "main",
                        WebviewUrl::External(format!("http://localhost:{}", TARGET_PORT).parse().unwrap())
                    )
                    .title("OpenAutoNote")
                    .inner_size(1200.0, 800.0)
                    .build();
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

#!/usr/bin/env node
/**
 * Cross-platform backend starter for `tauri dev`.
 * Prefers python-core venv / oan2, then python3, then python.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "python-core");
const script = path.join(root, "main.py");

function exists(p) {
  try {
    fs.accessSync(p, fs.constants.X_OK);
    return true;
  } catch {
    try {
      fs.accessSync(p, fs.constants.F_OK);
      return true;
    } catch {
      return false;
    }
  }
}

const candidates = [
  path.join(root, "venv", "bin", "python"),
  path.join(root, "oan2", "bin", "python"),
  path.join(root, "venv", "Scripts", "python.exe"),
  path.join(root, "oan2", "Scripts", "python.exe"),
  "python3",
  "python",
];

const python = candidates.find((p) => (p.includes(path.sep) ? exists(p) : true));
if (!python) {
  console.error("No Python interpreter found. Install Python 3.10+.");
  process.exit(1);
}

const extra = process.argv.slice(2);
const args = [script, "--port", "8964", "--host", "127.0.0.1", ...extra];
console.log(`[start-backend] ${python} ${args.join(" ")}`);

const child = spawn(python, args, {
  cwd: root,
  stdio: "inherit",
  env: process.env,
  windowsHide: true,
});
child.on("exit", (code) => process.exit(code ?? 1));
child.on("error", (err) => {
  console.error("Failed to start Python backend:", err.message);
  process.exit(1);
});

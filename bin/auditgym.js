#!/usr/bin/env node
// AuditGym one-command launcher:  npx -y --no-update-notifier https://codeload.github.com/itsaibarr/auditgym/tar.gz/master [command ...]   (default command: tour)
// Needs only Node >= 16 (the tarball URL needs no git; `npx github:...` would). It finds or downloads `uv`
// (a single static binary), which brings its own Python 3.11 and the locked dependencies. No API keys, no GPU.
"use strict";
const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const LAUNCHER = "npx -y --no-update-notifier --prefer-offline https://codeload.github.com/itsaibarr/auditgym/tar.gz/master";  // tarball URL: no git; --prefer-offline: printed next steps start in ~1 s offline
const CACHE = process.env.AUDITGYM_HOME || path.join(os.homedir(), ".cache", "auditgym");
const args = process.argv.slice(2).length ? process.argv.slice(2) : ["tour"];
const isWin = process.platform === "win32";

function say(msg) { process.stderr.write(`[auditgym] ${msg}\n`); }
function which(bin) {
  const r = spawnSync(isWin ? "where" : "which", [bin], { encoding: "utf8" });
  return r.status === 0 ? r.stdout.split(/\r?\n/)[0].trim() : null;
}
function exists(p) { try { return fs.statSync(p).isFile(); } catch { return false; } }

const UV_VERSION = "0.12.17";  // same uv as the Docker image
const UV_SHA256 = {  // as pinned in https://astral.sh/uv/0.12.17/install.sh
  "aarch64-apple-darwin": "85f00cbdc6dd3e97eba4c31b4d014375a9fdfe8f570023b84e5102fc3456896b",
  "x86_64-apple-darwin": "8dcf05a8c809bb3c471d2b614788ba27a6e41298fc8c31ac84b5f4339fd468e5",
  "aarch64-unknown-linux-musl": "a6096da273d548cb9f277d237a01ac7344a39ef0f455c0e148e4dc9737c1596b",
  "x86_64-unknown-linux-musl": "6401c4665d8fa2a9893e087c91f585430738e3170f5398a1141483efb4a93310",
};
function installUv() {
  // Download the static uv binary with Node itself (no curl/wget needed), pinned version, into ~/.cache/auditgym/bin.
  const arch = { x64: "x86_64", arm64: "aarch64" }[process.arch];
  const target = process.platform === "darwin" ? `${arch}-apple-darwin` : `${arch}-unknown-linux-musl`;
  if (!arch || !["darwin", "linux"].includes(process.platform)) { say(`unsupported platform ${process.platform}/${process.arch}`); process.exit(2); }
  // Astral's own CDN first (as its install.sh does: often much faster than GitHub), then GitHub
  const urls = ["https://releases.astral.sh/github/uv/releases/download", "https://github.com/astral-sh/uv/releases/download"]
    .map((b) => `${b}/${UV_VERSION}/uv-${target}.tar.gz`);
  const bin = path.join(CACHE, "bin");
  fs.mkdirSync(bin, { recursive: true });
  const tgz = path.join(bin, `uv-${process.pid}.tar.gz`);
  say(`downloading uv ${UV_VERSION} into ${bin} (besides uv's own caches ~/.cache/uv and ~/.local/share/uv, nothing else on your system changes) ...`);
  // curl honours HTTPS_PROXY; Node https (no proxy support) is the fallback
  const bar = process.stderr.isTTY ? "-#" : "-sS";
  const fetch1 = (url) => which("curl") ? spawnSync("curl", ["-fL", bar, "-C", "-", "--retry", "3", "--connect-timeout", "20", "--speed-limit", "1000", "--speed-time", "60", "-o", tgz, url], { stdio: ["ignore", "inherit", "inherit"] })
    : which("wget") ? spawnSync("wget", [...(process.stderr.isTTY ? [] : ["-q"]), "-c", "-T", "30", "-t", "3", "-O", tgz, url], { stdio: ["ignore", "inherit", "inherit"] })
    : spawnSync(process.execPath, ["-e", `
    const https = require("https"), fs = require("fs");
    const get = (u, n) => https.get(u, { headers: { "user-agent": "auditgym-launcher" }, timeout: 30000 }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && n < 8) { res.resume(); return get(res.headers.location, n + 1); }
      if (res.statusCode !== 200) { console.error("HTTP " + res.statusCode + " for " + u); process.exit(1); }
      const total = +res.headers["content-length"] || 0; let got = 0, last = Date.now();
      res.on("data", (c) => { got += c.length; if (Date.now() - last > 5000) { last = Date.now(); process.stderr.write("[auditgym] uv: " + (got / 1e6).toFixed(1) + "/" + (total / 1e6).toFixed(1) + " MB\\n"); } });
      res.pipe(fs.createWriteStream(process.argv[2])).on("finish", () => process.exit(0));
    }).on("timeout", function () { console.error("download stalled (no data for 30 s)"); this.destroy(); process.exit(1); })
      .on("error", (e) => { console.error(String(e)); process.exit(1); });
    get(process.argv[1], 0);`, url, tgz], { stdio: ["ignore", "inherit", "inherit"], env: { ...process.env, NODE_USE_ENV_PROXY: "1" } });  // Node >= 22.21 / 24.5: honour HTTPS_PROXY
  let dl, sum = "";
  for (const url of urls) {
    dl = fetch1(url);
    for (let k = 2; k <= 3 && dl.status !== 0; k++) { say(`download interrupted, attempt ${k}/3 ...`); dl = fetch1(url); }
    try { sum = require("crypto").createHash("sha256").update(fs.readFileSync(tgz)).digest("hex"); } catch {}
    if (dl.status === 0 && sum === UV_SHA256[target]) break;
    fs.rmSync(tgz, { force: true });
  }
  if (dl.status !== 0 || sum !== UV_SHA256[target]) {
    fs.rmSync(tgz, { force: true });
    say("could not download uv (network drop or proxy?); nothing was installed - run the same command again, or use Docker (see README)");
    process.exit(2);
  }
  const tmp = fs.mkdtempSync(path.join(bin, ".unpack-"));
  const tx = spawnSync("tar", ["-xzf", tgz, "-C", tmp, "--strip-components=1"], { stdio: "inherit" });
  fs.rmSync(tgz, { force: true });
  const p = path.join(bin, "uv");
  if (tx.status !== 0 || !exists(path.join(tmp, "uv"))) { fs.rmSync(tmp, { recursive: true, force: true }); say("could not unpack uv (needs `tar`)"); process.exit(2); }
  fs.renameSync(path.join(tmp, "uv"), p);  // atomic: the cached uv is always complete and verified
  fs.rmSync(tmp, { recursive: true, force: true });
  return p;
}

if (isWin) {
  // The isolated policy runner relies on POSIX process control; on Windows we run the same thing in Docker.
  if (!which("docker")) {
    say("Windows: install Docker Desktop (https://docs.docker.com/desktop/) or run this same command inside WSL (`wsl --install`).");
    process.exit(2);
  }
  say("Windows detected: building the Docker image from this package (first time only, needs network) ...");
  let r = spawnSync("docker", ["build", "-q", "-t", "auditgym:1.0.0", ROOT], { stdio: ["ignore", 2, "inherit"] });
  if (r.status !== 0) {
    const have = spawnSync("docker", ["image", "inspect", "auditgym:1.0.0"], { stdio: "ignore" }).status === 0;
    if (!have) { say("docker build failed (the first run needs the network to pull python:3.11-slim); start Docker Desktop and run the same command again"); process.exit(r.status || 1); }
    say("rebuild failed (offline?): using the existing auditgym:1.0.0 image");
  }
  const run = ["run", "--rm", "-i", "--network", "none"];
  let rep = null;
  if (args[0] === "tour" && !args.includes("--out")) {
    rep = path.join(process.cwd(), "auditgym-report");
    fs.mkdirSync(rep, { recursive: true });
    run.push("-v", `${rep}:/report`);
    args.push("--out", "/report");
  }
  r = spawnSync("docker", [...run, "-e", "AUDITGYM_LAUNCHER=docker run --rm -i --network none auditgym:1.0.0", "-e", "AUDITGYM_SHELL=powershell",
    ...(rep ? ["-e", `AUDITGYM_REPORT_HOST=${rep}`] : []), "auditgym:1.0.0", ...args], { stdio: "inherit" });
  process.exit(r.status === null ? 1 : r.status);
}

const first = !fs.existsSync(path.join(ROOT, ".venv"));  // first run: say the plan before any download, show uv's progress
const T0 = Date.now();
if (first) { say("first run: one-time setup before the tour, usually 20-60 s (up to 3 min on a slow network):"); say("  uv, a Python installer (~20 MB), and Python 3.11 (~30 MB download), each only if missing; then 19 pinned packages."); say("  Ctrl-C is safe: running the same command again resumes. Later runs start in a few seconds."); }
const runs = (u) => spawnSync(u, ["--version"], { timeout: 15000 }).status === 0;  // a cached/system uv must actually start
let uv = process.env.AUDITGYM_UV ||
  [path.join(CACHE, "bin", "uv"), which("uv"), path.join(os.homedir(), ".local", "bin", "uv"), path.join(os.homedir(), ".cargo", "bin", "uv")].find((u) => u && exists(u) && runs(u));
if (!uv) uv = installUv();
// run a setup step quietly with a live line every 2 s (bin/tick.sh, shared with run.sh); this path is macOS/Linux only
const ticked = (label, cmd) => spawnSync("sh", ["-c", `say() { printf '[auditgym] %s\\n' "$*" >&2; }; UV="$1"; shift; . "$1"; shift; tick "$@"`,
  "sh", uv, path.join(ROOT, "bin", "tick.sh"), label, ...cmd], { stdio: ["ignore", 2, "inherit"] }).status;
if (first) {  // quiet installs with a live line every 2 s, then one receipt line
  if (spawnSync(uv, ["python", "find", "3.11"], { stdio: "ignore" }).status !== 0) {
    say("downloading Python 3.11 (~30 MB download, ~60 MB on disk; usually 5-30 s, one time) ...");
    if (ticked("Python 3.11 download", [uv, "python", "install", "--quiet", "--no-bin", "3.11"]) !== 0) {
      say("Python download failed: run the same command again (finished downloads are kept)"); process.exit(2);
    }
  }
  say("installing the 19 pinned packages (usually 5-20 s) ...");
  if (ticked("package install", [uv, "sync", "--quiet", "--frozen", "--extra", "gym", "--project", ROOT]) !== 0) {
    say("package install failed: run the same command again (finished downloads are kept)"); process.exit(2);
  }
  say(`✓ setup done in ${Math.round((Date.now() - T0) / 1000)} s: uv, Python 3.11 and the packages (later runs start in a few seconds)`);
}
const r = spawnSync(uv, ["run", "--quiet", "--frozen", "--extra", "gym", "--project", ROOT, "auditgym", ...args], {
  stdio: "inherit",
  env: { ...process.env, AUDITGYM_LAUNCHER: LAUNCHER },
});
if (r.error) { say(String(r.error)); process.exit(2); }
if (r.status !== 0 && r.status !== 1) say("exit code 2: if the message above is a network/download error, run the same command again (finished downloads are kept); otherwise it explains what to change");
process.exit(r.status === null ? 1 : r.status);

#!/usr/bin/env node
// AuditGym one-command launcher:  npx -y https://codeload.github.com/itsaibarr/auditgym/tar.gz/master [command ...]   (default command: tour)
// Needs only Node >= 16 (the tarball URL needs no git; `npx github:...` would). It finds or downloads `uv`
// (a single static binary), which brings its own Python 3.11 and the locked dependencies. No API keys, no GPU.
"use strict";
const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const LAUNCHER = "npx -y https://codeload.github.com/itsaibarr/auditgym/tar.gz/master";  // tarball URL: npm needs no git for it
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
function installUv() {
  // Download the static uv binary with Node itself (no curl/wget needed), pinned version, into ~/.cache/auditgym/bin.
  const arch = { x64: "x86_64", arm64: "aarch64" }[process.arch];
  const target = process.platform === "darwin" ? `${arch}-apple-darwin` : `${arch}-unknown-linux-musl`;
  if (!arch || !["darwin", "linux"].includes(process.platform)) { say(`unsupported platform ${process.platform}/${process.arch}`); process.exit(2); }
  const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${target}.tar.gz`;
  const bin = path.join(CACHE, "bin");
  fs.mkdirSync(bin, { recursive: true });
  const tgz = path.join(bin, "uv.tar.gz");
  say(`uv not found: downloading uv ${UV_VERSION} (~20 MB, one time) into ${bin}; nothing else on your system changes ...`);
  const dl = spawnSync(process.execPath, ["-e", `
    const https = require("https"), fs = require("fs");
    const get = (u, n) => https.get(u, { headers: { "user-agent": "auditgym-launcher" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && n < 8) { res.resume(); return get(res.headers.location, n + 1); }
      if (res.statusCode !== 200) { console.error("HTTP " + res.statusCode + " for " + u); process.exit(1); }
      res.pipe(fs.createWriteStream(process.argv[2])).on("finish", () => process.exit(0));
    }).on("error", (e) => { console.error(String(e)); process.exit(1); });
    get(process.argv[1], 0);`, url, tgz], { stdio: ["ignore", "inherit", "inherit"] });
  if (dl.status !== 0) { say("could not download uv; see https://docs.astral.sh/uv/getting-started/installation/"); process.exit(2); }
  const tx = spawnSync("tar", ["-xzf", tgz, "-C", bin, "--strip-components=1"], { stdio: "inherit" });
  fs.rmSync(tgz, { force: true });
  const p = path.join(bin, "uv");
  if (tx.status !== 0 || !exists(p)) { say("could not unpack uv (needs `tar`)"); process.exit(2); }
  return p;
}

if (isWin) {
  // The isolated policy runner relies on POSIX process control; on Windows we run the same thing in Docker.
  if (!which("docker")) {
    say("Windows: install Docker Desktop (https://docs.docker.com/desktop/) or run this same command inside WSL (`wsl --install`).");
    process.exit(2);
  }
  say("Windows detected: building the Docker image from this package (first time only, needs network) ...");
  let r = spawnSync("docker", ["build", "-q", "-t", "auditgym:1.0.0", ROOT], { stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status || 1);
  const run = ["run", "--rm", "-i", "--network", "none"];
  if (args[0] === "tour" && !args.includes("--out")) {
    const rep = path.join(process.cwd(), "auditgym-report");
    fs.mkdirSync(rep, { recursive: true });
    run.push("-v", `${rep}:/report`);
    args.push("--out", "/report");
  }
  r = spawnSync("docker", [...run, "-e", "AUDITGYM_LAUNCHER=docker run --rm -i --network none auditgym:1.0.0", "auditgym:1.0.0", ...args], { stdio: "inherit" });
  process.exit(r.status === null ? 1 : r.status);
}

let uv = process.env.AUDITGYM_UV || which("uv") ||
  [path.join(CACHE, "bin", "uv"), path.join(os.homedir(), ".local", "bin", "uv"), path.join(os.homedir(), ".cargo", "bin", "uv")].find(exists);
if (!uv) uv = installUv();
say(`running: auditgym ${args.join(" ")}   (first run downloads Python 3.11 + pinned deps; later runs start in ~1 s)`);
const r = spawnSync(uv, ["run", "--quiet", "--frozen", "--extra", "gym", "--project", ROOT, "auditgym", ...args], {
  stdio: "inherit",
  env: { ...process.env, AUDITGYM_LAUNCHER: LAUNCHER },
});
if (r.error) { say(String(r.error)); process.exit(2); }
if (r.status !== 0 && r.status !== 1) say("if this was a download error: just run the same command again (finished downloads are kept), or use Docker (see README)");
process.exit(r.status === null ? 1 : r.status);

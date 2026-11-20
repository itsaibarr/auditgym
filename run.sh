#!/bin/sh
# AuditGym one-command launcher for macOS / Linux, nothing preinstalled except curl (or wget) and tar:
#   curl -fsSL https://raw.githubusercontent.com/itsaibarr/auditgym/master/run.sh | sh              # the jury tour
#   curl -fsSL https://raw.githubusercontent.com/itsaibarr/auditgym/master/run.sh | sh -s -- check  # any command
# Downloads the repository and `uv` (a single static binary) into ~/.cache/auditgym; uv brings Python 3.11 and the
# locked dependencies. No API keys, no GPU, no sudo; nothing outside ~/.cache/auditgym is changed.
set -eu
REF="${AUDITGYM_REF:-master}"
HOME_DIR="${AUDITGYM_HOME:-$HOME/.cache/auditgym}"
SRC="$HOME_DIR/src-$REF"
URL="${AUDITGYM_TARBALL:-https://codeload.github.com/itsaibarr/auditgym/tar.gz/$REF}"
say() { printf '[auditgym] %s\n' "$*" >&2; }
fetch() {  # fetch URL to stdout, with timeouts and retries (a stalled download must not hang the jury)
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 600 "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -T 30 -t 3 -O- "$1"
  else say "need curl or wget"; exit 2; fi
}
[ $# -eq 0 ] && set -- tour

if [ "$(uname -s)" != "Darwin" ] && [ "$(uname -s)" != "Linux" ]; then
  say "this launcher supports macOS and Linux; on Windows use Docker Desktop or WSL (see README)"; exit 2
fi

mkdir -p "$HOME_DIR/bin"
HERE=""
if [ -f "$0" ] && [ "$(basename "$0")" = "run.sh" ]; then HERE="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"; fi
if [ -z "${AUDITGYM_SRC:-}" ] && [ -n "$HERE" ] && [ -f "$HERE/uv.lock" ] && [ -d "$HERE/auditgym" ]; then
  AUDITGYM_SRC="$HERE"   # run from a checkout: use it, no download
fi
if [ -z "${AUDITGYM_SRC:-}" ]; then
  mkdir -p "$SRC"
  say "downloading AuditGym ($REF) into $SRC ..."
  case "$URL" in
    http*) fetch "$URL" | tar -xz --strip-components=1 -C "$SRC" ;;
    *) tar -xzf "$URL" --strip-components=1 -C "$SRC" ;;
  esac
else
  SRC="$AUDITGYM_SRC"
fi

UV="${AUDITGYM_UV:-}"
[ -z "$UV" ] && command -v uv >/dev/null 2>&1 && UV="$(command -v uv)"
[ -z "$UV" ] && [ -x "$HOME_DIR/bin/uv" ] && UV="$HOME_DIR/bin/uv"
if [ -z "$UV" ]; then
  case "$(uname -m)" in x86_64|amd64) A=x86_64 ;; arm64|aarch64) A=aarch64 ;; *) say "unsupported CPU $(uname -m); use Docker"; exit 2 ;; esac
  if [ "$(uname -s)" = "Darwin" ]; then T="$A-apple-darwin"; else T="$A-unknown-linux-musl"; fi
  say "downloading uv 0.12.17 ($T, static binary, ~20 MB) into $HOME_DIR/bin (one time) ..."
  if ! fetch "https://github.com/astral-sh/uv/releases/download/0.12.17/uv-$T.tar.gz" | tar -xz --strip-components=1 -C "$HOME_DIR/bin"; then
    say "could not download uv: run the same command again, or use Docker (see README)"; exit 2
  fi
  UV="$HOME_DIR/bin/uv"
  [ -x "$UV" ] || { say "uv download incomplete: run the same command again"; exit 2; }
fi

say "running: auditgym $*   (first run downloads Python 3.11 + pinned deps; later runs start in ~1 s)"
rc=0
AUDITGYM_LAUNCHER="sh $SRC/run.sh" AUDITGYM_SRC="$SRC" "$UV" run --quiet --frozen --extra gym --project "$SRC" auditgym "$@" || rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
  say "if this was a download error: just run the same command again (finished downloads are kept), or use Docker (see README)"
fi
exit "$rc"

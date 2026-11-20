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
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 --speed-limit 1000 --speed-time 60 "$1"
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
  say "downloading AuditGym ($REF) into $SRC ..."
  TGZ="$SRC.$$.tgz"; NEW="$SRC.new.$$"; rm -rf "$TGZ" "$NEW"
  case "$URL" in http*) fetch "$URL" > "$TGZ" || true ;; *) cp "$URL" "$TGZ" ;; esac
  if [ -f "$SRC/.tarball.cksum" ] && [ "$(cksum < "$TGZ")" = "$(cat "$SRC/.tarball.cksum")" ]; then
    rm -f "$TGZ"   # same version as last time: keep the tree and its environment (no rebuild, works offline)
  else
    mkdir -p "$NEW"
    if ! tar -xzf "$TGZ" --strip-components=1 -C "$NEW" 2>/dev/null; then
      rm -rf "$NEW" "$TGZ"; say "download failed or was incomplete; nothing changed - run the same command again"; exit 2
    fi
    cksum < "$TGZ" > "$NEW/.tarball.cksum"; rm -f "$TGZ"
    mv "$SRC" "$SRC.old.$$" 2>/dev/null || true                        # a fresh tree: no stale files from older versions
    if [ -e "$SRC" ]; then rm -rf "$NEW"; else mv "$NEW" "$SRC"; fi    # a concurrent run already swapped: use its tree
    rm -rf "$SRC.old.$$"
  fi
else
  SRC="$AUDITGYM_SRC"
fi

UV="${AUDITGYM_UV:-}"
[ -z "$UV" ] && command -v uv >/dev/null 2>&1 && UV="$(command -v uv)"
[ -z "$UV" ] && [ -x "$HOME_DIR/bin/uv" ] && "$HOME_DIR/bin/uv" --version >/dev/null 2>&1 && UV="$HOME_DIR/bin/uv"  # must start
if [ -z "$UV" ]; then
  case "$(uname -m)" in x86_64|amd64) A=x86_64 ;; arm64|aarch64) A=aarch64 ;; *) say "unsupported CPU $(uname -m); use Docker"; exit 2 ;; esac
  if [ "$(uname -s)" = "Darwin" ]; then T="$A-apple-darwin"; else T="$A-unknown-linux-musl"; fi
  case "$T" in  # sha256 of the release assets (as pinned in https://astral.sh/uv/0.12.17/install.sh)
    aarch64-apple-darwin) S=85f00cbdc6dd3e97eba4c31b4d014375a9fdfe8f570023b84e5102fc3456896b ;;
    x86_64-apple-darwin) S=8dcf05a8c809bb3c471d2b614788ba27a6e41298fc8c31ac84b5f4339fd468e5 ;;
    aarch64-unknown-linux-musl) S=a6096da273d548cb9f277d237a01ac7344a39ef0f455c0e148e4dc9737c1596b ;;
    x86_64-unknown-linux-musl) S=6401c4665d8fa2a9893e087c91f585430738e3170f5398a1141483efb4a93310 ;;
  esac
  say "downloading uv 0.12.17 ($T, static binary, ~20 MB) into $HOME_DIR/bin (one time) ..."
  TMP="$HOME_DIR/bin/.dl.$$"; rm -rf "$TMP"; mkdir -p "$TMP"
  fetch "https://github.com/astral-sh/uv/releases/download/0.12.17/uv-$T.tar.gz" > "$TMP/uv.tgz" || true
  GOT=$( (sha256sum "$TMP/uv.tgz" 2>/dev/null || shasum -a 256 "$TMP/uv.tgz") | cut -d' ' -f1)
  if [ "$GOT" != "$S" ] || ! tar -xzf "$TMP/uv.tgz" --strip-components=1 -C "$TMP" 2>/dev/null; then
    rm -rf "$TMP"; say "could not download uv (network drop?); nothing was installed - run the same command again, or use Docker (see README)"; exit 2
  fi
  mv -f "$TMP/uv" "$HOME_DIR/bin/uv"; rm -rf "$TMP"   # atomic: the cached uv is always complete and verified
  UV="$HOME_DIR/bin/uv"
  [ -x "$UV" ] || { say "uv download incomplete: run the same command again"; exit 2; }
fi

say "running: auditgym $*   (first run downloads Python 3.11 + pinned deps; later runs start in ~1 s)"
rc=0
AUDITGYM_LAUNCHER="sh '$SRC/run.sh'" AUDITGYM_SRC="$SRC" "$UV" run --quiet --frozen --extra gym --project "$SRC" auditgym "$@" || rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
  say "if this was a download error: just run the same command again (finished downloads are kept), or use Docker (see README)"
fi
exit "$rc"

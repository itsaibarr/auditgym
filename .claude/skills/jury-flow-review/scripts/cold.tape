# A real first-time juror: empty HOME (no npm, uv or Python caches), only node on PATH, uv may not reuse any installed
# Python (only-managed), the README one-liner, nothing pre-warmed.
Output cold.mp4
Output cold.txt
Set Shell "bash"
Set FontSize 16
Set Width 1920
Set Height 1080
Set Padding 20
Set TypingSpeed 18ms
Set Theme "Catppuccin Mocha"
Set WaitTimeout 12m
Set WaitPattern /\$\s*$/
Hide
Type `export PS1='$ ' && rm -rf /tmp/jury-cold && mkdir -p /tmp/jury-cold/home /tmp/jury-cold/work && export HOME=/tmp/jury-cold/home PATH="$(dirname "$(command -v node)"):/usr/bin:/bin" UV_PYTHON_PREFERENCE=only-managed && cd /tmp/jury-cold/work && clear`
Enter
Wait
Show
Type `echo "[auditgym] fetching the launcher (7 KB) ..."; curl -fsSL https://raw.githubusercontent.com/itsaibarr/auditgym/master/run.sh | sh`
Enter
Wait
Sleep 10s

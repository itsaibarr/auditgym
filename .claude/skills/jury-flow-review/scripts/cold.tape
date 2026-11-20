# A real first-time juror: empty HOME (no npm, uv or Python caches), the README one-liner, nothing pre-warmed.
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
Type "export PS1='$ ' && rm -rf /tmp/jury-cold && mkdir -p /tmp/jury-cold/home /tmp/jury-cold/work && export HOME=/tmp/jury-cold/home && cd /tmp/jury-cold/work && clear"
Enter
Show
Type "npx -y https://codeload.github.com/itsaibarr/auditgym/tar.gz/master"
Enter
Wait
Sleep 10s

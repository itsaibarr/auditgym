# AuditGym demo (VHS): the jury path end to end, on the public repo.
# Render: cd <repo> && vhs submission/demo.tape   -> submission/demo.mp4
# Needs: node (npx), docker image auditgym:1.0.0 (docker build -t auditgym:1.0.0 .), openssl.
# The uv/Python cache (~/.cache/auditgym) is warmed before recording; a first run adds 1-3 min of downloads.
Output submission/demo.mp4

Require npx
Require docker
Require openssl

Set Shell "bash"
Set FontSize 16
Set Width 1920
Set Height 1080
Set Padding 20
Set TypingSpeed 18ms
Set Theme "Catppuccin Mocha"
Set WaitTimeout 4m
Set WaitPattern /\$\s*$/

Hide
Type "export PS1='$ ' DOCKER_CLI_HINTS=false && rm -rf /tmp/jury && mkdir -p /tmp/jury/strict && cd /tmp/jury && clear"
Enter
Show

Type "# AuditGym: an RL environment and grader for AI financial auditors. Genesis Hackathon FinTech 2026, team tryHard"
Enter
Type "# A juror runs one line from the README. No Python, git, keys or GPU needed. (A first run spends 1-3 min downloading; cached here.)"
Enter
Sleep 2s
Type "npx -y https://codeload.github.com/itsaibarr/auditgym/tar.gz/master"
Enter
Wait
Sleep 14s

Type "clear"
Enter
Type "# The tour left a report folder: report.md / report.json (scorecard), check/ (one line per check), closed/ (runs to re-check)"
Enter
Type "ls auditgym-report auditgym-report/closed"
Enter
Wait
Sleep 4s
Type "# From here on, auditgym means the same launch command"
Enter
Type "alias auditgym='npx -y --prefer-offline https://codeload.github.com/itsaibarr/auditgym/tar.gz/master'"
Enter
Type "# The first 'next step' from the tour: replay its hidden run and grade it again"
Enter
Type "AUDITGYM_SALT=$(cat auditgym-report/closed/SALT) auditgym replay auditgym-report/closed/reconcile --split - < auditgym-report/closed/split.json | tail -3"
Enter
Wait
Sleep 6s

Type "clear"
Enter
Type "# Your own hidden split (random seeds, organizers' format), then the case's 4 reference policies on it"
Enter
Type "auditgym split > my_split.json && auditgym run --policy random,reconcile,flag_all,oracle --split - --public-salt --out out < my_split.json"
Enter
Wait
Sleep 10s

Type "clear"
Enter
Type "# Compare the runs, with found / false alarms / missed for each error type"
Enter
Type "auditgym compare out/random out/reconcile out/flag_all out/oracle --per-type"
Enter
Wait
Sleep 12s

Type "clear"
Enter
Type "# Test your own agent: a Python class with reset/step (example in examples/my_policy.py)"
Enter
Type "auditgym run --policy my_policy:LookAround --policy-path examples --split dev --out out --trust-policies"
Enter
Wait
Sleep 8s

Type "clear"
Enter
Type "# Strict closed evaluation in Docker, no network: your own secret salt, split on stdin; the agent sees neither seeds nor salt"
Enter
Type "SALT=$(openssl rand -hex 32)"
Enter
Type "docker run -i --rm --network none -e AUDITGYM_SALT=$SALT -v $PWD/strict:/out auditgym:1.0.0 run --policy expert,reconcile,flag_all,oracle --split - --out /out < my_split.json"
Enter
Wait
Sleep 10s
Type "# A salt-leak attempt through PID 1 (docker --init): the runner refuses to start"
Enter
Type "docker run -i --rm --init --network none -e AUDITGYM_SALT=$SALT auditgym:1.0.0 run --policy expert --split - --out /tmp/x < my_split.json; echo exit=$?"
Enter
Wait
Sleep 8s
Type "# More: README, docs/DESIGN.md, docs/FAILURE_ANALYSIS.md, docs/COUNCIL.md. Thank you!"
Enter
Sleep 6s

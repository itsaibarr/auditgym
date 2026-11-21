# AuditGym demo (VHS): the jury tour, then the case's own checking process (section 06) step by step, then your own agent.
# Render: cd <repo> && vhs submission/demo.tape   -> submission/demo.mp4
# Needs: curl, docker image auditgym:1.0.0 (docker build -t auditgym:1.0.0 .), openssl.
# Recorded with a neutral HOME (/tmp/jury/home), warmed up off camera; a real first run adds a 20-60 s one-time setup.
Output submission/demo.mp4

Require curl
Require docker
Require openssl

Set Shell "bash"
Set FontSize 16
Set Width 1920
Set Height 1080
Set Padding 20
Set TypingSpeed 15ms
Set Theme "Catppuccin Mocha"
Set WaitTimeout 4m
Set WaitPattern /\$\s*$/

Hide
Type `export PS1='$ ' DOCKER_CLI_HINTS=false DOCKER_CONFIG="$HOME/.docker" && rm -rf /tmp/jury && mkdir -p /tmp/jury/out /tmp/jury/home && export HOME=/tmp/jury/home && cd /tmp/jury && curl -fsSL https://raw.githubusercontent.com/itsaibarr/auditgym/master/run.sh | sh -s -- --version >/dev/null 2>&1; clear`
Enter
Wait
Show

Type "# AuditGym: an RL environment and grader for AI financial auditors. Genesis Hackathon FinTech 2026, team tryHard"
Enter
Type "# 1) The jury tour: one line from the README (it downloads only itself; every check runs offline), no keys or GPU. Ends with a scorecard."
Enter
Sleep 2s
Type `echo "[auditgym] fetching the launcher (7 KB) ..."; curl -fsSL https://raw.githubusercontent.com/itsaibarr/auditgym/master/run.sh | sh`
Enter
Wait
Sleep 12s

Type "clear"
Enter
Type "# 2) The case's own checking process (section 06): the case's policies on hidden seeds you choose, each step one command."
Enter
Type "# D = an offline container with the case's limits (2 CPU, 4 GB) and a secret salt only the runner sees"
Enter
Type "docker run --rm --network none auditgym:1.0.0 split > my_split.json && SALT=$(openssl rand -hex 32)"
Enter
Wait
Type `D="docker run -i --rm --network none --cpus 2 --memory 4g -e AUDITGYM_SALT=$SALT"`
Enter
Type "# Our adapters of the case's four policies on 20 hidden episodes: random (fixed seed), simple reconciliation, flag everything, oracle"
Enter
Type "$D -v $PWD/out:/out auditgym:1.0.0 run --policy random,reconcile,flag_all,oracle --policy-arg seed=7 --split - --out /out < my_split.json"
Enter
Wait
Sleep 10s

Type "clear"
Enter
Type "# Data and answer key, checked from the records by code that does not use the generator"
Enter
Type "$D auditgym:1.0.0 datacheck --split - < my_split.json"
Enter
Wait
Sleep 10s

Type "clear"
Enter
Type "# 20 paired checks: rows shuffled, document/line IDs renamed; the oracle must stay at 1.0 (we also show if expert's answer changes)"
Enter
Type "$D auditgym:1.0.0 paired --split - < my_split.json"
Enter
Wait
Sleep 10s

Type "clear"
Enter
Type "# Exact replay of the recorded run (re-graded), then a scan of everything the policies saw for leaked answers"
Enter
Type "$D -v $PWD/out:/out:ro auditgym:1.0.0 replay /out/reconcile --split - < my_split.json | tail -2"
Enter
Wait
Type "docker run --rm --network none -v $PWD/out:/out:ro auditgym:1.0.0 leakscan /out/random /out/reconcile /out/flag_all /out/oracle"
Enter
Wait
Type "# Attack check: if any other process in the container can read the secret salt (here docker --init), the runner refuses"
Enter
Type "docker run -i --rm --init --network none -e AUDITGYM_SALT=$SALT auditgym:1.0.0 run --policy expert --split - --out /tmp/x < my_split.json; echo exit=$?"
Enter
Wait
Sleep 8s

Type "clear"
Enter
Type "# 3) Your own agent: init copies an example to edit. This toy one accuses nobody: +1 on clean books, -1 on the rest."
Enter
Type "# From here on, auditgym means the same one-line launch command"
Enter
Type "alias auditgym='sh ~/.cache/auditgym/src-master/run.sh'"
Enter
Type "auditgym init my-agent && auditgym run --policy my_policy:LookAround --policy-path my-agent --split dev --out out2 --trust-policies"
Enter
Wait
Sleep 9s
Type "# More: README (sections for the jury and for the organizers' process), docs/DESIGN.md, docs/FAILURE_ANALYSIS.md. Thank you!"
Enter
Sleep 6s

# AuditGym demo (VHS): the jury path end to end, on the public repo.
# Render: cd <repo> && vhs submission/demo.tape   -> submission/demo.mp4
# Needs: node (npx), docker image auditgym:1.0.0 (docker build -t auditgym:1.0.0 .), openssl.
# The uv/Python cache (~/.cache/auditgym) is warmed before recording; the first run would add 1-3 min of downloads.
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

Type "# AuditGym: RL-среда и оценщик для ИИ-аудиторов. Genesis Hackathon FinTech 2026, команда tryHard"
Enter
Type "# Жюри запускает одну строку из README. Python, git, ключи и GPU не нужны. Первый запуск качает uv и Python за 1-3 минуты, здесь кеш уже есть"
Enter
Sleep 2s
Type "npx -y https://codeload.github.com/itsaibarr/auditgym/tar.gz/master"
Enter
Wait
Sleep 12s

Type "clear"
Enter
Type "# Отчёт тура: итог по пяти критериям кейса, таблица политик и прогоны для повторной проверки"
Enter
Type "ls auditgym-report auditgym-report/closed && head -24 auditgym-report/report.md"
Enter
Wait
Sleep 10s

Type "clear"
Enter
Type "# Дальше auditgym означает ту же команду запуска"
Enter
Type "alias auditgym='npx -y --prefer-offline https://codeload.github.com/itsaibarr/auditgym/tar.gz/master'"
Enter
Type "# Первая команда из «Что дальше»: побайтный повтор закрытого прогона тура и перепроверка оценки"
Enter
Type "AUDITGYM_SALT=$(cat auditgym-report/closed/SALT) auditgym replay auditgym-report/closed/reconcile --split - < auditgym-report/closed/split.json | tail -4"
Enter
Wait
Sleep 5s
Type "# Свой закрытый split: новые случайные сиды в форме организаторов, на нём четыре политики организаторов"
Enter
Type "auditgym split > my_split.json && auditgym run --policy random,reconcile,flag_all,oracle --split - --public-salt --out out < my_split.json"
Enter
Wait
Sleep 9s

Type "clear"
Enter
Type "# Сравнение прогонов с разбивкой по типам ошибок"
Enter
Type "auditgym compare out/random out/reconcile out/flag_all out/oracle --per-type"
Enter
Wait
Sleep 10s

Type "clear"
Enter
Type "# Своя политика: класс Python с reset/step, пример лежит в examples/my_policy.py"
Enter
Type "auditgym run --policy my_policy:LookAround --policy-path examples --split dev --out out --trust-policies"
Enter
Wait
Sleep 7s

Type "clear"
Enter
Type "# Строгая закрытая проверка в Docker без сети: своя соль и split через stdin, политика не видит ни сидов, ни соли"
Enter
Type "SALT=$(openssl rand -hex 32)"
Enter
Type "docker run -i --rm --network none -e AUDITGYM_SALT=$SALT -v $PWD/strict:/out auditgym:1.0.0 run --policy expert,reconcile,flag_all,oracle --split - --out /out < my_split.json"
Enter
Wait
Sleep 8s
Type "# Попытка утечки соли через PID 1 (docker --init): раннер отказывается стартовать"
Enter
Type "docker run -i --rm --init --network none -e AUDITGYM_SALT=$SALT auditgym:1.0.0 run --policy expert --split - --out /tmp/x < my_split.json; echo exit=$?"
Enter
Wait
Sleep 8s
Type "# Подробности: README, docs/DESIGN.md, docs/FAILURE_ANALYSIS.md, docs/COUNCIL.md. Спасибо!"
Enter
Sleep 6s

# AuditGym demo (VHS). Render: cd <repo> && vhs submission/demo.tape   -> submission/demo.mp4
# Needs: docker image auditgym:1.0.0 (docker build -t auditgym:1.0.0 .), openssl.
Output submission/demo.mp4

Set Shell "bash"
Set FontSize 17
Set Width 1600
Set Height 900
Set Padding 24
Set TypingSpeed 25ms
Set Theme "Catppuccin Mocha"

Hide
Type "export PS1='$ ' DOCKER_CLI_HINTS=false && rm -rf /tmp/agdemo && mkdir -p /tmp/agdemo/out /tmp/agdemo/closed && cp configs/closed_example.json /tmp/agdemo/closed_split.json && clear"
Enter
Show

Type "# AuditGym — воспроизводимая RL-среда и оценщик для ИИ-аудиторов (Genesis Hackathon FinTech 2026, команда tryHard)"
Enter
Sleep 1.5s
Type "# Шаг 1. Одна офлайн-команда: тесты, базовые политики, порядки наград, leakscan, парные проверки, replay"
Enter
Type "docker run --rm --network none --cpus 2 --memory 4g auditgym:1.0.0 check 2>&1 | tail -16"
Enter
Sleep 48s

Type "clear"
Enter
Type "# Шаг 2. Пять базовых политик на dev: oracle = 1, expert без эталона ≈ 0.95, наивная сверка < 0, flag_all = −1"
Enter
Type "docker run --rm --network none -v /tmp/agdemo/out:/out auditgym:1.0.0 run --policy oracle,expert,reconcile,flag_all,random --split dev --out /out 2>/dev/null"
Enter
Sleep 12s

Type "clear"
Enter
Type "# Шаг 3. Закрытая проверка: секретная соль (≥ 32 символов) и split через stdin — политика не видит ни сидов, ни соли"
Enter
Type "SALT=$(openssl rand -hex 32)"
Enter
Type "docker run -i --rm --network none -e AUDITGYM_SALT=$SALT -v /tmp/agdemo/closed:/out auditgym:1.0.0 run --policy expert,reconcile,flag_all,oracle --split - --out /out < /tmp/agdemo/closed_split.json 2>/dev/null"
Enter
Sleep 16s

Type "# Попытка утечки соли через PID 1 (docker --init) — раннер отказывается стартовать"
Enter
Type "docker run -i --rm --init --network none -e AUDITGYM_SALT=$SALT auditgym:1.0.0 run --policy expert --split - --out /tmp/x < /tmp/agdemo/closed_split.json; echo exit=$?"
Enter
Sleep 9s

Type "clear"
Enter
Type "# Шаг 4. Replay закрытого прогона: побайтный повтор каждого шага, перепроверка оценки и подписи run.json"
Enter
Type "docker run -i --rm --network none -e AUDITGYM_SALT=$SALT -v /tmp/agdemo/closed:/out:ro auditgym:1.0.0 replay /out/expert --split - < /tmp/agdemo/closed_split.json | tail -4"
Enter
Sleep 14s
Type "docker run --rm --network none -e AUDITGYM_SALT=$SALT -v /tmp/agdemo/closed:/out:ro auditgym:1.0.0 compare /out/expert /out/reconcile /out/flag_all --per-type 2>/dev/null"
Enter
Sleep 8s
Type "# Репозиторий: README (RU), docs/DESIGN.md, docs/FAILURE_ANALYSIS.md, docs/COUNCIL.md. Спасибо!"
Enter
Sleep 4s

# Измеренный разбор сбоев (failure analysis)

Сгенерировано `uv run python scripts/failure_analysis.py` по трассам сопоставления (`matches` в results.jsonl). Все политики работают в eval-режиме в изолированных дочерних процессах (oracle — привилегированно в раннере).

## open train+dev (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.000 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.956 | +0.953 | +0.956 | 23.1 | 0.0 | 0.060 |
| reconcile | 20 | 0.136 | 0.589 | 0.225 | -0.583 | -1.000 | -0.479 | 23.1 | 1.0 | 0.061 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.6 | 1.0 | 0.058 |
| random | 20 | 0.000 | 0.000 | 0.050 | -0.910 | -0.550 | -1.000 | 79.8 | 0.75 | 0.062 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | no_issue: 210 | duplicate_invoice: 163, fx_conversion: 6, payment_allocation: 26, period_cutoff: 15 | fx_conversion: 14, payment_allocation: 4, period_cutoff: 5 | submitted: 20 |
| flag_all | no_issue: 600 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 13, fx_conversion: 16, payment_allocation: 14, period_cutoff: 13 | submitted: 20 |
| random | no_issue: 23 | duplicate_invoice: 6, fx_conversion: 4, payment_allocation: 9, period_cutoff: 4 | duplicate_invoice: 13, fx_conversion: 16, payment_allocation: 14, period_cutoff: 13 | step_limit: 11, submitted: 9 |

## closed-shaped new seeds (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.000 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.960 | +0.962 | +0.959 | 21.1 | 0.0 | 0.061 |
| reconcile | 20 | 0.212 | 0.721 | 0.296 | -0.439 | -1.000 | -0.252 | 21.1 | 1.0 | 0.058 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.4 | 1.0 | 0.053 |
| random | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 58.7 | 1.0 | 0.058 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | no_issue: 182 | duplicate_invoice: 128, fx_conversion: 8, payment_allocation: 28, period_cutoff: 18 | fx_conversion: 13, payment_allocation: 3, period_cutoff: 3 | submitted: 20 |
| flag_all | no_issue: 600 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 14, fx_conversion: 18, payment_allocation: 18, period_cutoff: 18 | submitted: 20 |
| random | no_issue: 32 | duplicate_invoice: 13, fx_conversion: 7, payment_allocation: 8, period_cutoff: 4 | duplicate_invoice: 14, fx_conversion: 18, payment_allocation: 18, period_cutoff: 18 | step_limit: 4, submitted: 16 |

## Интерпретация

- **oracle = 1.000 везде**: эталон достижим, у каждого вывода есть полный допустимый набор доказательств (проверка максимума награды, не способности агента).
- **expert** (без доступа к эталону, только наблюдения + опубликованная политика) находит все ошибки: задача решаема по видимым записям; потери только на стоимости шагов (полное чтение таблиц ≈ 20–45 шагов → −0.04…−0.09).
- **reconcile** (наивная сверка по строкам дат, без часовых поясов, без учёта переноса курса) получает много FP: `no_issue` по `duplicate_invoice` — это отвлекающие повторяющиеся подписки с одинаковой суммой; по `payment_allocation` — переплаты с возвратом (2300) и платежи с комиссией; по `period_cutoff` — документы, записанные в UTC около полуночи; по `fx_conversion` — дата курса взята из строки времени без перевода в зону компании. Поэтому на чистых книгах reconcile = −1: отвлекающие случаи действительно ловят простые трюки.
- **flag_all** = −1 на всех эпизодах: массовые обвинения не окупаются (сумма, ID и доказательства должны совпасть).
- **random** почти никогда не отправляет ответ → усечение на 100 шагах: −1 на книгах с ошибками и 0.8 на чистых (см. ограничения).

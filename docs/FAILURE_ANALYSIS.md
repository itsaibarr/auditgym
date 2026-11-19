# Измеренный разбор сбоев (failure analysis)

Сгенерировано `uv run python scripts/failure_analysis.py` по трассам сопоставления (`matches` в results.jsonl). Все политики работают в eval-режиме в изолированных дочерних процессах (oracle — привилегированно в раннере).

## open train+dev (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.000 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.956 | +0.957 | +0.956 | 22.8 | 0.0 | 0.068 |
| reconcile | 20 | 0.130 | 0.611 | 0.218 | -0.595 | -1.000 | -0.421 | 22.8 | 1.0 | 0.061 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.5 | 1.0 | 0.055 |
| random | 20 | 0.000 | 0.000 | 0.050 | -0.910 | -0.700 | -1.000 | 72.5 | 0.8333 | 0.067 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | no_issue: 220 | duplicate_invoice: 162, fx_conversion: 7, payment_allocation: 36, period_cutoff: 15 | fx_conversion: 13, payment_allocation: 4, period_cutoff: 4 | submitted: 20 |
| flag_all | no_issue: 600 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 13, fx_conversion: 15, payment_allocation: 13, period_cutoff: 13 | submitted: 20 |
| random | no_issue: 26 | duplicate_invoice: 6, fx_conversion: 6, payment_allocation: 6, period_cutoff: 8 | duplicate_invoice: 13, fx_conversion: 15, payment_allocation: 13, period_cutoff: 13 | step_limit: 10, submitted: 10 |

## closed-shaped new seeds (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.000 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.961 | +0.962 | +0.960 | 20.7 | 0.0 | 0.067 |
| reconcile | 20 | 0.223 | 0.750 | 0.327 | -0.375 | -1.000 | -0.167 | 20.7 | 1.0 | 0.057 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.3 | 1.0 | 0.066 |
| random | 20 | 0.000 | 0.000 | 0.150 | -0.730 | +0.080 | -1.000 | 70.3 | 0.4 | 0.064 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | no_issue: 178 | duplicate_invoice: 129, fx_conversion: 6, payment_allocation: 25, period_cutoff: 18 | fx_conversion: 11, payment_allocation: 3, period_cutoff: 3 | submitted: 20 |
| flag_all | no_issue: 600 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 14, fx_conversion: 18, payment_allocation: 18, period_cutoff: 18 | submitted: 20 |
| random | no_issue: 24 | duplicate_invoice: 4, fx_conversion: 5, payment_allocation: 9, period_cutoff: 6 | duplicate_invoice: 14, fx_conversion: 18, payment_allocation: 18, period_cutoff: 18 | step_limit: 7, submitted: 13 |

## Интерпретация

- **oracle = 1.000 везде**: эталон достижим, у каждого вывода есть полный допустимый набор доказательств (проверка максимума награды, не способности агента).
- **expert** (без доступа к эталону, только наблюдения + опубликованная политика) находит все ошибки: задача решаема по видимым записям; потери только на стоимости шагов (полное чтение таблиц ≈ 20–45 шагов → −0.04…−0.09).
- **reconcile** (наивная сверка по строкам дат, без часовых поясов, без учёта переноса курса) получает много FP: `no_issue` по `duplicate_invoice` — это отвлекающие повторяющиеся подписки с одинаковой суммой; по `payment_allocation` — переплаты с возвратом (2300) и платежи с комиссией; по `period_cutoff` — документы, записанные в UTC около полуночи; по `fx_conversion` — дата курса взята из строки времени без перевода в зону компании. Поэтому на чистых книгах reconcile = −1: отвлекающие случаи действительно ловят простые трюки.
- **flag_all** = −1 на всех эпизодах: массовые обвинения не окупаются (сумма, ID и доказательства должны совпасть).
- **random** почти никогда не отправляет ответ → усечение на 100 шагах: −1 на книгах с ошибками и 0.8 на чистых (см. ограничения).

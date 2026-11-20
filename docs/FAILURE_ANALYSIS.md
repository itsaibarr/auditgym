# Измеренный разбор сбоев (failure analysis)

Сгенерировано `uv run python scripts/failure_analysis.py` по трассам сопоставления (`matches` в results.jsonl). Все политики работают в eval-режиме в изолированных дочерних процессах (oracle — привилегированно в раннере).

## open train+dev (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.003 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.956 | +0.954 | +0.957 | 22.9 | 0.0 | 0.076 |
| reconcile | 20 | 0.113 | 0.509 | 0.191 | -0.649 | -1.000 | -0.561 | 22.9 | 1.0 | 0.064 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.5 | 1.0 | 0.055 |
| random | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 72.5 | 1.0 | 0.079 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | insufficient_evidence: 6, no_issue: 214 | duplicate_invoice: 161, fx_conversion: 6, payment_allocation: 32, period_cutoff: 21 | fx_conversion: 13, payment_allocation: 4, period_cutoff: 10 | submitted: 20 |
| flag_all | amount: 2, no_issue: 598 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 13, fx_conversion: 15, payment_allocation: 14, period_cutoff: 13 | submitted: 20 |
| random | no_issue: 26 | duplicate_invoice: 6, fx_conversion: 6, payment_allocation: 6, period_cutoff: 8 | duplicate_invoice: 13, fx_conversion: 15, payment_allocation: 14, period_cutoff: 13 | step_limit: 10, submitted: 10 |

## closed-shaped new seeds (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.003 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.961 | +0.962 | +0.960 | 20.7 | 0.0 | 0.068 |
| reconcile | 20 | 0.192 | 0.603 | 0.265 | -0.499 | -1.000 | -0.332 | 20.7 | 1.0 | 0.060 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.3 | 1.0 | 0.054 |
| random | 20 | 0.000 | 0.000 | 0.150 | -0.730 | +0.080 | -1.000 | 70.3 | 0.4 | 0.065 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | insufficient_evidence: 7, no_issue: 166 | duplicate_invoice: 128, fx_conversion: 6, payment_allocation: 14, period_cutoff: 25 | fx_conversion: 14, payment_allocation: 3, period_cutoff: 10 | submitted: 20 |
| flag_all | amount: 1, no_issue: 599 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 14, fx_conversion: 18, payment_allocation: 18, period_cutoff: 18 | submitted: 20 |
| random | no_issue: 24 | duplicate_invoice: 4, fx_conversion: 5, payment_allocation: 9, period_cutoff: 6 | duplicate_invoice: 14, fx_conversion: 18, payment_allocation: 18, period_cutoff: 18 | step_limit: 7, submitted: 13 |

## Интерпретация

- **oracle = 1.000 везде**: эталон достижим, у каждого вывода есть полный допустимый набор доказательств (проверка максимума награды, не способности агента).
- **expert** (без доступа к эталону, только наблюдения + опубликованная политика) находит все ошибки: задача решаема по видимым записям; потери только на стоимости шагов (полное чтение таблиц ≈ 20–45 шагов → −0.04…−0.09).
- **reconcile** (наивная сверка по строкам дат, без часовых поясов, без учёта переноса курса) получает много FP: `no_issue` по `duplicate_invoice` — это отвлекающие повторяющиеся подписки с одинаковой суммой; по `payment_allocation` — переплаты с возвратом (2300) и платежи с комиссией; по `period_cutoff` — документы, записанные в UTC около полуночи; по `fx_conversion` — дата курса взята из строки времени без перевода в зону компании. Поэтому на чистых книгах reconcile = −1: отвлекающие случаи действительно ловят простые трюки.
- **flag_all** = −1 на всех эпизодах: массовые обвинения не окупаются (сумма, ID и доказательства должны совпасть).
- **random** с вероятностью 1% за шаг отправляет случайные выводы (всегда FP → −1), иначе упирается в лимит 100 шагов: −1 на книгах с ошибками и 0.8 на чистых (см. ограничения).

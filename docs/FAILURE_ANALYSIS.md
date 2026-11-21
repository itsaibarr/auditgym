# Измеренный разбор сбоев (failure analysis)

Сгенерировано `uv run python scripts/failure_analysis.py` по трассам сопоставления (`matches` в results.jsonl). Все политики работают в eval-режиме в изолированных дочерних процессах (oracle — привилегированно в раннере).

Колонки: P и R посчитаны по всем выводам вместе (micro); F1 — среднее по эпизодам, и пустой ответ на чистых книгах даёт F1 = 1, поэтому F1 бывает больше нуля при P = R = 0; return — средняя терминальная награда, отдельно на чистых книгах и на книгах с ошибками; steps — среднее число шагов; clean-FP rate — доля чистых эпизодов, где политика кого-то обвинила; s/episode — время на эпизод по часам, оно зависит от машины.

## open train+dev (20)

| policy | episodes | P | R | F1 | return | return (clean) | return (with errors) | steps | clean-FP rate | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.002 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.956 | +0.954 | +0.957 | 22.9 | 0.0 | 0.036 |
| reconcile | 20 | 0.113 | 0.509 | 0.191 | -0.649 | -1.000 | -0.561 | 22.9 | 1.0 | 0.031 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.5 | 1.0 | 0.027 |
| random | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 72.5 | 1.0 | 0.034 |

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
| oracle | 20 | 1.000 | 1.000 | 1.000 | +1.000 | +1.000 | +1.000 | 1.0 | 0.0 | 0.001 |
| expert | 20 | 1.000 | 1.000 | 1.000 | +0.963 | +0.962 | +0.963 | 19.8 | 0.0 | 0.035 |
| reconcile | 20 | 0.178 | 0.548 | 0.271 | -0.486 | -1.000 | -0.314 | 19.8 | 1.0 | 0.031 |
| flag_all | 20 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 4.3 | 1.0 | 0.027 |
| random | 20 | 0.000 | 0.000 | 0.150 | -0.730 | +0.080 | -1.000 | 77.3 | 0.4 | 0.035 |

| policy | FP по причинам | FP по типам | FN по типам | завершение |
|---|---|---|---|---|
| oracle | — | — | — | submitted: 20 |
| expert | — | — | — | submitted: 20 |
| reconcile | insufficient_evidence: 8, no_issue: 204 | duplicate_invoice: 155, fx_conversion: 6, payment_allocation: 27, period_cutoff: 24 | fx_conversion: 20, payment_allocation: 5, period_cutoff: 13 | submitted: 20 |
| flag_all | no_issue: 600 | duplicate_invoice: 160, fx_conversion: 140, payment_allocation: 160, period_cutoff: 140 | duplicate_invoice: 24, fx_conversion: 23, payment_allocation: 18, period_cutoff: 19 | submitted: 20 |
| random | no_issue: 23 | duplicate_invoice: 8, fx_conversion: 4, payment_allocation: 6, period_cutoff: 5 | duplicate_invoice: 24, fx_conversion: 23, payment_allocation: 18, period_cutoff: 19 | step_limit: 8, submitted: 12 |

## Интерпретация

- **oracle = 1.000 везде**: эталон достижим, у каждого вывода есть полный допустимый набор доказательств (проверка максимума награды, не способности агента).
- **expert** (без доступа к эталону, только наблюдения + опубликованная политика) находит все ошибки: задача решаема по видимым записям; потери только на стоимости шагов (полное чтение таблиц ≈ 20–45 шагов → −0.04…−0.09).
- **reconcile** (наивная сверка по строкам дат, без часовых поясов, без учёта переноса курса) получает много FP: `no_issue` по `duplicate_invoice` — это отвлекающие повторяющиеся подписки с одинаковой суммой; по `payment_allocation` — переплаты с возвратом (2300) и платежи с комиссией; по `period_cutoff` — документы, записанные в UTC около полуночи; по `fx_conversion` — дата курса взята из строки времени без перевода в зону компании. Поэтому на чистых книгах reconcile = −1: отвлекающие случаи действительно ловят простые трюки.
- **flag_all** = −1 на всех эпизодах: массовые обвинения не окупаются (сумма, ID и доказательства должны совпасть).
- **random** с вероятностью 1% за шаг отправляет случайные выводы (всегда FP → −1), иначе упирается в лимит 100 шагов: −1 на книгах с ошибками и 0.8 на чистых (см. ограничения).

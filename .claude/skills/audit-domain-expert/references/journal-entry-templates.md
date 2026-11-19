# Journal-entry templates (Dr/Cr), worked in integer minor units

Base/functional currency **KZT (exp 2)**. Every number below was computed with Python `Decimal`, ROUND_HALF_EVEN, and one conversion per document:

`base_minor = round(doc_minor * Decimal(rate) * 10**(e_base - e_doc))`

Rate is in KZT per 1 unit of the source currency. Dates are company-local (Asia/Almaty, +05). 2026-10-31 is a Saturday.

## Chart of accounts (minimal)

| Code | Account | Code | Account |
|---|---|---|---|
| 1000 | Bank-KZT | 3000 | Equity / opening balance |
| 1010 | Bank-USD (valued in base) | 4000 | Revenue |
| 1020 | Bank-EUR (valued in base) | 4100 | Sales returns & allowances |
| 1100 | Accounts receivable (control) | 4200 | Sales discounts |
| 1200 | Accrued revenue / contract asset | 6000 | Operating expenses |
| 2100 | Accrued liabilities | 6500 | Bank charges |
| 2150 | Customer advances / unapplied cash | 7100 | Realized FX gain |
| 2300 | Customer refunds payable | 7200 | Realized FX loss |
| | | 7150 | Unrealized FX (only if revaluation on) |

Ledger row conventions:
- `currency` is always the base currency. `invoice_id` and `payment_id` link rows to documents, and a missing link is an explicit `null`.
- One JE = consecutive `row_id`s `JE{n}-{line}`, renamed by the seeded bijection for paired checks.

## Templates

**T0 Opening balance** (period start)
| Dr | Cr |
|---|---|
| 1000 Bank-KZT 300000000 | 3000 Equity 347825000 |
| 1100 AR 47825000 (prior-period USD invoice 1,000.00 carried at 478.25) | |

**T1 Invoice issued**: USD 1,000.00 (100000) on 2026-10-12 at `478.25`
| Dr | Cr |
|---|---|
| 1100 AR 47825000 | 4000 Revenue 47825000 |

**T2 Full receipt with realized FX** (IAS 21.28). The payment is USD 1,000.00.
- Gain, on 2026-10-27 at `482.10`:

| Dr | Cr |
|---|---|
| 1010 Bank-USD 48210000 | 1100 AR 47825000 |
| | 7100 FX gain 385000 |

- Loss variant, at `475.60`:

| Dr | Cr |
|---|---|
| 1010 Bank-USD 47560000 | 1100 AR 47825000 |
| 7200 FX loss 265000 | |

**T3 Partial receipt**: USD 400.00 (40000) at `482.10`. AR is relieved at the carried rate.
| Dr | Cr |
|---|---|
| 1010 Bank-USD 19284000 | 1100 AR 19130000 |
| | 7100 FX gain 154000 |

Open after: USD 600.00 = 28695000 carried, and 19130000 + 28695000 = 47825000. On the final allocation, relieve the *remaining carrying amount*, not a fresh conversion, so AR clears to exactly 0.

**T4 One payment allocated to two invoices**
- Invoices: EUR A 2,000.00 and B 500.00, both on 2026-10-13 at `515.40`, giving A 103080000 and B 25770000.
- Payment: EUR 2,500.00 on 10-27 at `519.90`, refs [A, B].

| Dr | Cr |
|---|---|
| 1020 Bank-EUR 129975000 | 1100 AR (inv A) 103080000 |
| | 1100 AR (inv B) 25770000 |
| | 7100 FX gain 1125000 (A 900000 + B 225000) |

Each AR row carries the same `payment_id` and its own `invoice_id`.

**T5 On-account receipt** (`invoice_refs: null`): KZT 500,000.00
| Dr | Cr |
|---|---|
| 1000 Bank 50000000 | 2150 Unapplied cash 50000000 (`invoice_id` null) |

This is legitimate. It becomes E2 only when the doc *references* an invoice.

**T6 Overpayment, then refund**
- Receipt: invoice USD 1,000.00 at 478.25, paid with USD 1,050.00 at `482.10`.

| Dr | Cr |
|---|---|
| 1010 Bank-USD 50620500 | 1100 AR 47825000 |
| | 7100 FX gain 385000 |
| | 2300 Refunds payable 2410500 (USD 50 at 482.10) |

- Refund: USD 50.00 at `480.00`, settling a liability carried at 2410500.

| Dr | Cr |
|---|---|
| 2300 Refunds payable 2410500 | 1010 Bank-USD 2400000 |
| | 7100 FX gain 10500 |

**T7 Credit note**: KZT 50,000.00, with `doc_type: credit_note` and `ref_invoice_id`
| Dr | Cr |
|---|---|
| 4100 Sales returns 5000000 | 1100 AR 5000000 |

A foreign-currency CN is converted at **its own** `issued_at` date rate (policy). The difference from the invoice's carried amount stays in AR until settlement.

**T8 Cash refund with no prior overpayment** (goods returned after payment): KZT 12,000.00
| Dr | Cr |
|---|---|
| 4100 Sales returns 1200000 | 1000 Bank 1200000 |

**T9 Accrued revenue** (delivered in Oct, invoiced 11-03): KZT 2,000,000.00
- On 10-31:

| Dr | Cr |
|---|---|
| 1200 Accrued revenue 200000000 | 4000 Revenue 200000000 |

- On 11-03:

| Dr | Cr |
|---|---|
| 1100 AR 200000000 | 1200 Accrued revenue 200000000 |

This is the correct-cutoff twin of E4-late.

**T10 Accrued expense + auto-reversal**: KZT 350,000.00
- On 10-31:

| Dr | Cr |
|---|---|
| 6000 Expense 35000000 | 2100 Accrued liabilities 35000000 |

- On 11-01, the reversal:

| Dr | Cr |
|---|---|
| 2100 Accrued liabilities 35000000 | 6000 Expense 35000000 |

This is distractor D11.

**T11 Bank fee deducted from receipt**
- Invoice: KZT 250,000.00.
- Payment doc: `amount_minor` 24850000, plus a stated fee of 150000.

| Dr | Cr |
|---|---|
| 1000 Bank 24850000 | 1100 AR 25000000 |
| 6500 Bank charges 150000 | |

The fee must be stated on the doc, or this case cannot be told apart from a partial allocation.

**T12 Early-payment discount**: terms 2/10 net 30 on KZT 250,000.00, paid within 10 days
| Dr | Cr |
|---|---|
| 1000 Bank 24500000 | 1100 AR 25000000 |
| 4200 Sales discounts 500000 | |

**T13 Period-end revaluation** (IAS 21.23; **off in v1**): open USD 1,000.00 carried at 478.25, closing rate `480.00`
| Dr | Cr |
|---|---|
| 1100 AR 175000 | 7150 Unrealized FX 175000 |

## Exponent and rounding checks

| Case | Input | Result |
|---|---|---|
| JPY (exp 0) | 120000 at `3.1745` -> x10^2 | **38094000** |
| KWD (exp 3) | 1234567 at `1561.20` -> x10^-1 = 192740600.04 | **192740600** |
| Tie | USD 12.50 (1250) at `478.21` = 597762.5 | HALF_EVEN **597762**, HALF_UP 597763 |
| Per-line drift | 3 x USD 19.99 at `478.255` | per line 2868096, per total **2868095** |
| Inverse-rate trap | rate stored as KZT->USD `0.002091`, inverted | 47824008 instead of 47825000 (Reg. 1103/97 forbids this) |
| Truncated rate | `478` instead of `478.25` | 47800000 (delta 25000) |

## Errors: correct vs recorded JEs

**E1 duplicate_invoice**
- Correct: T1 for INV-1042 only (47825000).
- Recorded: T1 again for clone `INV1042` dated 10-14 at `479.10`: Dr 1100 47910000 / Cr 4000 47910000.
- Truth: {orig, dup}, **47910000** KZT.

**E2 payment_allocation** (partial; the payment refs [X], and X totals KZT 1,250,000.00)

| | Dr | Cr |
|---|---|---|
| Correct | 1000 125000000 | 1100 (X) 125000000 |
| Recorded | 1000 125000000 | 1100 (X) 80000000 ; 2150 45000000 |

- Truth: {pay, X}, **45000000**.
- Missing variant: the whole 125000000 goes to 2150.

**E3 fx_conversion** (invoice USD 2,400.00 issued 10-20, rate `479.80`)

| | Dr | Cr |
|---|---|---|
| Correct | 1100 115152000 | 4000 115152000 |
| Recorded at the 10-12 rate `478.25` | 1100 114780000 | 4000 114780000 |

- Truth: {inv}, **372000**.
- For a payment, Bank is the wrong row and 7100/7200 absorbs the delta.

**E4 period_cutoff** (premature)
- Invoice: KZT 3,400,000.00, `delivered_at` `2026-10-31T19:30:00Z` (= 11-01 00:30 local), `issued_at` 11-01.
- Recorded: Dr 1100 340000000 / Cr 4000 340000000 dated `2026-10-31T21:00:00+05:00`.
- Truth: {inv}, **340000000**.
- Clean twin: `delivered_at` `2026-10-31T23:50:00+05:00`, posted 10-31, so it is fine.

**X1 partial USD payment at the boundary**
- Facts:
  - Invoice: USD 1,000.00, 10-12 at 478.25.
  - Payment doc: USD 400.00, `2026-10-31T23:40:00+05:00` (Sat), refs [inv].
  - Rate date: Fri 10-30, `481.50`.

| | Dr | Cr |
|---|---|---|
| Correct (dated 10-31) | 1010 19260000 | 1100 19130000 ; 7100 130000 |
| Recorded (dated 11-02, rate `484.90`) | 1010 19396000 | 1100 7173750 (USD 150 carried) ; 7100 99750 ; 2150 12122500 |

| Linked finding | ids | amount | evidence |
|---|---|---|---|
| period_cutoff | {pay} | 19260000 (correct value) | pay doc, `fx:2026-10-30:USD:KZT`, Bank row |
| fx_conversion | {pay} | 136000 (19396000 - 19260000) | pay doc, `fx:2026-10-30:USD:KZT`, Bank row |
| payment_allocation | {pay, inv} | 11956250 (USD 250 x 478.25) | pay doc, inv doc, invoice-date fx row, the payment's AR (1100) row |

The alternative amounts 12037500 (USD 250 at 481.50) and 12122500 (the 2150 row as recorded) must fall outside the matcher tolerance.

**X2 foreign duplicate + credit note**
- Facts:
  - A: EUR 3,000.00, 10-05 at `512.30` = 153690000.
  - CN: EUR 500.00, 10-14, refs A.
  - Clone D `INV-731` (A is `INV-0731`): 10-07 at `513.00` = 153900000.

| | Dr | Cr |
|---|---|---|
| CN correct (rate `514.10`) | 4100 25705000 | 1100 25705000 |
| CN recorded (A's rate) | 4100 25615000 | 1100 25615000 |

| Linked finding | ids | amount | evidence |
|---|---|---|---|
| duplicate_invoice | {A, D} | 153900000 | A doc, D doc, D AR row |
| fx_conversion | {CN} | 90000 | CN doc, `fx:2026-10-14:EUR:KZT`, CN 4100 row |

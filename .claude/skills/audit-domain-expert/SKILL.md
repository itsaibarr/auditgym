---
name: audit-domain-expert
description: Financial-audit and accounting domain expert for AuditGym (RL environment for AI auditors). Use when building or reviewing the company generator, ledger/invoice/payment/fx data, posting templates, error injectors, ground truth (truth.json, provenance.json), evidence sets, distractors, FX/rounding or period-cutoff logic, or when acting as the "auditor" council reviewer scoring the project 0-10.
---

# Audit domain expert (AuditGym)

You are a statutory auditor who also writes the generator. Every number must be reproducible from visible records by a skeptical reviewer with a calculator. Case text: `research/case.pdf` (RU). Background research: `research/02-finance-audit-references.md`.

Settled decisions (do not relitigate): ledger amounts are in **base (functional) currency**, transaction currency lives on documents; interacting errors are recorded as **2-3 linked findings**, each matched separately.

## 1. Core principles

1. **Truth-first.** Seeded world of business events -> correct books (posting engine, templates T0-T13) -> recorded books = correct books + injected mutations. `truth.json` is *derived* from the provenance diff, never hand-written.
2. **Documents are true, the recorded ledger lies.** Injectors mutate ledger rows. The only exception is E1, where the spurious document itself is the error.
3. **Books stay balanced after injection.** Every injector edits whole journal entries (JEs) and rebalances through the account the real mistake would hit (7100/7200 FX, 2150 unapplied cash). A "trial balance does not balance" signal is a cheap trick, not an audit finding.
4. **A discrepancy is visible only doc<->ledger or subledger<->GL.** Each issue breaks a named invariant (I3, I5, I6, I7, I8) by exactly `expected_amount_minor`, and nothing else breaks.
5. **Positive evidence only.** Each issue is proved by records that exist and contradict each other, never by absence alone. Late cutoff therefore needs the ledger export to include a post-period window (`manifest.cutoff_window_days`).
6. **Sufficient and available (ISA 500).** Each acceptable evidence set is a *minimal* set of record IDs whose joint contents prove the discrepancy, given `policy.json`. Never require an unavailable fact: a missing fx row, an unpublished rule, a delivery date absent from the doc, or knowledge of the seed.
7. **One right answer.** Type, canonical `affected_ids`, amount and currency are unique. Every plausible alternative amount differs from the true one by more than 2x the matcher tolerance.
8. **Fictional only.** Use seeded synthetic names, IDs, rates and banks: no real companies, people, IBANs, cards, or real FX series. Use your own name lists, not Faker.

## 2. Invariants (assert before injection; recheck after)

| ID | Invariant | Broken by |
|---|---|---|
| I1 | Each JE balances in base: sum(debit_minor) = sum(credit_minor) | never (gate) |
| I2 | Trial balance nets to 0, both at period end and over the full export | never (gate) |
| I3 | AR subledger (invoices - credit notes - doc-stated allocations, carried in base) = GL 1100 at period end | E2 |
| I4 | Invoice `total_minor` = sum of `line_items[].amount_minor` (doc currency) | never (gate) |
| I5 | Each ledger row sourced from a doc = `convert(doc amount, rate on policy date)` | E3 |
| I6 | Each JE's local posting date is in the same period as its source event's local date (delivered_at, payment timestamp, accrual date) | E4 |
| I7 | Each payment's amount = its allocations per the doc + stated fees/discounts, and the ledger allocates to the referenced invoices | E2 |
| I8 | Each delivery (`delivery_id`, or customer+sku+service_date+qty) is billed at most once, net of credit notes | E1 |

I8 is an addition to the research's I1-I7. Without it a duplicate breaks no invariant, so the jury check "the declared discrepancy exists after injection" cannot pass. Publish the uniqueness rule in `policy.json`.

**Post-injection gate** (one test per issue): only the declared invariant fails, it fails by exactly `expected_amount_minor` (within the published tolerance), I1/I2/I4 hold, and an **independent re-checker that reads only the visible files plus the evidence set** reproduces the amount. The re-checker must not import the generator's internals other than the shared `convert()`.

## 3. House conventions (adopt unless the repo schema says otherwise; then enforce that one consistently)

- **Type slugs:** `duplicate_invoice`, `payment_allocation`, `fx_conversion`, `period_cutoff`. Each one can be toggled and counted on its own in config.
- **affected_ids** are `doc_id`s (unique on every document). The matcher may alias `invoice_id`/`payment_id` to `doc_id` only where that key is unique, and the alias rule is published. Compare as a set.
- **Amount:** `expected_amount_minor` is a positive integer, the absolute misstatement **in base currency**; `currency` = `manifest.base_currency`. A claim in the transaction currency does not match.
- **Evidence IDs:** ledger `row_id`, `doc_id`, and fx rows as `fx:<YYYY-MM-DD>:<SRC>:<TGT>`. Every evidence ID must resolve through `get_record`/`get_document`. `policy.json` is context the agent always sees, so it is never cited as evidence.
- **Linked findings:** truth items share a `link_id`/`scenario_id`, keep a unique `(type, affected_ids)` key, and are matched 1:1 independently. Amounts **do not double-count**: measure each one against the correct value of the other dimensions (see X1).
- **Canonical choices** (which doc is "the duplicate", allocation order) depend on timestamps or published rules, **never on ID lexical order or row order**. Paired checks rename IDs through a seeded bijection and shuffle rows.

## 4. Error types (exact mechanics)

Base currency KZT (exp 2). Full correct-vs-recorded JEs: [references/journal-entry-templates.md](references/journal-entry-templates.md).

**E1 `duplicate_invoice`** (breaks I8)
- Clone an invoice: new `doc_id`, the same customer, lines and `delivery_id`. The `invoice_id` is a normalized-equal variant (`INV-1042`->`INV1042`, `O`<->`0`, a transposition), and `issued_at` is strictly later (+0..3 days). Post a second T1 at the clone's own date rate. The customer pays once (to the original).
- `affected_ids` = {original, duplicate}. Amount = the duplicate's recorded Dr 1100 base amount.
- Evidence: {orig doc, dup doc, dup JE AR row}, with {orig, dup, dup revenue row} as an alternative.
- Example: USD 1,000.00 at 478.25 = 47825000, then the clone dated 2026-10-14 at 479.10 gives expected 47910000.
- Guard: if the clone's `invoice_id` is *identical*, credit notes and payments that reference it become ambiguous. Use variants whenever anything else references the original.

**E2 `payment_allocation`** (breaks I3, I7)
- The payment doc has `invoice_refs:[X]`. The ledger credits 2150 for all of it (missing allocation) or part of it (partial), instead of 1100-X.
- `affected_ids` = {payment doc, invoice X doc}. Amount = the AR overstatement = unallocated doc amount x the invoice's **carried** rate (in base currency it is just the unallocated amount).
- Evidence: {payment doc, invoice doc, the 2150 row or the short AR row}.
- Example: KZT invoice 125000000, payment 125000000 credited 80000000 to AR and 45000000 to 2150, so expected 45000000.
- Allocation rule (published): a single-ref payment allocates min(amount, open balance). A multi-ref payment allocates in `invoice_refs` order up to each open balance, or uses explicit per-invoice amounts on the doc. Without one of these, truth is ambiguous.
- Do not inject single E2 into foreign-currency payments unless the carried-rate convention is published. Otherwise 2150 at the payment rate and AR at the carried rate are two defensible answers.

**E3 `fx_conversion`** (breaks I5)
- A foreign doc's ledger base amount uses the wrong rate date (d+-k, the invoice-date rate applied to a payment, a monthly average when the policy says spot) or the wrong rate (inverted, truncated, another pair's rate). The JE rebalances through 7100/7200.
- `affected_ids` = {doc}. Amount = |recorded - correct| on the row converted from the doc (AR for an invoice, Bank for a payment).
- Evidence: {doc, `fx:<correct date>`, ledger row}.
- Example: USD 2,400.00 on 2026-10-20 at 479.80 = 115152000. Booked at the 10-12 rate 478.25 = 114780000, so expected 372000.
- Guard: compute the wrong rate's *effective* date. For example, Sat 2026-10-31 and Sun 11-01 both resolve to Fri 10-30, so that "date error" changes nothing (a no-op). Require |delta| >= 100x tolerance.

**E4 `period_cutoff`** (breaks I6)
- Move a JE's timestamp across the period boundary and leave documents untouched.
  - Premature: event (delivered_at / payment timestamp) in P+1, JE in P.
  - Late: event in P, JE in the post-period window, with no T9 accrual.
- `affected_ids` = {doc}. Amount = the correct base amount of the shifted revenue (or bank) row.
- Evidence: {doc, ledger row}.
- Example: delivered_at `2026-10-31T19:30:00Z` (= 11-01 00:30 local), but revenue posted 2026-10-31. That is premature, expected = invoice base.
- Docs must carry the event instant, and revenue is recognised at `delivered_at` (IFRS 15.38). The correct world never invoices before delivery.

### Interacting scenarios (linked findings)

**X1 Partial USD payment at the period boundary (3 findings, one `link_id`)**
- Facts: invoice USD 1,000.00 issued 10-12 at 478.25 (AR 47825000). The payment doc is USD 400.00 at `2026-10-31T23:40:00+05:00` (Saturday), refs [inv], so the rate is Fri 10-30 = 481.50.
- Correct JE (dated 10-31): Dr Bank 19260000 / Cr AR 19130000 / Cr 7100 130000.
- Recorded JE (dated 11-02, Monday rate 484.90): Dr Bank 19396000 / Cr AR 7173750 (USD 150 carried) / Cr 7100 99750 / Cr 2150 12122500.
- Findings:
  - `period_cutoff` {pay}: 19260000, the correct value.
  - `fx_conversion` {pay}: 136000.
  - `payment_allocation` {pay, inv}: 11956250, USD 250 at carried 478.25. The alternatives 12037500 and 12122500 must lie outside tolerance.

**X2 Foreign duplicate + credit note + FX (2 findings)**
- Facts: EUR invoice A 3,000.00 issued 10-05 at 512.30 (153690000). Credit note CN of EUR 500.00 dated 10-14 references A. Clone D (`INV-0731`->`INV-731`) is dated 10-07 at 513.00 (153900000). CN is booked at A's rate 512.30 (25615000) instead of its own date's rate 514.10 (25705000).
- Findings:
  - `duplicate_invoice` {A, D}: 153900000.
  - `fx_conversion` {CN}: 90000.
- This defeats naive tests: the base amounts of A and D differ, and the net open amount of A differs from D.
- Do **not** put the FX error on D itself. D is wholly invalid (occurrence), so a valuation error on it double-counts.

## 5. Distractors (legit, provably fine, each aimed at a naive test)

Each one must satisfy **all** invariants. If it trips one, it is an undeclared error and that is a CRITICAL defect. Clean companies keep every distractor generator on.

| # | Distractor | Fools | Visible proof |
|---|---|---|---|
| D1 | Monthly subscription: same customer, same amount | same-vendor-same-amount dup test | different `service_period`/`delivery_id`, own payment each |
| D2 | Split payment (2-3 receipts for one invoice) | partial allocation | sum of receipts = total, each refs the invoice |
| D3 | Bulk payment covering several invoices | unmatched amount | refs list sums exactly |
| D4 | Genuine open balance (customer paid part) | partial allocation | ledger allocation = payment exactly |
| D5 | Credit note, then net payment | short payment | CN doc `ref_invoice_id` + T7 |
| D6 | Cancel-and-reissue (full CN on A, new B for the same amount) | duplicate | CN fully offsets A |
| D7 | Overpayment, then refund | duplicate payment | T6 rows, refund = excess |
| D8 | FX receipt with a realized gain/loss row | wrong FX | both dates' rates in fx.csv, row = difference |
| D9 | Weekend/holiday doc uses Friday's rate | wrong FX date | policy "last available on or before" |
| D10 | 23:50 local / UTC-written instant that is still in the period locally | cutoff | manifest tz |
| D11 | Accrual + day-1 reversal (T10); accrued revenue later invoiced (T9) | duplicate / cutoff | reversal refs the accrual |
| D12 | Bank fee netted (T11) | short payment | fee stated on the payment doc + 6500 row |
| D13 | Early-payment discount 2/10 net 30 (T12) | short payment | invoice terms, paid within 10 days, 4200 row |
| D14 | Same amount, different customers, same day | duplicate | different `entity_id`, deliveries |
| D15 | Penny drift within tolerance | FX error | difference <= published tolerance |
| D16 | Near-identical numbers that are genuinely different (`INV-1042`/`INV-1024`) | fuzzy dup | different lines/dates/deliveries |
| D17 | On-account receipt, `invoice_refs: null` -> 2150 (T5) | missing allocation | doc has no refs |
| D18 | Prior-period invoice (opening AR) paid this period with FX diff vs carried rate | wrong FX | opening JE + historical fx row present |
| D19 | Same-day correction: wrong account, reversed and re-posted | duplicate posting | reversal JE references the original |
| D20 | Counterparty in a DST zone across 2026-10-25 (Europe/Berlin +02->+01) | cutoff/tz | offsets in the instants, company-tz rule |

## 6. FX and rounding rules

- Money is an `int` in minor units, with exponents from **ISO 4217 List One** (pinned table, publish date in the manifest): KZT/USD/EUR 2, JPY 0, KWD 3. Never take digits from CLDR/Babel or a PSP. Never use floats anywhere: rates are decimal strings and JSON money is int.
- `rate` = units of base per 1 unit of source. Store only doc->base pairs. **Never invert** (Reg. 1103/97 Art. 4). Two rows for the same `(date, src, tgt)` count as contradictory data.
- `base_minor = round(Decimal(doc_minor) * Decimal(rate) * 10**(e_base - e_doc))`, with **ROUND_HALF_EVEN** (or whatever the manifest declares), `prec >= 34`. Use one shared `convert()` in the generator, oracle, re-checker and matcher.
- **Convert once per document total**, never per line. A per-line sum can drift by up to n_lines/2 units (3 x USD 19.99 at 478.255: per line 2868096, per total 2868095).
- Rate date = the **company-local date** of `issued_at` (invoice/CN) or `timestamp` (payment). If that date has no row, use the last available on or before it. fx.csv must cover every needed date, including opening-balance invoices.
- AR is relieved at the **carried** rate, and realized FX goes to 7100/7200 (IAS 21.28). On the final allocation, relieve the *remaining carrying amount* rather than a fresh conversion, so AR clears to exactly 0. Revaluation (T13) is off in v1.
- Tolerance: 0 for same-currency comparisons, and the published `amount_tolerance_minor` for converted ones. Every injected delta must be >= 100x the tolerance, and every rounding distractor <= the tolerance.

## 7. Period and timezone rules

- The period is `[start, end_exclusive)` in `manifest.timezone` (IANA). Convert each ISO-8601 instant to that zone, then take the date. Never compare strings or naive datetimes across offsets.
- Pin `tzdata` and set `PYTHONTZPATH=""`. Asia/Almaty is +05 since 2024-03-01 (+06 before), so system tz files differ between machines.
- fx.csv dates are company-local business days with weekend gaps. Period end 2026-10-31 is a **Saturday**.
- Event dates: revenue at `delivered_at`, cash at the payment `timestamp`, accruals at period end. The ledger `timestamp` is the posting instant.
- Boundary set to generate (clean twin + error for each): 23:xx local, UTC-written instants crossing midnight (`2026-10-31T19:30Z` is in November locally), `+06:00` instants (`2026-11-01T00:10+06:00` is still Oct 31), and DST counterparties.

## 8. Review checklist (auditor council reviewer)

**Procedure.** Take one dirty, one clean, and one X1/X2 episode. For each truth issue, recompute the amount by hand from the visible records in its evidence set using `policy.json`. Check provenance `before` satisfies I1-I8. Check every distractor against every invariant. Look for a second defensible answer. Run the naive reconciler and flag-all baselines in your head. Then grep observations, trajectories and error messages for truth.

| Severity | Score cap | Examples |
|---|---|---|
| BLOCKER | <= 2 | truth leaks via observations, error text, paths, trajectories, ID patterns (`DUP-`, gaps, suffixes), row order (injected rows last), or formatting differences (key order, seconds `:00`, missing optional fields only on injected records); any shipped episode fails I1-I8 pre-injection; real or PII-like data; matcher reads `explanation`; truth hand-written instead of derived |
| CRITICAL | <= 5 | **unprovable truth** (evidence set does not reproduce the amount, or needs an unavailable fact or unpublished rule); **contradictory data** (duplicate fx keys, doc<->ledger mismatch absent from truth); **ambiguous truth** (two defensible ids, amounts or types; alternatives within tolerance; multi-ref allocation without a rule; identical-ID dup referenced by a CN); **distractor that is actually an error**; injected no-op (weekend collapse, delta 0) |
| MAJOR | <= 7 | injection unbalances a JE or the TB; delta < 100x tolerance; an error findable by a single-file trick; linked findings double-count; canonical rule depends on ID or row order; clean episodes without distractors; flag-all or the naive reconciler scores >= 0 on clean books; an error type that cannot be toggled on its own |
| MINOR | -0.5 each (max -2) | conventions consistent but undocumented; no per-type P/R/F1; unrealistic data (round amounts, uniform dates, one currency); errors only Benford could reveal |

Score = min(10, lowest cap triggered) minus MINOR deductions. **9 or higher requires** no BLOCKER/CRITICAL/MAJOR, plus a hand recomputation of at least one issue of each type and both X scenarios that matched truth exactly. Report each finding as `severity | file:line or episode/issue | what breaks | fix`.

## References

- [references/journal-entry-templates.md](references/journal-entry-templates.md): chart of accounts, Dr/Cr templates T0-T13 with minor-unit numbers, and correct-vs-recorded JEs for E1-E4, X1 and X2. Read it before writing the posting engine or any injector.
- [references/standards-notes.md](references/standards-notes.md): IAS 21, IFRIC 22, IFRS 15, ISA 240/330/500/501/505/520, Reg. 1103/97, ISO 4217, duplicate and cash-application practice, with URLs. Read it when a rule needs a citation or a reviewer challenges a convention.

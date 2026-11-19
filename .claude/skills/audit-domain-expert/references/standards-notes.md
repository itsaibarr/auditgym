# Standards notes (what each rule means for AuditGym)

Paragraph numbers follow the clarified ISAs and current IAS 21. The official IFRS and IAASB texts are behind a login or only in handbooks, so the URLs below are national adoptions with verbatim-equivalent text, or summaries. Mark a citation *(secondary)* when you rely on a summary. AuditGym is a software task, not an audit opinion (case p.6). Cite standards to justify conventions, not to claim compliance.

## IFRS: measurement rules the generator encodes

**IAS 21 The Effects of Changes in Foreign Exchange Rates**
- 21.8 definitions:
  - **Spot rate** is the rate for immediate delivery. **Closing rate** is the spot rate at the end of the reporting period.
  - **Monetary items** are units of currency held, plus assets and liabilities to be received or paid in a fixed or determinable number of currency units. Trade receivables, cash and refund liabilities are monetary.
- 21.21: on initial recognition, a foreign-currency transaction is recorded in the functional currency at the **spot rate on the transaction date**. This gives one rate per document, at its own date.
- 21.22: the transaction date is when the item first qualifies for recognition. An **average rate** for a week or month is allowed only if rates do not fluctuate significantly. So "used the monthly average when policy says spot" is a legitimate E3 variant, but only if `policy.json` says spot.
- 21.23: at the reporting date, monetary items are retranslated at the closing rate. That is T13 revaluation, off in v1, and the policy must say so.
- 21.28: exchange differences on **settlement** of monetary items, or on translation at rates different from initial recognition, go to **profit or loss**. That is realized FX via 7100/7200, with AR relieved at the carried rate.
- URLs: https://www.iasplus.com/en/standards/ias/ias21 · https://ifrscommunity.com/knowledge-base/ias-21-effects-of-changes-in-foreign-exchange-rates/ · AASB 121 (equivalent text) https://standards.aasb.gov.au/node/728 · ICAEW https://www.icaew.com/technical/financial-reporting/ifrs/ifrs-standards/ias-21-the-effects-of-changes-in-foreign-exchange-rates

**IFRIC 22 Foreign Currency Transactions and Advance Consideration**
- For a prepayment, the rate date is the date the non-monetary advance asset or liability was first recognised, with one date per advance.
- Relevant only if customer advances are later applied to invoices. Otherwise keep advances in 2150 as cash received.
- URL: https://www.iasplus.com/en/standards/ifric/ifric-22

**IFRS 15 Revenue from Contracts with Customers**
- 15.31 and 15.38: revenue at a point in time is recognised when **control transfers**. Indicators are a present right to payment, legal title, physical possession, risks and rewards, and acceptance.
- For AuditGym this means revenue is recognised at `delivered_at`, and E4 is a mismatch between `delivered_at` and the posting date. T9 (accrued revenue) is the correct treatment for "delivered, not yet invoiced".
- URLs: https://www.iasplus.com/en/standards/ifrs/ifrs15 · https://dart.deloitte.com/USDART/home/publications/deloitte/additional-deloitte-guidance/roadmap-ifrs-us-gaap-comparison/chapter-3-revenues-expenses/3-1-revenue-recognition

**EU Reg. 1103/97, Art. 4-5 (conversion precedent)**
- Rates carry six significant figures and are **not rounded or truncated** during conversion. **Inverse rates must not be used.** The result is rounded to the smallest unit, with half rounded up.
- This supports: round once at the end; the inverted or truncated rate as an E3 variant; and HALF_UP as a legitimate declared alternative to HALF_EVEN.
- URL: https://eur-lex.europa.eu/eli/reg/1997/1103/2001-01-01/eng

**ISO 4217 List One (SIX)**
- The authoritative source for minor-unit exponents: KZT/USD/EUR 2, JPY/KRW/ISK 0, KWD/BHD/OMR/JOD/TND 3, CLF 4.
- CLDR (Babel) "usual digits" and PSP conventions such as Stripe's ISK/HUF differ from it, so do not use them.
- URL: https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml

**Reference-rate practice (ECB)**
- Rates are published on TARGET business days only, so there is no rate on weekends or holidays. This is why fx.csv has gaps and `policy.json` must declare "last available on or before".
- URL: https://www.ecb.europa.eu/stats/eurofxref

## ISA: what makes truth "provable"

**ISA 500 Audit Evidence**
- 500.6: obtain **sufficient appropriate** evidence. Sufficient is quantity; appropriate is relevance and reliability.
- 500.7: consider the relevance and reliability of the information used.
- 500.9: information produced by the entity must be accurate and complete enough for the purpose.
- 500.11: if evidence from one source is inconsistent with another, resolve it.
- A31 (2009 numbering; later editions renumber it): reliability is higher for external, documentary and original evidence.
- What this means for AuditGym:
  - Each acceptable evidence set is a **minimal set of records that jointly proves the discrepancy**. Documents play the role of external/originating evidence, and the ledger is the entity's record under test.
  - Contradictory data that is not part of truth (500.11) is a data defect, not difficulty (case p.5).
  - No approved ISA 500 (Revised) is in force. The IAASB's 2026 exposure drafts for 500/330/520 are proposals only.
- URLs: ISA 500 (IAASA) https://iaasa.ie/wp-content/uploads/2023/10/ISA_500_UpdatedOct2023.pdf · ACCA "when is enough enough" https://accaglobal.com/uk/en/technical-activities/uk-tech/in-practice/2024/April/Audit-evidence-when-is-enough-enough.html · status note https://www.learnsignal.com/blog/isa-500-revised-does-not-exist-yet/ *(secondary)*

**ISA 330 The Auditor's Responses to Assessed Risks**
- 330.18: perform substantive procedures for each material class of transactions and balance, irrespective of assessed risk.
- 330.20: closing-process procedures, i.e. **agree or reconcile the statements with the accounting records** and **examine material journal entries and adjustments**.
- What this means for AuditGym:
  - Tests of details carry the load. The subledger<->GL reconciliation (I3) and the doc<->ledger vouching (I5, I6) are the agent's procedures.
  - Cutoff testing is a test of details around period end: **trace forward** from shipping or delivery to invoice to ledger (completeness, late), and **vouch backward** from ledger to delivery (occurrence, premature). Typical windows are 5-10 days either side, which is why `cutoff_window_days` exists.
- URLs: ASA 330 (equivalent) https://standards.auasb.gov.au/node/861 · ACCA https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f8/technical-articles/ISA330-responses-assessed-risks.html · cutoff procedure https://open-exam-prep.com/study-guides/cpa-aud/revenue-receivables/revenue-receivables-cycle *(teaching)*

**ISA 315 (Revised 2019): assertions**
- The transaction-class assertions are occurrence, completeness, accuracy, **cutoff** and classification. The error types map onto them:
  - E1 is occurrence.
  - E2 is classification/accuracy (cash credited to 2150 instead of AR).
  - E3 is accuracy/valuation.
  - E4 is cutoff.
- Linked findings in X1 are *different assertions* on one record, which is legitimate. A valuation finding on a record that fails occurrence (an FX error on a duplicate) is not, because it double-counts.
- URL: https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f8/technical-articles/assertions.html

**ISA 240 Fraud**
- Journal-entry testing: inquire about inappropriate or unusual entries, **select entries at the end of the reporting period**, and consider entries throughout the period. This is 240.32 in the 2009 text and para 33 in ICAEW's current numbering.
- ISA 240 (Revised) applies to periods beginning on or after **15 Dec 2026**. It adds evidence on the **completeness of the journal-entry population**.
- What this means for AuditGym: period-end and post-period-window entries are where E4 and X1 sit. Red-flag heuristics also fire on legitimate reversals and accruals (D11, D19), so they need clean twins.
- URLs: https://www.icaew.com/technical/audit-and-assurance/audit/risk-assessment-internal-control-and-response/efficient-journals-testing · ISA 240 (Revised) text https://mia.org.my/storage/2025/10/ISA-240-Revised.pdf · IAASB https://www.iaasb.org/publications/isa-240-revised-auditor-s-responsibilities-relating-fraud-audit-financial-statements

**ISA 501 Specific Considerations for Selected Items**
- Covers inventory (attending counts), litigation and claims, and segment information.
- **Out of scope** for AuditGym: there is no inventory or litigation data. Do not design errors that need these.
- URL: https://iaasa.ie/?p=4105 · summary https://opentuition.com/glossary/isa-501-audit-evidence-specific-considerations-for-selected-items/

**ISA 505 External Confirmations**
- Third-party confirmations, such as bank and customer balance letters, are the most reliable evidence. AuditGym has no third party.
- `payments.jsonl` stands in for the bank statement or remittance advice, and `invoices.jsonl` for the issued documents.
- Never require a fact that only a confirmation could supply, such as "the customer says it never received INV-X".
- URL: https://iaasa.ie/wp-content/uploads/2022/09/ISA505I.pdf

**ISA 520 Analytical Procedures**
- 520.5: develop an expectation precise enough to detect a misstatement, and set the acceptable difference.
- 520.7: investigate significant differences.
- With one period per episode, trend analytics are mostly unavailable. Benford-style tests are too coarse at 200-2,000 rows. **Every error must be provable from specific records**, never only from a statistical anomaly.
- URLs: https://www.accaglobal.com/gb/en/student/exam-support-resources/professional-exams-study-resources/p7/technical-articles/analytical-procedures.html · Nigrini (Benford) https://www.journalofaccountancy.com/issues/1999/may/nigrini.html

## Practice sources (duplicates, cash application, subledger)

**Duplicate-payment tests**
- ERP default checks block only an exact 4-field match: invoice ID, date, amount and vendor.
- Real duplicates differ by a letter O vs zero (`891112450`/`89111245O`), transposed digits, a +-2 day date, or a vendor set up twice.
- Normalize IDs (strip spaces, hyphens and leading zeros; ignore case), then cascade: exact match, then same counterparty + amount within a ~30-day window, then fuzzy.
- E1 variants copy these patterns. D1, D14 and D16 are the clean twins.
- URLs: Oversight Systems https://www.oversight.com/hubfs/Detect%20the%20Duplicate%20Infographic%20Apr21.pdf · Corpay https://www.corpay.com/resources/blog/duplicate-payment-detection *(vendor)*

**Cash application**
- Match by explicit reference first, then by amount combinations (subset-sum) as a last resort. Combination matching breaks when invoice values repeat or tolerances apply.
- Over- or under-payment within tolerance creates an adjustment item.
- Fees, write-offs and rounding each get their own row and account (ERPNext). Partial reconciliations carry an exchange-difference move (Odoo).
- This is the basis for the published allocation rule and for stating fees and discounts on the payment doc.
- URLs: ERPNext FX https://docs.frappe.io/erpnext/manage-foreign-exchange-difference · Odoo multi-currency https://www.odoo.com/documentation/19.0/applications/finance/accounting/get_started/multi_currency.html · PeopleSoft algorithm groups https://docs.oracle.com/en/applications/peoplesoft/financials-and-supply-chain-management/9.2.056/peoplesoft-receivables/understanding-sample-algorithm-groups.html · subset-sum matching https://arxiv.org/abs/2508.19218

**AR subledger vs control account**
- The GL control balance equals the sum of customer balances.
- Differences usually come from manual entries posted straight to the control account. That is I3, and it is how E2 shows up.
- URL: https://www.accountingtools.com/articles/how-to-reconcile-accounts-receivable.html

**Credit notes, overpayments, refunds**
- A credit memo is Dr Sales returns / Cr AR.
- An overpayment is a customer credit balance or a refund liability, and the refund is booked against that liability.
- URLs: https://courses.lumenlearning.com/suny-finaccounting/chapter/seller-entries/ · https://www.netsuite.com/portal/resource/articles/accounting/overpayment.shtml

**Time zones**
- Kazakhstan moved to a single zone of UTC+5 on 2024-03-01: Asia/Almaty went from +06 to +05, which is visible in tzdata 2024a and later.
- Pin `tzdata` and set `PYTHONTZPATH=""` so period boundaries are identical on every machine.
- URLs: https://pypi.org/project/tzdata/ · https://docs.python.org/3.11/library/zoneinfo.html

**Rounding**
- Python `decimal` defaults to ROUND_HALF_EVEN with prec 28. Raise the precision locally (>= 34) for conversions.
- URL: https://docs.python.org/3/library/decimal.html

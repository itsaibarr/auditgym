# 02 — Finance / Audit References for the Audit-Gym Environment

Research date: 2026-11-19. Scope: Genesis Hackathon FinTech case "RL-среды для дообучения моделей финансового аудита" (`research/case.pdf`).
Every claim carries a URL. "Verified" means I read the primary page, repo file or GitHub API metadata myself. Claims taken only from a secondary summary are marked *(secondary)*.

The constraints from the case that drive the recommendations below:
- One company and one reporting period per episode, with 200–2,000 ledger postings, 20–200 documents and 0–12 errors.
- Visible files: `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`, `policy.json`, `manifest.json`.
- Money is stored as integer minor units with a currency exponent. FX rates are decimal strings. Timestamps are ISO-8601 with an offset.
- The jury checks three things in each hidden episode: "accounting equalities hold before the error is injected", "the declared discrepancy exists after injection", and "independent sufficiency of evidence". Contradictory or unsolvable data is not credited as difficulty (case p.5).
- The organisers run 4 fixed policies: random explorer, a simple reconciliation algorithm, flag-everything, and an oracle (case p.4).

---

## 1. Project table: generators, datasets, ledgers, money libraries

| # | Project | URL | License (verified) | What it is | What we borrow |
|---|---|---|---|---|---|
| 1 | **DataSynth / SyntheticData** (Rust; EY / ASU lineage) | https://github.com/mivertowski/SyntheticData (repo returned 404 to GitHub API on 2026-11-19); docs https://mivertowski.github.io/SyntheticData/ ; crates https://docs.rs/crate/datasynth-generators/1.2.0/source/README.md | Apache-2.0 per crate metadata *(secondary: https://awesome.ecosyste.ms/api/v1/projects/github.com%2Fey-asu-rnd%2Fsyntheticdata)* | Synthetic enterprise data: GL journal entries, chart of accounts (small/medium/large), document flows, subledgers, SAP BKPF/BSEG/ACDOCA exports. "ACFE-aligned fraud labels" and a duplicates injector. It claims to preserve accounting identities (debits = credits, A = L + E) *(secondary: search snippet of repo description)*. | **Ideas only.** It is Rust and the repo is currently unreachable, so we can't depend on it. Borrow: (a) the "document flow → postings" architecture, (b) invariant-gate thinking (balance check before export), (c) labelled-injector pattern. |
| 2 | **deepAI GTC-2018 lab** (Schreyer et al.) | https://github.com/GitiHubi/deepAI | GPL-3.0 (GitHub API) | Contains `data/fraud_dataset_v1.csv` / `v2.csv` (~23–27 MB) with SAP-style journal attributes: `BELNR` (doc no.), `BUKRS` (company code), `WAERS` (currency key), `KTOSL` (GL account key), `PRCTR` (profit center), `BSCHL` (posting key), `HKONT` (GL account), `DMBTR` (amount in local currency), `WRBTR` (amount in document currency). Verified by grepping `GTC_2018_Lab.ipynb`. | **Schema idea only** (GPL, don't vendor the data). The `DMBTR` vs `WRBTR` split is exactly our "document amount vs base/functional amount" distinction. |
| 3 | Schreyer et al., *Detection of Anomalies in Large Scale Accounting Data using Deep Autoencoder Networks* | https://arxiv.org/abs/1709.05254 | arXiv | Real SAP data with **injected** "global" anomalies (attribute values not in the original data) and "local" anomalies (attribute combinations not in the original data), so ground truth is known (verified from PDF text). | This is the canonical "inject into clean data and record labels" methodology. It shows why statistically anomalous ≠ erroneous: their score flags rare entries, and the paper notes red-flag tests produce "a high volume of false positive alerts due to events such as reverse postings, provisions and year-end adjustments." That is a ready-made distractor list. |
| 4 | Gronewald, Scherer, Mehdiyev, *ML for journal entry testing: a type-aware evaluation under a review budget* (Oct 2026) | https://arxiv.org/abs/2610.06133 | CC BY 4.0 paper | Typed injected anomalies in 4 client ledgers, plus a public synthetic ledger. A search snippet describes it as "DIB General Ledger Data", a fictional gardening-tools trader with 256,232 posting lines and 8 marking types *(secondary)*. Key finding: high hit rates hide blind spots, because detectors find ~98 of the top 100 but ≥95.8 of those are one type. | Supports per-type reporting of P/R/F1. A "flag everything of one easy type" policy can look good on aggregate metrics, so we should show per-type recall in diagnostics. |
| 5 | **PaySim** | https://github.com/final12/PaySim | GPL-3.0 (GitHub API) | Agent-based mobile-money simulator calibrated from real logs. Fraud is labelled. | Not a ledger. Borrow only the idea of a calibrated agent simulation with a fixed seed. Don't vendor (GPL). |
| 6 | **AMLSim** (IBM) | https://github.com/IBM/AMLSim | Apache-2.0 (GitHub API) | Multi-agent bank-transaction simulator. Typologies come from `alertPatterns.csv`, and `conf.json` has `"random_seed": 0` and `base_date`. Outputs are transactions plus alert/SAR label files (README: https://github.com/IBM/AMLSim/blob/master/README.md). | **Config-driven injection pattern**: a parameter file of "patterns" plus a seed produces transactions plus a separate label file. Our `config` → `truth.json` / `provenance.json` mirrors this. |
| 7 | **IBM AML-Data** (Altman et al.) | https://github.com/IBM/AML-Data | Code Apache-2.0; **data CDLA-Sharing-1.0** (README) | Fully synthetic multi-agent world. Each transaction carries a "laundering" tag. | Same "simulate the world first, then label" approach. We don't need the data. |
| 8 | **FinMaster / FinSim** (2025) | https://github.com/JiangInsight/Finmaster ; paper https://arxiv.org/html/2505.13533v1 | **No license file** (GitHub API: none), so we can't reuse the code | Simulates 5 company types with purchases/sales, depreciation, cash↔bank transfers and expenses. For auditing it injects **12 error types**: 7 record errors (type, date, payment status, method, qty, unit price, receive method), 3 calculation errors (amount, tax, profit) and 2 approval mismatches. Ground truth comes from the simulator. | Closest prior art to our generator. Borrow the **taxonomy style**: separate record vs calculation vs control errors, at single-error and multi-error levels. |
| 9 | **python-accounting** (ekmungai) | https://github.com/ekmungai/python-accounting | MIT (GitHub API + README) | Double-entry library focused on IFRS reporting. It has account types (receivable, bank, revenue, control…) and transactions (client invoice, client receipt, journal entry). Its **Assignments** let a receipt partially clear an invoice (README: assigning 15 of a 50 receipt leaves a receipt balance of 35). | The **Assignment / clearing** model (invoice cleared amount, receipt unassigned balance). Too heavy as a dependency (SQLAlchemy), so re-implement ~40 lines. |
| 10 | **Odoo** `account.partial.reconcile` / `account.full.reconcile` | https://github.com/odoo/odoo (LICENSE: LGPLv3, verified) ; multi-currency doc https://www.odoo.com/documentation/19.0/applications/finance/accounting/get_started/multi_currency.html | LGPL-3.0 | Each partial reconcile links a debit line, a credit line, an amount, amounts in currency, and optionally an **exchange-difference move**. FX differences are posted automatically to dedicated gain/loss accounts in a dedicated journal (Odoo doc). Exchange differences are now also created on partial reconciliations, not only full ones *(secondary: https://git.coopdevs.org/coopdevs/odoo/OCB/-/commit/5883f0a84cd3cd2330cc5ff34261252dacd8c5b9)*. | **Data model for allocations**: `allocation(payment_id, invoice_id, amount_doc_minor, amount_base_minor, fx_diff_base_minor)`. Our hidden truth should carry exactly this table. |
| 11 | **ERPNext** | https://github.com/frappe/erpnext (GPL-3.0, GitHub API); FX doc https://docs.frappe.io/erpnext/manage-foreign-exchange-difference ; reconciliation https://docs.frappe.io/erpnext/payment-reconciliation-problems | GPL-3.0 | Worked example: invoice EUR 1,200 @1.10 = USD 1,320; receipt @1.15 = USD 1,380; USD 60 realized gain posted to the Exchange Gain/Loss account. The doc says bank fees, write-offs and rounding "each need their own row and account", and that revaluation (unrealized) is separate from realized gain on payment. A setting chooses whether the gain/loss posting date is the invoice date, the payment date or the reconciliation date. | Template for the **realized FX on settlement** entry. The "posting-date inheritance" setting is a real-world policy choice, so it belongs in `policy.json`. |
| 12 | **Beancount** | https://github.com/beancount/beancount ; tolerances https://beancount.io/docs/Basics/precision | GPL-2.0 (GitHub API) | Plain-text double-entry. Inferred tolerance = 0.5 × the last decimal digit present (rustledger documents the same rule: https://docs.rs/crate/rustledger-core/0.13.0/source/src/amount.rs). Balance assertions use twice that tolerance. | **Tolerance rule** (half a unit of the smallest digit) and the **balance-assertion** idea for our pre-injection equalities. Don't depend on it (GPL). |
| 13 | **hledger** | https://github.com/hledger/hledger (moved from simonmichael/hledger) | GPL-3.0 (GitHub API) | Plain-text accounting with CSV import rules (https://hledger.org/). | Optional external sanity check during development only: export ledger → journal → `hledger bal`. |
| 14 | **Ledger (ledger-cli)** | https://github.com/ledger/ledger | BSD-style (LICENSE.md, verified) | Original plain-text double-entry tool. | Same, dev-time only. |
| 15 | **py-moneyed** | https://github.com/py-moneyed/py-moneyed | BSD-3-Clause | Money/Currency classes. | Not needed. A 20-row exponent table from ISO 4217 is simpler and deterministic. |
| 16 | **Babel** | https://github.com/python-babel/babel ; https://babel.pocoo.org/en/latest/api/numbers.html | BSD-3-Clause | `format_currency(1099.98,'JPY')` → `¥1,100`, because it uses the currency's decimal digits from CLDR. | **Warning**: CLDR "usual digits" can differ from ISO 4217 minor units (*secondary*: https://allratestoday.com/blog/free-currency-data-api-iso-4217-metadata/). Use ISO, not Babel, for exponents. |
| 17 | **ISO 4217 List One** (SIX, maintenance agency) | https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml | Free download | Downloaded and parsed (published 2026-09-17): **USD 2, EUR 2, GBP 2, CNY 2, RUB 2, KZT 2, JPY 0, KRW 0, ISK 0, KWD 3, BHD 3, OMR 3, JOD 3, TND 3, IQD 3, LYD 3, CLF 4, UYW 4.** | **The source of truth for `exponent`**. Vendor a small pinned table with the List One publish date in `manifest.json`. |
| 18 | **Faker** | https://github.com/joke2k/faker | MIT (GitHub API) | Fake names and addresses. | Optional. Output depends on the Faker version, which threatens bit-exact reproducibility, so either pin it exactly or use our own small name lists (recommended). |
| 19 | **Synthetic-General-Ledger-Fuzzer** | https://github.com/Atiqul-Akash/Synthetic-General-Ledger-Fuzzer | MIT | 2 stars and buzzword-heavy ("SOX-404 Compliant", "triple-entry DLT"). It claims a double-entry zero-sum gate and an IAS 21 triangulation engine. | Low trust. Skim for ideas only. |
| 20 | **Gymnasium** | https://github.com/Farama-Foundation/Gymnasium (MIT) ; API https://gymnasium.farama.org/api/env/ | MIT | `reset(seed, options) -> (obs, info)` and `step -> (obs, reward, terminated, truncated, info)`. | The contract named in the case. |

---

## 2. Reconciliation / cash-application engines and journal-entry tests

### 2.1 Engines and algorithms

| Name | URL | License | Relevant mechanics |
|---|---|---|---|
| **Reconify** (Go CLI/lib) | https://github.com/reconifyhq/reconify | MIT (README badge + API) | Normalizes rows from banks, PSPs and ledgers, then reports: exact matches, missing records on either side, amount and timing differences, **duplicate groups**, and **grouped settlements (one-to-many, many-to-many)**. Its output categories map almost 1:1 to our finding types, so it's a good reference for the baseline policy. |
| **remittance-assigner** | https://github.com/brutus5000/remittance-assigner | MIT | Searches invoice combinations matching a payment while testing early-payment discounts of 0/2/3% *(secondary description: https://awesome.ecosyste.ms/projects/github.com%2Fbrutus5000%2Fremittance-assigner)*. A good **distractor** source: a legit discounted payment looks short. |
| **Subset Sum Matching Problem** (Wu, Veloso et al., ECAI 2025) | https://arxiv.org/abs/2508.19218 | arXiv | Formalizes reconciliation as a combinatorial "subset sum matching" task with two heuristic algorithms and one optimal algorithm, plus a generated benchmark. |
| Oracle PeopleSoft Receivables algorithm groups | https://docs.oracle.com/en/applications/peoplesoft/financials-and-supply-chain-management/9.2.056/peoplesoft-receivables/understanding-sample-algorithm-groups.html | docs | If a payment doesn't balance against the referenced items, the algorithm creates an **underpayment/overpayment item** *(secondary snippet)*. |
| Infor "algorithm method" | https://docs.infor.com/ifsm/2025.x/en-us/ifsmolh/ifsmreceiveug/cmo1450797813694.html | docs | Tries 1:1 first, then combinations of a capped number of the oldest invoices. Variance tolerance applies **only to 1:1** matches. Over/short within tolerance creates an adjustment record *(secondary snippet)*. |
| Combination-matching caveats | https://peakflo.co/blog/cash-application-automation-freight-receipts-logistics | vendor blog | Subset-sum becomes unreliable "as soon as invoice values repeat or a tolerance is applied", and unapplied credit notes break it. Recommended order: references → PO → remittance lines → solver last *(vendor source)*. |
| Odoo reconciliation models | https://www.odoo.com/documentation/17.0/applications/finance/accounting/bank/reconciliation_models.html | LGPL docs | "Rule to match invoices/bills" with a **payment tolerance**. The residual goes to a counterpart account *(secondary snippet)*. |

**Takeaway for the "simple reconciliation" baseline** (the organisers will run their own, but our env must defeat a naive one without being unsolvable):
1. Match payments to invoices by explicit `invoice_refs` (exact key).
2. For each invoice, compare `total_minor` to the sum of allocated `amount_minor` in the same currency, and flag the difference.
3. Duplicates: same `entity_id` + same `total_minor` + same currency, which is the "same vendor, same amount" test.

Steps 2 and 3 are deliberately fooled by our distractors (§4): legit open balances, credit notes, recurring equal amounts. Step 3 also misses fuzzy invoice-number duplicates, so the true duplicates require document-level evidence.

### 2.2 Duplicate-invoice / duplicate-payment tests (audit analytics)
- ERP default duplicate checks stop a payment only when **4 fields match exactly** (Invoice ID, Invoice Date, Invoice Amount, Vendor ID). Real duplicates slip through via: the same invoice received by EDI and also keyed manually, vendors set up twice in the vendor master, and amounts or invoice IDs altered to bypass controls. Concrete examples from the source: the same ID with dates Oct 12 vs Oct 14, `00028458` with $25,289.42 vs $25,289.24 (transposed cents), and `891112450` vs `89111245O` (letter O for zero). The IIA is cited for "duplicate payments up to 0.5% of total invoice payments". Source: Oversight Systems infographic, text extracted from PDF: https://www.oversight.com/hubfs/Detect%20the%20Duplicate%20Infographic%20Apr21.pdf (vendor source).
- Normalization recipe: strip spaces, hyphens and leading zeros, ignore case, and map vendor aliases. Then use a progressive cascade: exact pass, then same vendor + amount within a date window, then a fuzzy vendor-name pass. Corpay suggests pairs within about 30 days (*secondary/vendor*: https://www.corpay.com/resources/blog/duplicate-payment-detection, https://www.docsumo.com/blog/duplicate-invoice-detection).
- **Benford's law**: first-digit frequency is ~30% for "1" and <5% for "9". AP and journal data generally conform, but data with human-imposed min/max limits does not. Nigrini's first-two-digit test on AP shows excess duplication as spikes. It is a high-level reasonableness test, and too coarse to catch 1–2 falsified entries. Sources: Nigrini, Journal of Accountancy 1999 https://www.journalofaccountancy.com/issues/1999/may/nigrini.html ; Durtschi et al. https://researchgate.net/profile/Cindy-Durtschi-2/publication/241401706_The_Effective_Use_of_Benfords_Law_to_Assist_in_Detecting_Fraud_in_Accounting_Data/links/54982f4a0cf2c5a7e342a59e/The-Effective-Use-of-Benfords-Law-to-Assist-in-Detecting-Fraud-in-Accounting-Data.pdf . **Implication:** at 200–2,000 rows Benford is useless as a detector, so don't build errors that only Benford would reveal. The jury requires errors to be provable from specific records.
- **ISA 240 journal-entry testing**: inquire about inappropriate or unusual journal activity, **select entries made at the end of the reporting period** (ISA (UK) adds post-closing entries), and consider testing throughout the period. ICAEW places these in para 33 of the current text: https://www.icaew.com/technical/audit-and-assurance/audit/risk-assessment-internal-control-and-response/efficient-journals-testing . Other summaries number it para 32 (older numbering): https://ciferi.pages.dev/blog/how-to-test-journal-entries-fraud-isa-240 *(secondary)*. ISA 240 Revised text (2025): https://mia.org.my/storage/2025/10/ISA-240-Revised.pdf .

---

## 3. LLM benchmarks / RL environments for accounting, audit and finance

| Work | URL | What it measures | How ground truth is built | Lesson for us |
|---|---|---|---|---|
| **Automating Financial Statement Audits with LLMs** (Jun 2025) | https://arxiv.org/html/2506.17282v1 | An agent gets a company's transaction history plus a (possibly erroneous) financial statement. It must flag errors, explain them, cite FASB standards and revise the statement. There are 5 stages: judgment → identification → resolution → citation → revision. | Real S&P 500 statements (371). **GPT-4 synthesizes the transactions and also injects 4 error types** (missing row, numerical error, redundant row, misclassification). Labels are "aligned" because the same model generated the errors. Stages are scored with EM, BERTScore and BLEU. | Errors that are both generated and labelled by an LLM, plus fuzzy text metrics, are exactly what the jury penalises. **Our ground truth must be computed programmatically, and our matcher must use fields only (no text similarity).** |
| **FinMaster** (May 2025) | https://arxiv.org/html/2505.13533v1 ; code https://github.com/JiangInsight/Finmaster | 183 tasks across literacy, accounting, auditing and consulting. Accuracy drops from >90% on basic tasks to ~40% on complex ones. | FinSim simulator plus 12 injected error types. Ground truth comes from the simulated records. | Validates the "simulate company → inject typed errors → derive truth" pipeline. |
| **APEX-Accounting** (Mercor + Ramp, Jul 2026) | https://arxiv.org/abs/2607.27189 ; https://www.mercor.com/blog/introducing-the-ai-productivity-index-for-accounting/ | 160 tasks in 10 simulated company "worlds" with an accounting system plus spreadsheets and PDFs: reconcile accounts, accrue expenses, post transactions, produce reports. Metrics: Mean Criteria@3, Pass^8, Pass@8. Top score ~56% *(secondary: search summary)*. Reportedly excludes multi-currency, audit, consolidation and tax *(secondary)*. | Experts write and solve each task and write rubrics. | Multi-currency audit with programmatic truth is an **open gap**, which makes it a good pitch point. |
| **AccountingBench** (Penrose, Jul 2025) | https://gigazine.net/gsc_news/en/20250724-accountingbench ; https://www.accountingtoday.com/news/generalist-ai-model-accuracy-tops-out-at-77-for-accounting | Month-by-month close on 1 year of a **real** SaaS company's books against a CPA baseline. Errors compound across months. Several models gave up mid-month *(secondary press)*. | Human CPA closing as reference. | Errors snowball, so keep episodes to a single period as the case requires. |
| **Finance Agent Benchmark** (Vals AI, 2025) | https://arxiv.org/abs/2508.00828 | 537 expert questions over SEC filings, with EDGAR and search tools. Best original accuracy ≤46.8% *(secondary)*. | Expert-written answers. | Tool-use harness design (search → parse → retrieve). |
| **FinanceBench** (Patronus, 2023) | https://arxiv.org/abs/2311.11944 | 10,231 open-book QA items with answer **plus evidence string**. License CC BY-NC-ND 4.0. | Human-annotated. | Answer + evidence pairing, analogous to our `evidence_ids`. |
| **FinQA** (EMNLP 2021) | https://arxiv.org/abs/2109.00122 ; https://github.com/czyssrs/FinQA (MIT) | 8,281 QA pairs over S&P 500 earnings reports, each with a **gold reasoning program** plus supporting facts. Experts score 91%, the model 61%. | 11 finance professionals. | A gold "program" (arithmetic steps) is like our recorded causal change. |
| **TAT-QA** (ACL 2021) | https://arxiv.org/abs/2105.07624 ; https://github.com/NExTplusplus/TAT-QA (MIT) | 16,552 questions over hybrid table + text. Derivations are labelled. | Human annotators. | Same. |
| **FinAuditing** (Oct 2025) | https://arxiv.org/abs/2510.08886 | XBRL multi-document auditing: semantic matching, relation extraction, math reasoning. 1,102 annotated instances. | Real XBRL filings. | Real-data audit is long-context and retrieval-heavy, while ours is tool-interaction-heavy. |
| **AuditFlow** (Jun 2026) | https://arxiv.org/html/2606.03031 | XBRL consistency checks (DQC sign, dimension, calc-tree). The LLM chooses typed deterministic tools; **the tools compute and the LLM doesn't do arithmetic**. Explicitly "tests numerical reconstruction for known inconsistencies, **not false-positive behavior on compliant filings**". | Real engine-recorded DQC violations, no injection. | Our env **must** test false positives via clean companies and distractors, which AuditFlow admits it doesn't. Our `calculator` action matches the "tools compute" pattern. |
| **ESCTR** (HF hackathon env, Apr 2026) | https://huggingface.co/spaces/musharraf7/esctr-environment/blob/main/Blog.md | RLVR env: the agent is a financial controller with `query_database`, `read_document`, `communicate_vendor` and `submit_financial_decision`. Scenarios are procedurally generated from a seed. | Programmatic: the exact adjustment amount comes from contract terms. | GRPO starved of variance with sparse reward, which they fixed with per-step shaping and temperature. **The case forbids exploration reward (case p.3)**, so expect the sparse-reward problem if anyone trains, and document it as a limitation. |
| **Snorkel FinQA OpenEnv** (Mar 2026) | https://snorkel.ai/blog/building-finqa-an-open-rl-environment-for-financial-reasoning-agents/ | 290 questions, tools `get_descriptions` / `get_table_info` / `sql_query` / `submit_answer`, ≤50 tool calls, binary reward with normalized numeric matching "within tolerance". Docker-isolated. | Expert-curated. | Normalization before matching (parenthesized negatives, separators) is useful in our matcher for `claimed_amount_minor` parsing. |
| **τ-bench** (ICLR 2025) | https://arxiv.org/abs/2406.12045 ; https://github.com/sierra-research/tau-bench (MIT) | Compares the **final DB state to an annotated goal state**. pass^k measures reliability over k trials. | Annotated goal states. | Report pass^k-style consistency across the 20 paired permuted episodes. |

---

## 4. Domain notes (authoritative sources)

### 4.1 FX: IAS 21 / IFRIC 22
- **Initial recognition** (IAS 21.21): a foreign-currency transaction is recorded in the functional currency at the **spot exchange rate at the date of the transaction**. **IAS 21.22**: the transaction date is when it first qualifies for recognition, and an average rate for a week or month may be used if rates don't fluctuate significantly. Sources: AASB 121 (verbatim-equivalent Australian text) https://standards.aasb.gov.au/node/728 ; https://ifrscommunity.com/knowledge-base/ias-21-effects-of-changes-in-foreign-exchange-rates/ ; ICAEW standard page https://www.icaew.com/technical/financial-reporting/ifrs/ifrs-standards/ias-21-the-effects-of-changes-in-foreign-exchange-rates . (I couldn't fetch the paid IFRS Foundation text, so the paragraph numbers are from these secondary sources, which agree.)
- **Reporting date** (IAS 21.23): monetary items (receivables, cash) are retranslated at the **closing rate**. Non-monetary items stay at the historical rate.
- **Settlement** (IAS 21.28): exchange differences from **settling** monetary items, or translating them at rates different from initial recognition, go to **profit or loss** in the period they arise. That is the realized FX gain/loss.
- **IFRIC 22**: for advance consideration, the transaction date for the rate is the date the **non-monetary prepayment asset or deferred-income liability is first recognised**, with one date per advance if there are several advances. https://www.iasplus.com/en/standards/ifric/ifric-22 ; https://wiki.treasurers.org/wiki/IFRIC_22 .
- **Reference rates as data**: the ECB publishes euro reference rates around 16:00 CET on TARGET working days only, "for information purposes only". There are **no rates on weekends or holidays**, and RUB was suspended in 2024. https://www.ecb.europa.eu/stats/eurofxref . **Implication:** `fx.csv` should have gaps on weekends, and `policy.json` must state the fallback ("last available rate on or before transaction date").
- **EU conversion rule precedent** (Reg. 1103/97 Art. 4–5): rates have six significant figures and **"shall not be rounded or truncated when making conversions"**. **"Inverse rates derived from the conversion rates shall not be used."** Results are rounded to the nearest cent (or smallest unit), and an exact half is **rounded up**. https://eur-lex.europa.eu/eli/reg/1997/1103/2001-01-01/eng ; text read via https://lawplayer.com/eu/act/31997R1103 . This gives us a citable precedent for (a) round once at the end, (b) inverted-rate errors as a realistic error mode, and (c) half-up as a legitimate policy choice.

### 4.2 Revenue cutoff: IFRS 15, ISA 315/330/500, ISA 240/520
- **IFRS 15.38**: at a point in time, revenue is recognised when **control transfers**. Indicators are a present right to payment, legal title, physical possession, risks and rewards, and customer acceptance. Shipping terms matter, and control often passes on delivery to the retailer. https://dart.deloitte.com/USDART/home/publications/deloitte/additional-deloitte-guidance/roadmap-ifrs-us-gaap-comparison/chapter-3-revenues-expenses/3-1-revenue-recognition ; PwC retail https://pwc.com/gx/en/audit-services/ifrs/publications/ifrs-15/ifrs-15-solutions-retail-consumer-industry-pwc.pdf .
- **Cutoff assertion** (ISA 315 Revised 2019): "transactions and events have been recorded in the correct accounting period". It is a transaction-class assertion alongside occurrence, completeness and accuracy. https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f8/technical-articles/assertions.html
- **Cutoff procedure** (substantive, under ISA 330): take documents a few days **before and after** period end (commonly 5–10 days, by risk). **Trace forward** shipping docs → invoices → ledger to test completeness or misdating. **Vouch backward** post-year-end invoices → shipping docs to test premature recognition. Check sequence numbers. Sources: https://open-exam-prep.com/study-guides/cpa-aud/revenue-receivables/revenue-receivables-cycle ; https://jaeyoonyu.github.io/assets/revenue_cycle_audit.html *(teaching sources)*. **Implication:** to make cutoff errors provable, the invoice document must carry a delivery or service date (`line_items[].service_date` or `delivered_at`) that contradicts the ledger posting period.
- **ISA 500 evidence**: "sufficient" means quantity and "appropriate" means relevance + reliability. Reliability is higher for external, original and documentary evidence. "If the audit evidence is unreliable, it will never be appropriate, no matter how much of it the auditor gathers." https://iaasa.ie/wp-content/uploads/2023/10/ISA_500_UpdatedOct2023.pdf ; https://accaglobal.com/uk/en/technical-activities/uk-tech/in-practice/2024/April/Audit-evidence-when-is-enough-enough.html . **Implication for `truth.json`:** define each acceptable evidence set as the **minimal set of records whose joint contents prove the discrepancy**. Example: {duplicate invoice doc, original invoice doc, ledger row of duplicate}. Allow alternative sets, such as payment doc vs bank-side ledger row.
- **ISA 240 para 33** (journal entries at period end): see §2.2.
- **ISA 520** analytical procedures (ratio and trend expectations) need multiple periods, which is out of scope for a single-period episode, so don't design for it.

### 4.3 AR subledger vs GL control account
- The AR control account in the GL must equal the **sum of the customer balances** in the subledger. Differences usually come from manual entries posted directly to the control account (reclass, write-off, correction) that move no customer balance. https://www.accountingtools.com/articles/how-to-reconcile-accounts-receivable.html?no_redirect=true ; Wisconsin DOA AR control reconciliation manual https://doa.wi.gov:443/budget/SCO/WI%20Reconciliation%20Manual/Subsystem%20Reconciliations%20-%20AR%20Control%20Account%201351000%2007012016.pdf .
- In our model the "subledger" is derivable from `invoices.jsonl` + `payments.jsonl` (+ credit notes), so **Σ(open invoice balances in base) == GL balance of 1100 AR** is a jury-checkable pre-injection equality.

### 4.4 Overpayments, credit notes, refunds
- **Credit memo** (no cash): Dr Sales Returns & Allowances (contra-revenue) / Cr AR. **Cash refund**: Dr Sales Returns / Cr Cash. https://courses.lumenlearning.com/suny-finaccounting/chapter/seller-entries/ ; https://www.wallstreetmojo.com/sales-return-journal-entry .
- **Overpayment** sits as a credit on the customer's AR (negative AR) or is reclassified to a liability ("Customer Refunds Payable" / "Advances from customers"). A refund is booked against that, not against expense. https://www.netsuite.com/portal/resource/articles/accounting/overpayment.shtml ; https://beancount.io/blog/2026/09/20/customer-overpayments-unapplied-cash-ar-clearing-guide *(secondary)*.

### 4.5 Journal-entry templates (Dr/Cr), worked in integer minor units
Base/functional currency **KZT (exp 2)**. Amounts were computed with Python `Decimal`, ROUND_HALF_EVEN, round-once.
Conversion: `base_minor = round(doc_minor × Decimal(rate) × 10^(exp_base − exp_doc))`.

| # | Event | Dr | Cr | Example |
|---|---|---|---|---|
| T1 | **Invoice issued** (USD 1,000.00 = 100000 minor, 2026-10-10, rate `478.25`) | 1100 AR — 47825000 | 4000 Revenue — 47825000 | 100000 × 478.25 = 47,825,000 KZT-minor |
| T2 | **Full receipt in USD with realized FX gain** (2026-11-05, rate `482.10`) | 1010 Bank-USD (base value) — 48210000 | 1100 AR — 47825000 ; 7100 Realized FX gain — 385000 | IAS 21.28. Loss case: Dr 7200 FX loss. |
| T3 | **Partial receipt** USD 400.00 (40000) at `482.10` | 1010 Bank — 19284000 | 1100 AR — 19130000 (= 40000 × 478.25, carried rate) ; 7100 FX gain — 154000 | Open AR left = 60000 USD-minor = 28695000 base. Check: 19130000 + 28695000 = 47825000 ✓ |
| T4 | **One payment allocated to 2 invoices** | 1000 Bank — P | 1100 AR (inv A) — a ; 1100 AR (inv B) — b ; ±7100/7200 FX | Two ledger rows with the same `payment_id` and different `invoice_id` (the allocation table). |
| T5 | **Unallocated / on-account receipt** (`invoice_refs` = null) | 1000 Bank — P | 2150 Customer advances (or 1100 AR with `invoice_id` = null) — P | Legit when there's no remittance. It becomes the **"missing allocation" error** only when the payment doc references an invoice but the ledger doesn't allocate it. |
| T6 | **Overpayment then refund** | receipt: 1000 Bank — P | 1100 AR — invoice amt ; 2300 Customer refunds payable — excess | refund: Dr 2300 / Cr 1000 Bank (± FX diff if foreign and the rate moved) |
| T7 | **Credit note** (KZT 50,000.00 = 5000000) | 4100 Sales returns & allowances — 5000000 | 1100 AR — 5000000 | Document type `credit_note` with `ref_invoice_id`. Negative `total_minor` or a separate doc type (pick one and put it in the schema). |
| T8 | **Cash refund with no prior overpayment** (goods returned after payment) | 4100 Sales returns — R | 1000 Bank — R | |
| T9 | **Accrued revenue at period end** (service delivered in Oct, invoiced Nov 3) | Oct 31: 1200 Accrued revenue / contract asset — X | 4000 Revenue — X | Nov 3: Dr 1100 AR / Cr 1200 Accrued revenue. This is the "correct cutoff" counterpart of error type 4. |
| T10 | **Accrued expense + auto-reversal** | Oct 31: 6000 Expense — Y | 2100 Accrued liabilities — Y | Nov 1: reverse (Dr 2100 / Cr 6000). This is a **distractor**: it looks like a duplicate or opposite posting. |
| T11 | **Bank fee deducted from receipt** | 1000 Bank — P−f ; 6500 Bank charges — f | 1100 AR — P | ERPNext: fees get their own row and account and are never netted into FX. |
| T12 | **Early-payment discount** | 1000 Bank — P−d ; 4200 Sales discounts — d | 1100 AR — P | Distractor: "short payment". |
| T13 | *(optional, policy-flagged)* **Period-end revaluation** of open foreign AR (IAS 21.23), closing rate `480.00` | 1100 AR — 175000 | 7150 Unrealized FX gain — 175000 | 100000 × 480.00 = 48000000 − 47825000. Recommend **off** in v1 (`"revalue_open_items": false`) because it adds complexity without adding a required error type. |

Other-currency examples (exponent handling):
- JPY 120,000 (exp 0, minor 120000) at `3.1745` → 120000 × 3.1745 × 10² = **38094000** KZT-minor.
- KWD 1,234.567 (exp 3, minor 1234567) at `1561.20` → 1234567 × 1561.20 × 10⁻¹ = 192740600.04 → **192740600**.

### 4.6 Rounding and tolerance
- **Python `decimal`**: `ROUND_HALF_EVEN` means "round to nearest with ties going to nearest even integer". `ROUND_HALF_UP` means ties go away from zero. The **default context is HALF_EVEN, prec=28**. https://docs.python.org/3/library/decimal.html . The EU euro regulation uses half-up (above). Either is fine if `policy.json` declares it and the generator, matcher and oracle share one `convert()` function.
- **Tie example**: USD 12.50 (1250) at `478.21` → 597762.5 KZT-minor. HALF_EVEN gives **597762** and HALF_UP gives **597763**. The rounding mode alone creates a 1-minor-unit difference.
- **Per-line vs per-total (penny) drift**: 3 lines of USD 19.99 at `478.255`. Per line each is 956031.745 → 956032, so the sum is **2868096**. Converting the total 5997 → 2868095.235 gives **2868095**. That's a 1-unit drift, and the worst case is about ≤ n_lines/2 units. Fowler's Money `allocate` pattern distributes remainders so no value is lost (*secondary*: https://verraes.net/2011/04/fowler-money-pattern-in-php/).
- **Inverse-rate trap**: storing KZT→USD as `0.002091` (6 dp) and inverting reproduces Reg. 1103/97's warning. 100000 USD-minor × (1/0.002091) ≠ 47825000. This makes a nice "wrong rate" error mode: an inverted or truncated rate.
- **Zero/three-decimal pitfalls in practice**: Stripe expects amounts in minor units. JPY and KRW are zero-decimal. ISK and UGX are represented as 2-decimal with "00" for backward compatibility, and HUF/TWD payouts must be divisible by 100. https://docs.stripe.com/currencies . **Use ISO List One exponents and never a PSP's conventions.**
- **Tolerance rule to adopt**:
  - Same-currency comparisons: `tol = 0`.
  - FX-converted comparisons: `tol = ceil(n_conversions / 2)` base-minor units, minimum 1. This matches Beancount's "half of the last digit" logic generalised to n roundings: https://beancount.io/docs/Basics/precision .
  - The matcher's amount tolerance for findings: `max(1, ceil(0.005 × |expected|))` **or** a fixed `≤ 1 major unit`. Pick one, publish it in `policy.json` / README, and **make every injected error at least 100× the rounding tolerance** so rounding noise can't be mistaken for an error and vice versa.

---

## 5. Realistic distractor patterns (legit, but look like errors)

Each pattern exercises the "silence beats false accusations on clean books" test. Each must be **provably fine** from visible records.

| # | Distractor | Looks like | Why it's provably correct (visible evidence) |
|---|---|---|---|
| D1 | **Recurring equal invoices**: monthly subscription, same customer, same amount, sequential invoice numbers, ~30 days apart | Duplicate invoice | Different `line_items[].service_period`; each has its own payment. Same-vendor-same-amount naive test fires (Corpay rule). |
| D2 | **Split payment**: 2–3 receipts for one invoice | Partial allocation | Σ receipts = invoice total; each payment has `invoice_refs` to it. |
| D3 | **Bulk payment** covering several invoices | Unmatched amount | `invoice_refs` lists all invoices, and the sum matches. This is the subset-sum case (SSMP). |
| D4 | **Genuine open balance**: customer paid only part, books correctly show the remainder open | Partial allocation error | Ledger allocation = payment amount exactly. The invoice remains open, which is consistent. |
| D5 | **Credit note then net payment** | Short payment | Credit note doc (`ref_invoice_id`) + T7 posting. Payment = invoice − credit note. |
| D6 | **Cancel-and-reissue**: full credit note on INV-A, new INV-B same amount | Duplicate invoice | Credit note fully offsets A, so net exposure is one invoice. |
| D7 | **Overpayment then refund** | Duplicate payment / outflow anomaly | T6 rows; refund amount = excess. |
| D8 | **FX payment with realized gain/loss row** | Wrong FX amount (base AR ≠ base cash) | Invoice-date and payment-date rates both in `fx.csv`. The FX row equals the difference. |
| D9 | **Weekend transaction** using Friday's rate | Wrong FX date | `policy.json`: "last available rate on or before". `fx.csv` has no Sat/Sun rows (like ECB). |
| D10 | **Timestamp near period boundary, correct in the company timezone** (e.g. `2026-10-31T23:50:00+05:00` is still Oct in the company TZ but Nov 1 in some other TZ, or a UTC-recorded payment that is Oct 31 local) | Cutoff error | `manifest.timezone` + `policy.period_boundary`: the comparison is done in company local time. |
| D11 | **Accrual + day-1 reversal** (T10), or accrued revenue later invoiced (T9) | Duplicate / cutoff | The reversal row references the same accrual. ISA 240 red flags fire on these, as the Schreyer paper notes about "reverse postings, provisions and year-end adjustments". |
| D12 | **Bank fee netted** (T11) | Short payment / partial allocation | Fee row in 6500; allocation to AR = full invoice. |
| D13 | **Early-payment discount** (T12) | Short payment | Invoice `payment_terms` "2/10 net 30"; paid within 10 days; discount row. Pattern from remittance-assigner. |
| D14 | **Same amount, different customers, same day** | Duplicate | Different `entity_id`, different docs. |
| D15 | **Rounding pennies within tolerance** (per-line conversion drift) | FX error | Difference ≤ declared tolerance. |
| D16 | **Near-duplicate invoice numbers that are genuinely different** (e.g. `INV-1042` and `INV-1024` for different goods) | Fuzzy duplicate | Different line items, amounts or dates. This guards against "fuzzy-match everything". |

---

## 6. RECOMMENDATION

### 6.1 Architecture: truth-first, event-sourced, three layers
1. **World / events (hidden, seeded)**: `random.Random(seed)` only, never global `random`, `hash()` or set iteration order. It produces customers, a price list, and a calendar of business events: `Sale(delivery_at, lines, ccy)`, `Receipt(at, ccy, amount, allocations[])`, `CreditNote`, `Refund`, `AccrualRevenue`, `AccrualExpense+Reversal`, `BankFee`, `Discount`. Allocations are first-class, mirroring Odoo `account.partial.reconcile` and python-accounting Assignments: `(payment_id, invoice_id, amount_doc_minor, amount_base_minor, fx_diff_base_minor)`.
2. **Correct books (hidden)**: a **posting engine** maps each event to journal lines via the templates T1–T13. One shared `convert(doc_minor, doc_ccy, date, policy)` is used everywhere: generator, oracle and matcher.
3. **Recorded books (visible)**: a copy of layer 2 plus documents, then **error injectors mutate the recorded layer** and append to `provenance.json` `{issue_id, type, injector, params, before:[rows], after:[rows], causal_change}`. `truth.json` is derived from the diff, never hand-written.

**Pre-injection invariants** (assert in the generator, export as a test, and these are what the jury checks):
- I1: every journal entry balances in base (Σdr = Σcr per `je_id`).
- I2: the trial balance nets to 0.
- I3: Σ open invoice balances (from invoices − allocations − credit notes) = GL 1100 AR.
- I4: each invoice `total_minor` = Σ `line_items`.
- I5: each ledger base amount = `convert(doc amount, rate per policy)`.
- I6: each ledger timestamp's local date falls inside the period per `policy`.
- I7: each payment's `amount_minor` = Σ its allocations (+ fees/discounts).

**Post-injection**: assert the declared invariant(s) now fail **by exactly `expected_amount_minor`**, all other invariants still hold, and the evidence set alone reproduces the failure. The last check is an independent re-check function that reads only the visible files.

**Errors should keep double-entry balanced** (I1/I2 still hold), as in real life. Discrepancies show up only in document↔ledger or subledger↔GL comparisons, so trivial "TB doesn't balance" tricks don't work.

### 6.2 Ledger schema decisions
- `ledger.csv` uses exactly the case columns. **Amounts are in base/functional currency** (IAS 21.21: books are kept in the functional currency), so the `currency` column is always the base currency. The transaction currency lives on the documents. This is the SAP `DMBTR` (local) vs `WRBTR` (document) split from the Schreyer data. The agent must convert, which is exactly the skill under test.
- `row_id` = `JE{n:05d}-{line:02d}`, so journal grouping is visible without an extra column. For the paired permutation/ID-renaming checks, map IDs through a seeded bijection after generation, so truth IDs remap mechanically.
- Explicit `null` for `invoice_id` / `payment_id` where there's no link (case requirement).
- `invoices.jsonl` adds `doc_type ∈ {invoice, credit_note}`, `ref_invoice_id`, `line_items[{sku, qty, unit_price_minor, amount_minor, service_date}]` and `delivered_at`. Cutoff evidence needs a delivery or service date, and the remaining fields are optional extensions inside `line_items`.

### 6.3 Minimal chart of accounts (≈16 accounts, IFRS-flavoured, 4-digit)
`1000 Bank-<BASE>`, `1010 Bank-USD`, `1020 Bank-EUR` (bank accounts per currency, valued in base), `1100 Accounts receivable (control)`, `1200 Accrued revenue / contract asset`, `2000 Accounts payable` (only if the AP mirror is added), `2100 Accrued liabilities`, `2150 Customer advances / unapplied cash`, `2300 Customer refunds payable`, `3000 Equity / opening balance`, `4000 Revenue`, `4100 Sales returns & allowances`, `4200 Sales discounts`, `6000 Operating expenses`, `6500 Bank charges`, `7100 Realized FX gain`, `7200 Realized FX loss` (`7150 Unrealized FX`, only if revaluation is on).
Open the period with an opening-balance JE (Dr bank and AR carried in / Cr 3000) so prior-period invoices paid in this period exist. That's needed for cutoff and interacting scenarios.

### 6.4 `policy.json` (all knobs the agent needs to decide correctness)
```json
{
  "fx_rate_source": "fx.csv",
  "fx_rate_date": {"invoice": "issued_at_local_date", "payment": "timestamp_local_date"},
  "fx_missing_date_rule": "last_available_on_or_before",
  "fx_rate_direction": "units of base per 1 unit of source; never invert",
  "conversion": "base_minor = round(doc_minor * rate * 10**(exp_base-exp_doc)); convert once per document total",
  "rounding_mode": "ROUND_HALF_EVEN",
  "period": {"start": "2026-10-01T00:00:00+05:00", "end_exclusive": "2026-11-01T00:00:00+05:00", "timezone": "Asia/Almaty"},
  "revenue_recognition": "at delivered_at (local date)",
  "realized_fx_posting_date": "payment date",
  "revalue_open_items": false,
  "amount_tolerance_minor": {"same_currency": 0, "fx_converted": 1},
  "currency_exponents": {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0, "KWD": 3},
  "iso4217_list_one_published": "2026-09-17"
}
```
Converting once per document total avoids per-line penny drift. If you want D15 distractors, convert per line and set the tolerance to `ceil(n_lines/2)`.

### 6.5 Error-injection mechanics (each mutates recorded books plus documents, and records the causal change)

| Type | Injector | Canonical `affected_ids` | `expected_amount_minor` / currency | Minimal evidence sets |
|---|---|---|---|---|
| **E1 Duplicate invoice** | Clone an invoice doc with a new `doc_id`. Its `invoice_id` is either the same, or normalized-equal (`INV-1042` → `INV1042`, `O`↔`0`, a transposition), or the same lines with a ±2-day date shift (Oversight patterns). Post a second T1. Optionally the customer paid only once. | `[dup_invoice_id]` | Duplicate total, in the invoice currency (or base, but pick one and document it) | {orig doc, dup doc, dup ledger rows} |
| **E2 Missing / partial allocation** | The payment doc has `invoice_refs:[X]`, but in the ledger the credit goes to 2150 unapplied (missing) or only part goes to AR-X with the rest to 2150 (partial). | `[payment_id, invoice_id]` | Unallocated amount (base) | {payment doc, invoice doc, ledger rows of payment} |
| **E3a Wrong FX date** | Ledger base amount uses the rate from a different date: invoice date for a payment, d−1, or month-average when the policy says spot (IAS 21.22 average-rate misuse). | `[doc_id]` | Base difference | {doc, fx row correct date, ledger row} |
| **E3b Wrong FX rate** | Inverted, truncated, or a different currency pair's rate (Reg. 1103/97 inverse-rate trap). | same | same | same |
| **E4 Cutoff** | `delivered_at` is in the next period but the ledger posts in the current one (premature revenue), or the reverse (an Oct delivery booked in Nov). Make some boundary cases timezone-sensitive (D10 is the clean twin). | `[invoice_id]` | Invoice base amount misallocated | {invoice doc (delivered_at), ledger rows, policy} |
| **X1 Interacting**: partial foreign-currency payment at period boundary | USD receipt on the last day, local-time `23:40+05:00`. The ledger (a) posts it in the next period (cutoff), (b) uses the next day's rate (FX date), and (c) allocates only part (partial). | Truth lists **2–3 linked issues** sharing evidence; or one compound issue if you choose. Decide and document. | per issue | |
| **X2 Interacting**: duplicate of an invoice that also has a credit note, in foreign currency | The duplicate's FX conversion uses the wrong date. The credit note offsets the original only. | | | |

Injector rules:
- Never inject into a record that is already part of another issue unless it's an X scenario.
- Size every error ≥ 100 × tolerance.
- Each injector has an independent on/off and count in `config` (the case requires "independently configurable" types).
- Clean companies use `n_errors = 0` but **keep all distractor generators on**.

### 6.6 Reuse vs write
- **Write ourselves** (pure Python 3.11 stdlib: `decimal`, `random.Random`, `zoneinfo`, `json`, `csv`, `hashlib`): the generator, posting engine, injectors, invariant checker, matcher and env. Total maybe 1.5–2.5k LOC. There's no dependency risk for the offline 2-CPU evaluation.
- **Borrow concepts (no code)**:
  - Odoo partial-reconcile data model (LGPL).
  - ERPNext FX gain/loss posting and the "separate rows for fees/rounding" rule (GPL).
  - python-accounting Assignments (MIT; could vendor if desired, but unnecessary).
  - Beancount tolerance rule and balance assertions (GPL).
  - AMLSim config→labels file split (Apache-2.0).
  - FinSim error taxonomy (no license, so ideas only).
  - Schreyer's DMBTR/WRBTR split and global/local anomaly framing.
  - Reconify's output categories for the baseline reconciler (MIT).
- **Data to vendor**: a pinned ISO 4217 exponent subset from SIX List One (facts, tiny). Synthetic FX rates come from a seeded random walk around plausible levels, with weekend gaps, so no real rate data and no network are needed (the case forbids real financial data).
- **Avoid as dependencies**: Faker (version-dependent output; use our own seeded name lists), Babel (CLDR digits ≠ ISO), and any GPL code in the shipped package.
- **Matcher**: fields only, no text similarity (the lesson from arXiv 2506.17282).
  - Normalize IDs (case/whitespace), require exact canonical `affected_ids`, compare the amount within the published tolerance, require an exact currency, and evidence ⊇ one acceptable set.
  - Use deterministic Hungarian / greedy-by-score assignment with a published tie-break.
  - Report **per-type** P/R/F1 (the Gronewald finding: aggregate hit rates hide type blind spots).

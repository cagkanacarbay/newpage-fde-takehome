# Longevity / Geroscience RAG Corpus

**Field:** the biology of aging — mechanisms of aging, biomarkers of biological age, and pharmacological or dietary interventions that aim to extend healthspan and lifespan (geroscience).

All 27 documents are peer-reviewed, open-access research papers or reviews drawn from the PubMed Central Open Access subset, retrieved as publisher PDFs via Europe PMC. Every file carries a Creative Commons licence (26 x CC BY, 1 x CC BY-NC-ND) and was verified after download: valid `%PDF-` header, `file` reports a PDF document, size above 100 KB, and full text extractable with `pypdf` on every page.

Documents 001-020 are the original biology-of-aging set. Documents 021-027 are a second, equation-focused batch — see "Papers 021-027: the equation-focused batch" below.

## Why this set is good for testing retrieval

**Dense topical overlap.** The corpus deliberately clusters around a handful of concepts rather than spreading thin. Six papers discuss epigenetic clocks; five discuss cellular senescence; four evaluate a small-molecule geroprotector. A retriever cannot succeed by keyword-matching "aging" — nearly every document contains it. Discriminating between *DNA methylation clocks in mice* and *DNA methylation clocks in insects* requires real semantic resolution.

**Built-in contradictions.** Several pairs disagree, so a question can be answered two ways depending on what is retrieved, which makes faithfulness and citation quality measurable:

- **Telomeres cause senescence?** Berardi 2026 (004) argues the single shortest telomere drives both genome instability and replicative senescence; Coquette 2026 (005) shows telomere dysfunction and proteostasis decline define *distinct* senescence pathways.
- **Do senolytics reverse aging?** Hickson 2019 (015) and Justice 2019 (016) report senescent-cell clearance in first-in-human trials; Kasamoto 2026 (017) finds DNA-methylation signatures of senescence are *not* reversed by senolytic treatment.
- **Does chemical reprogramming rejuvenate?** Schoenfeldt 2025 (019) reports it ameliorates hallmarks of aging and extends lifespan; Mitchell 2026 (020) reports the same class of intervention causes toxic lipid-droplet accumulation in vivo.
- **Is metformin a geroprotector?** Mohammed 2021 (012) is an explicitly critical review of the evidence; Ivimey-Cook 2025 (013) finds rapamycin, but *not* metformin, mirrors dietary-restriction lifespan extension.
- **Which aging framework is right?** Sanada 2025 (001) works inside the hallmarks-of-aging framework; García-Barranquero 2025 (002) contrasts it against the competing SENS damage-repair vision.
- **Do epigenetic clocks generalise?** Koch 2026 (010) builds a pan-mammalian age predictor; Maleszka 2025 (011) reports there is still no evidence for an environmentally responsive epigenetic clock in insects.

**Mixed document structure.** The set spans a 1-page PNAS letter, 3-to-8-page reviews, and 32-page research articles with extensive supplementary-style content — so chunking strategy is exercised across very different document lengths. Numeric content is heavy: clinical-trial outcome tables (015, 016), mouse survival and healthspan statistics (014, 018), regression coefficients for clock construction (007, 008), and a fourteen-biomarker head-to-head comparison table (009). Questions such as "which biomarker predicted mortality best" are answerable only from a table, which tests PDF table extraction.

**Named-entity traps.** Nearby entities are easy to confuse: rapamycin vs trametinib, dasatinib vs quercetin vs fisetin, PhenoAge vs GrimAge vs DunedinPoAm, senescence vs senolysis. Study organisms vary (human, mouse, killifish, insect, cross-mammal), so "what was the effect on lifespan?" is under-specified unless the retriever surfaces the right species.

**Temporal spread.** 2018-2026, including two landmark biomarker papers (Levine 2018 PhenoAge, Belsky 2020 pace-of-aging) that later work cites and revises — useful for testing whether a system prefers recent evidence or conflates a method with its critique.

## Corpus

| # | Filename | Title | First author | Year | Venue | DOI / URL | Licence | Pages | Size |
|---|---|---|---|---|---|---|---|---|---|
| 001 | `001-sanada-2025-hallmarks-of-aging-therapeutic-targets.pdf` | Targeting the hallmarks of aging: mechanisms and therapeutic opportunities | Sanada F | 2025 | Frontiers in cardiovascular medicine | [10.3389/fcvm.2025.1631578](https://doi.org/10.3389/fcvm.2025.1631578) | CC BY | 6 | 490 KB |
| 002 | `002-garcia-barranquero-2025-sens-vs-hallmarks-of-aging-debate.pdf` | SENS vs. the hallmarks of aging: competing visions, shared challenges | García-Barranquero P | 2025 | Biogerontology | [10.1007/s10522-025-10248-5](https://doi.org/10.1007/s10522-025-10248-5) | CC BY | 11 | 556 KB |
| 003 | `003-ajoolabady-2025-hallmarks-mechanisms-cellular-senescence.pdf` | Hallmarks and mechanisms of cellular senescence in aging and disease | Ajoolabady A | 2025 | Cell death discovery | [10.1038/s41420-025-02655-x](https://doi.org/10.1038/s41420-025-02655-x) | CC BY | 8 | 1.1 MB |
| 004 | `004-berardi-2026-shortest-telomere-drives-senescence.pdf` | Both genome instability and replicative senescence stem from the shortest telomere in telomerase-negative cells | Berardi P | 2026 | Nature communications | [10.1038/s41467-026-70352-z](https://doi.org/10.1038/s41467-026-70352-z) | CC BY | 19 | 2.6 MB |
| 005 | `005-coquette-2026-telomere-dysfunction-proteostasis-senescence-pathways.pdf` | Telomere Dysfunction and Proteostasis Decline Define Distinct Pathways of Cellular Senescence in the Human Respiratory Tract | Coquette C | 2026 | Aging cell | [10.1111/acel.70512](https://doi.org/10.1111/acel.70512) | CC BY | 16 | 12.0 MB |
| 006 | `006-sun-2026-nad-homeostasis-antiaging-framework.pdf` | An integrated anti-aging framework targeting NAD+ homeostasis, mitochondrial quality control, and redox stability: Roles of NMN/NR, PQQ, and EGT | Sun Y | 2026 | Redox biology | [10.1016/j.redox.2026.104191](https://doi.org/10.1016/j.redox.2026.104191) | CC BY | 28 | 5.9 MB |
| 007 | `007-levine-2018-phenoage-epigenetic-biomarker-lifespan.pdf` | An epigenetic biomarker of aging for lifespan and healthspan | Levine ME | 2018 | Aging | [10.18632/aging.101414](https://doi.org/10.18632/aging.101414) | CC BY | 19 | 3.8 MB |
| 008 | `008-belsky-2020-dunedin-pace-of-aging-blood-test.pdf` | Quantification of the pace of biological aging in humans through a blood test, the DunedinPoAm DNA methylation algorithm | Belsky DW | 2020 | eLife | [10.7554/elife.54870](https://doi.org/10.7554/elife.54870) | CC BY | 25 | 1.3 MB |
| 009 | `009-vetter-2026-comparing-fourteen-biomarkers-of-aging.pdf` | Comparing fourteen consensus biomarkers of aging: epigenetic pace of aging as the strongest predictor of mortality in BASE-II | Vetter VM | 2026 | Biomarker research | [10.1186/s40364-026-00909-z](https://doi.org/10.1186/s40364-026-00909-z) | CC BY | 13 | 2.3 MB |
| 010 | `010-koch-2026-pan-epigenetic-age-prediction-mammals.pdf` | Pan-Epigenetic Age Prediction in Mammals | Koch Z | 2026 | Aging cell | [10.1111/acel.70380](https://doi.org/10.1111/acel.70380) | CC BY | 12 | 4.3 MB |
| 011 | `011-maleszka-2025-no-epigenetic-clock-in-insect.pdf` | Still no evidence for an environmentally responsive epigenetic clock in an insect | Maleszka R | 2025 | Proceedings of the National Academy of Sciences of the United States of America | [10.1073/pnas.2523241122](https://doi.org/10.1073/pnas.2523241122) | CC BY-NC-ND | 1 | 137 KB |
| 012 | `012-mohammed-2021-metformin-anti-aging-critical-review.pdf` | A Critical Review of the Evidence That Metformin Is a Putative Anti-Aging Drug That Enhances Healthspan and Extends Lifespan | Mohammed I | 2021 | Frontiers in endocrinology | [10.3389/fendo.2021.718942](https://doi.org/10.3389/fendo.2021.718942) | CC BY | 24 | 6.4 MB |
| 013 | `013-ivimey-cook-2025-rapamycin-not-metformin-dietary-restriction.pdf` | Rapamycin, Not Metformin, Mirrors Dietary Restriction-Driven Lifespan Extension in Vertebrates: A Meta-Analysis | Ivimey-Cook ER | 2025 | Aging cell | [10.1111/acel.70131](https://doi.org/10.1111/acel.70131) | CC BY | 15 | 723 KB |
| 014 | `014-gkioni-2025-trametinib-rapamycin-combined-healthspan.pdf` | The geroprotectors trametinib and rapamycin combine additively to extend mouse healthspan and lifespan | Gkioni L | 2025 | Nature aging | [10.1038/s43587-025-00876-4](https://doi.org/10.1038/s43587-025-00876-4) | CC BY | 32 | 9.0 MB |
| 015 | `015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human.pdf` | Senolytics decrease senescent cells in humans: Preliminary report from a clinical trial of Dasatinib plus Quercetin in individuals with diabetic kidney disease | Hickson LJ | 2019 | EBioMedicine | [10.1016/j.ebiom.2019.08.069](https://doi.org/10.1016/j.ebiom.2019.08.069) | CC BY | 11 | 2.4 MB |
| 016 | `016-justice-2019-senolytics-idiopathic-pulmonary-fibrosis-trial.pdf` | Senolytics in idiopathic pulmonary fibrosis: Results from a first-in-human, open-label, pilot study | Justice JN | 2019 | EBioMedicine | [10.1016/j.ebiom.2018.12.052](https://doi.org/10.1016/j.ebiom.2018.12.052) | CC BY | 10 | 520 KB |
| 017 | `017-kasamoto-2026-senolytics-do-not-reverse-senescence-methylation.pdf` | DNA Methylation Signatures of Cellular Senescence Are Not Reversed by Senolytic Treatment | Kasamoto J | 2026 | Aging cell | [10.1111/acel.70430](https://doi.org/10.1111/acel.70430) | CC BY | 17 | 2.4 MB |
| 018 | `018-moulds-2026-graded-calorie-restriction-epigenetic-ageing.pdf` | Graded Calorie Restriction Causes Graded Slowing of Epigenetic Ageing in Mice | Moulds TP | 2026 | Aging cell | [10.1111/acel.70342](https://doi.org/10.1111/acel.70342) | CC BY | 18 | 2.7 MB |
| 019 | `019-schoenfeldt-2025-chemical-reprogramming-extends-lifespan.pdf` | Chemical reprogramming ameliorates cellular hallmarks of aging and extends lifespan | Schoenfeldt L | 2025 | EMBO molecular medicine | [10.1038/s44321-025-00265-9](https://doi.org/10.1038/s44321-025-00265-9) | CC BY | 24 | 5.4 MB |
| 020 | `020-mitchell-2026-chemical-reprogramming-lipid-droplet-toxicity.pdf` | In Vivo Chemical Reprogramming Is Associated With a Toxic Accumulation of Lipid Droplets Hindering Rejuvenation | Mitchell W | 2026 | Aging cell | [10.1111/acel.70390](https://doi.org/10.1111/acel.70390) | CC BY | 18 | 9.7 MB |
| 021 | `021-flietner-2025-unifying-theory-of-aging-mortality.pdf` | A unifying theory of aging and mortality | Flietner V | 2025 | Scientific Reports | [10.1038/s41598-025-11454-4](https://doi.org/10.1038/s41598-025-11454-4) | CC BY | 16 | 4.3 MB |
| 022 | `022-nielsen-2024-gompertz-law-subcomponent-interdependencies.pdf` | The Gompertz Law emerges naturally from the inter-dependencies between sub-components in complex organisms | Nielsen PY | 2024 | Scientific Reports | [10.1038/s41598-024-51669-5](https://doi.org/10.1038/s41598-024-51669-5) | CC BY | 11 | 2.2 MB |
| 023 | `023-pyrkov-2021-loss-of-resilience-lifespan-limit.pdf` | Longitudinal analysis of blood markers reveals progressive loss of resilience and predicts human lifespan limit | Pyrkov TV | 2021 | Nature Communications | [10.1038/s41467-021-23014-1](https://doi.org/10.1038/s41467-021-23014-1) | CC BY | 10 | 1.1 MB |
| 024 | `024-oswal-2022-hierarchical-process-model-celegans-aging.pdf` | A hierarchical process model links behavioral aging and lifespan in C. elegans | Oswal N | 2022 | PLOS Computational Biology | [10.1371/journal.pcbi.1010415](https://doi.org/10.1371/journal.pcbi.1010415) | CC BY | 28 | 4.1 MB |
| 025 | `025-farrell-2024-epigenetic-pacemaker-aging-moderators.pdf` | Identifying epigenetic aging moderators using the epigenetic pacemaker | Farrell C | 2024 | Frontiers in Bioinformatics | [10.3389/fbinf.2023.1308680](https://doi.org/10.3389/fbinf.2023.1308680) | CC BY | 10 | 1.4 MB |
| 026 | `026-dolejs-2025-strehler-mildvan-correlation-mortality.pdf` | The Strehler-Mildvan correlation as a valuable tool for monitoring the long-term health status of a population | Dolejs J | 2025 | Frontiers in Public Health | [10.3389/fpubh.2025.1627111](https://doi.org/10.3389/fpubh.2025.1627111) | CC BY | 8 | 1.8 MB |
| 027 | `027-gavrilova-2025-compensation-effect-of-mortality.pdf` | The Compensation Effect of Mortality: A Global Analysis of Human Populations | Gavrilova NS | 2025 | Ageing and Longevity Research | [10.53941/alr.2025.100004](https://doi.org/10.53941/alr.2025.100004) | CC BY | 15 | 0.5 MB |

**Totals:** 27 documents, 425 pages, 92.7 MB, ~2.08M characters of extractable text.

## Topic coverage

| # | Topic |
|---|---|
| 001 | Framework / review |
| 002 | Framework / review |
| 003 | Senescence mechanisms |
| 004 | Telomeres |
| 005 | Telomeres / proteostasis |
| 006 | NAD+ / metabolism |
| 007 | Epigenetic clocks |
| 008 | Pace-of-aging biomarker |
| 009 | Biomarker comparison |
| 010 | Epigenetic clocks (cross-species) |
| 011 | Epigenetic clocks (negative result) |
| 012 | Metformin (critical review) |
| 013 | Rapamycin vs metformin |
| 014 | Rapamycin combination |
| 015 | Senolytics (clinical) |
| 016 | Senolytics (clinical) |
| 017 | Senolytics (negative result) |
| 018 | Caloric restriction |
| 019 | Partial/chemical reprogramming |
| 020 | Reprogramming (adverse effect) |
| 021 | Mathematical model of aging (Gompertz derivation) |
| 022 | Mathematical model of aging (Gompertz derivation) |
| 023 | Stochastic model of biological resilience |
| 024 | Survival/hazard modeling (C. elegans) |
| 025 | Epigenetic clocks (pacemaker regression) |
| 026 | Mortality law (Strehler-Mildvan correlation) |
| 027 | Mortality law (Gompertz-Makeham, compensation effect) |

## Papers 021-027: the equation-focused batch

Documents 001-020 are prose- and table-heavy but rarely display typeset mathematics. `docs/parsing.html` flagged this as an open question: *"whether equations survive [parsing] — zero `formula` items across the four papers [tested]. Telling 'rendered as images' from 'discarded' needs a visual diff against the PDFs."* Papers 021-027 were added specifically to give that question something to test against — each was confirmed, by downloading the PDF and running text extraction, to contain genuine displayed equations rather than prose descriptions of a method.

- **021 Flietner** — analytical derivation of the Gompertz law from a network model: Poisson transition rates, a mean-field damage fraction, rescaled ODEs, a numbered theorem and proof. The most equation-dense document in the corpus.
- **022 Nielsen** — 11 numbered displayed equations, including the Gompertz hazard, a logistic mean-field ODE and its solution, and a master equation over transition probabilities.
- **023 Pyrkov** — an autocorrelation function and a Langevin stochastic differential equation for loss of physiological resilience.
- **024 Oswal** — a battery of typeset hazard functions (Gompertz, Weibull, Weibull-with-frailty, inverse Gaussian) plus Wiener-process equations; also the longest document in the corpus (28 pages), useful for chunking tests independent of its math.
- **025 Farrell** — the elastic-net penalized-regression objective behind the Epigenetic Pacemaker model, displayed as an equation rather than described in prose.
- **026 Dolejs** — the Gompertz law and the Strehler-Mildvan correlation as two numbered equations, fit against 273 years of Swedish mortality data.
- **027 Gavrilova** — the canonical Gompertz-Makeham hazard equation and its compensation-effect regression, fit across 241 national populations.

**Naive extraction already corrupts this math.** Running `pdftotext` against these PDFs during verification (not Docling — that test is still open) showed the exact failure modes a parser must avoid: minus signs silently dropped from a Langevin equation (023), equation numbers stranded on their own line and detached from the equation body (022, 026), an `=` sign vanishing entirely from a penalized-regression objective (025), and custom font encodings turning Greek letters and operators into mojibake (023). These are recorded here as expected findings for whichever parser is tested next, not as a Docling result.

## Provenance and licensing

Every item was confirmed to be in the PubMed Central Open Access subset via the NCBI OA service (`https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=<PMCID>`) before download, and PDFs were fetched from Europe PMC (`https://europepmc.org/articles/<PMCID>?pdf=render`). No paywall was bypassed and no unauthorised source was used. CC BY and CC BY-NC-ND both permit redistribution with attribution; the single CC BY-NC-ND item (011) is usable for this non-commercial demonstration corpus provided it is not modified. Attribution for each work is the DOI link in the table above.

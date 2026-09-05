# Venue compliance note

Target: **Annals of Actuarial Science** (Cambridge University Press), research article,
initial submission. Fallback target: **ASTIN Bulletin**.

| item | rule | source | checked |
|---|---|---|---|
| length | "Submitted articles should be no more than 35 pages"; excess to supplementary material | AAS, [preparing your materials](https://www.cambridge.org/core/journals/annals-of-actuarial-science/information/author-instructions/preparing-your-materials) | 2026-09-05 |
| length (fallback venue) | papers over 30 pages "are advised to consider splitting their contribution into shorter contributions" | ASTIN Bulletin, [preparing your materials](https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/information/author-instructions/preparing-your-materials) | 2026-09-05 |
| supplement | published online alongside the article, not in the journal pages; **not typeset or copyedited**, so supplied exactly as it is to appear | AAS + ASTIN, same pages | 2026-09-05 |
| abstract | 150-200 words | AAS | 2026-09-05 |
| references | Harvard system | AAS | 2026-09-05 |
| required sections | Title; Author name(s); Abstract; Keywords; Correspondence details; Main text; Acknowledgements; Competing Interest Statement; Data Availability Statement; Funding Statement; References; Appendices | AAS | 2026-09-05 |
| template | CUP LaTeX template on Overleaf is recommended but **not required** | AAS | 2026-09-05 |
| initial submission format | single PDF incorporating all figures and tables | ASTIN | 2026-09-05 |
| open access | ASTIN Bulletin is Gold Open Access as of 31 July 2026 -- an APC applies there; check waiver eligibility before choosing that venue | ASTIN, instructions for contributors | 2026-09-05 |

## Decisions taken against these rules

- **Page target 32, not 35.** Leaves margin for the difference between this 11pt A4 build and
  CUP production. The 110-page version in `paper/` is untouched and remains the full record.
- **Bibliography stays `natbib` + `plainnat` author-year.** Harvard in substance; CUP applies
  its own style at production. `agsm.bst` (true Harvard) needs the `harvard` package rather
  than `natbib` and would mean rewriting every citation command for no gain at submission.
- **Supplement is a separate PDF**, S-numbered, sharing `../tables/` and `../figures/` with the
  main manuscript so no generated exhibit is ever duplicated in source.
- **No registered outcome lives only in the supplement.** Every pre-registered hypothesis keeps
  its verdict and its headline numbers in the main body, with an explicit pointer to the
  supplementary table carrying the full per-cell evidence.

## Still to confirm with the author

- Funding statement: is there a grant to name, or is this unfunded?
- Competing interests: assumed none.
- Data availability: HMD requires free registration and forbids redistribution; the statement
  points at `data/MANIFEST.sha256` for the pinned vintage.
- Repository URL: currently "released on acceptance" (the GitHub repo is private).

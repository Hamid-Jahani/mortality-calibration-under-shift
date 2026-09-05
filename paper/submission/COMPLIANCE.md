# Venue compliance note

Target: **ASTIN Bulletin -- The Journal of the IAA** (Cambridge University Press), research
article, initial submission. Chosen 2026-09-05. Rules below verified the same day against
[preparing your materials](https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/information/author-instructions/preparing-your-materials).

| item | rule (quoted) | done |
|---|---|---|
| length | "Authors intending to submit papers exceeding 30 pages are advised to consider splitting their contribution into shorter contributions." | **NO -- manuscript is 47 pp** |
| initial submission format | "Initial submissions to the editor must be in PDF format in a single file that incorporates all figures and tables." | yes -- `manuscript.pdf` embeds every exhibit |
| first page | "The first page of each paper should start with the title, the name(s) of the author(s), an abstract and a list of keywords. An institutional affiliation can be placed between the name(s) of the author(s) and the abstract." | yes |
| contact details | "The address and full contact details of at least one of the authors should be typed at the end of the paper following the references." | yes -- `Author's address` section after the bibliography |
| competing interests | "All authors must include a competing interest declaration in their main manuscript file." | yes -- in `backmatter.tex` |
| data and code | authors should "provide data and code during the review process"; "for accepted papers data and code should be made available as supplementary material" | yes -- data availability statement offers the repository to editor and referees on request |
| references | year in parentheses after the authors; journal italic, volume bold, e.g. "Jewell, W.S. (1975) Regularity conditions for exact credibility. *ASTIN Bulletin,* **8**, 336--341." | `apalike` -- matches the year placement and italics; volume is not bold, which CUP restyles at production |
| mathematics | "All mathematical symbols and equations...must be typeset using recognized mathematical typesetting software such as LaTeX" | yes |
| supplementary material | "published online alongside your article"; not typeset or copyedited | yes -- `supplement.pdf`, supplied as it appears |
| open access | "As of 31 July 2026, all articles are published on a Gold Open Access basis"; "All authors are able to publish on this basis in the journal, irrespective of their funding situation or affiliation"; APC waiver requests are handled through the journal's open access options | **APC not yet settled -- request the waiver in writing before submitting** |
| at acceptance | "the author(s) will be asked to provide all relevant files in electronic format, including separate files for each of the figures" | figures are generated PDFs in `paper/figures/`, ready |

## Open decisions

1. **47 pages against a 30-page steer.** ASTIN's own advice is to split rather than to lean on
   the supplement. The natural cut is the audit (H1-H4) as one paper and the actuarial
   propagation (H5) plus the twin-crises analysis as the other.
2. **APC.** Gold OA applies to every article since 31 July 2026. Confirm the waiver before
   submitting, not after acceptance.

## Decisions taken against these rules

- **The 110-page version in `paper/` is untouched** and remains the full record.
- **Bibliography is `natbib` + `apalike`.** ASTIN's examples put the year in parentheses
  after the authors, which `plainnat` does not; `apalike` does. Volume numbers are not bold,
  which CUP restyles at production.
- **Supplement is a separate PDF**, S-numbered, sharing `../tables/` and `../figures/` with the
  main manuscript so no generated exhibit is ever duplicated in source.
- **No registered outcome lives only in the supplement.** Every pre-registered hypothesis keeps
  its verdict and its headline numbers in the main body, with an explicit pointer to the
  supplementary table carrying the full per-cell evidence.

## Still to confirm with the author

- Funding statement: is there a grant to name, or is this unfunded? (currently: unfunded)
- Competing interests: assumed none.
- Data availability: HMD requires free registration and forbids redistribution; the statement
  points at `data/MANIFEST.sha256` for the pinned vintage.
- Repository URL: currently "released on acceptance" (the GitHub repo is private).

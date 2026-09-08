# Venue compliance note

Target: **Annals of Actuarial Science** (Cambridge University Press), research article,
initial submission. Chosen 2026-09-08 after ASTIN Bulletin declined manuscript
ASTIN-2026-0172 on 8 September 2026 -- desk decision, not peer review: "we present a nice
and interesting case study, however, for our publications we mainly require new
methodological developments. We therefore recommend that you send your paper to a more
applied journal that publishes case studies." AAS is that journal, and the framing this
project locked on 2026-08-24 (reliability audit, no new architecture) is what AAS publishes.

Rules verified 2026-09-05 against
[preparing your materials](https://www.cambridge.org/core/journals/annals-of-actuarial-science/information/author-instructions/preparing-your-materials).

| item | rule | status |
|---|---|---|
| length | "Submitted articles should be no more than 35 pages"; excess to supplementary material | **NO -- manuscript is 47 pp** |
| abstract | 150-200 words | yes -- 199 |
| references | Harvard system | yes -- natbib + apalike, year in parentheses after the authors |
| structure | Title; Author name(s); Abstract; Keywords; Correspondence details; Main text; Acknowledgements; Competing Interest Statement; Data Availability Statement; Funding Statement; References; Appendices | yes -- correspondence moved into the front matter, statements before the references |
| supplement | published online alongside the article, not typeset or copyedited | yes -- `supplement.pdf`, 87 pp, supplied as it appears |
| template | CUP's LaTeX template on Overleaf is recommended but not required | using article 11pt; swapping is preamble-only |
| AI use | CUP AI Contributions to Research Content Policy applies | **UNRESOLVED -- no declaration in the manuscript; see below** |

## Open decisions

1. **47 pages against a 35-page cap.** Unlike ASTIN's 30-page steer this is a stated limit,
   so roughly 12 pages have to come out. The exhibits are already abridged and total about
   5 pages; the reduction has to come from prose, which stands at 23,286 words.
2. **AI use declaration.** CUP's policy covers AAS as it covered ASTIN. An AI use declaration has to be settled before submission.

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

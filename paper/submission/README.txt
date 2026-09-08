LaTeX sources -- "Do Mortality Prediction Intervals Survive a Pandemic?
A Pre-Registered Calibration Audit of Ten Forecasting Families and Seven
Uncertainty Mechanisms Across the COVID-19 Break"

Hamid Jahani, Tarbiat Modares University. ORCID 0000-0002-2302-7674.

BUILD
  latexmk -pdf manuscript.tex     -> manuscript.pdf   (37 pp)
  latexmk -pdf supplement.tex     -> supplement.pdf   (87 pp)

Both compile with pdfTeX and BibTeX on a standard TeX Live installation
(built and verified on TeX Live 2026). No non-standard classes or styles:
article 11pt with amsmath, booktabs, longtable, threeparttable, natbib
(apalike) and hyperref. Sources are ASCII-only, so the XeTeX/LuaTeX route
renders identically; the preamble is engine-conditional.

FILES
  manuscript.tex     the submitted manuscript
  supplement.tex     the online supplement, S-numbered throughout
  preamble.tex       shared preamble, loaded by both after \documentclass
  frontmatter.tex    abstract and keywords
  backmatter.tex     acknowledgements, competing interests, data
                     availability and funding statements
  sections/          the nine manuscript sections
  supp/              the six supplement sections
  tables/            every table fragment. Names ending -main are the
                     abridged one-page views used in the manuscript; the
                     unabridged fragments of the same data are used by the
                     supplement
  figures/           the twelve figures, as PDFs
  references.bib     78 cited works

PROVENANCE
Every table fragment in tables/ is generated, not hand-written, by
scripts/make_tables.py in the project repository from the per-cell result
files; each carries a header naming the script, the date and the input
files. The abridged views come from the same script under --variant main
and read the same inputs, so no number in the manuscript is retyped from
the supplement. Figures are generated the same way.

DATA
Human Mortality Database (https://www.mortality.org), June 2026 vintage,
downloaded 25 August 2026. HMD requires free registration and its terms do
not permit redistribution, so no HMD file is included here. The exact
vintage is pinned by SHA-256 in data/MANIFEST.sha256 in the project
repository, which is available to the editor and referees on request.

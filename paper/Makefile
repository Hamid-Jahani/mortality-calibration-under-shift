# Build the paper.
#
#   make            -> main.pdf via latexmk -pdf if latexmk is on PATH,
#                      otherwise via tectonic (see below)
#   make latexmk    -> force the latexmk route
#   make tectonic   -> force the tectonic route
#   make clean      -> remove build products (keeps the PDF)
#   make distclean  -> remove build products and the PDF
#
# ON THIS MACHINE (2026-08-26): PATH points at E:\texlive\2026\bin\windows,
# but the E: drive that holds TeX Live is not mounted, so latexmk/pdflatex are
# absent. The working build route is tectonic 0.17.0 (self-contained XeTeX
# engine; https://tectonic-typesetting.github.io), which resolves natbib and
# BibTeX itself. Two things it needs here:
#   * the socks ALL_PROXY variable unset (its bundle download fails through
#     the socks proxy; the http(s) proxy variables are fine) -- done below;
#   * a tectonic binary: either on PATH, or pass TECTONIC=<path>, e.g.
#       make tectonic TECTONIC=/c/tools/tectonic/tectonic.exe
#     (the session that produced this draft used a copy in its scratchpad).
# The sources are ASCII-only and the preamble is engine-conditional, so both
# routes render the same document.
#
# `make` itself is not installed in the Git Bash here; the two commands below
# can be run by hand from paper/ if needed:
#   latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
#   env -u ALL_PROXY tectonic -X compile --keep-logs --keep-intermediates main.tex

LATEXMK  ?= latexmk
TECTONIC ?= tectonic
MAIN      = main
SOURCES   = $(MAIN).tex sections/*.tex tables/*.tex references.bib

.PHONY: all latexmk tectonic clean distclean

all:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then $(MAKE) latexmk; else $(MAKE) tectonic; fi

latexmk: $(SOURCES)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

tectonic: $(SOURCES)
	env -u ALL_PROXY $(TECTONIC) -X compile --keep-logs --keep-intermediates $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf

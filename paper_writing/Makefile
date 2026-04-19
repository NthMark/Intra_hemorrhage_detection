# Makefile for building the IEEE LaTeX paper
#
# Usage:
#   make          - build PDF (artifacts go into build/)
#   make watch    - rebuild on any .tex/.bib change (requires inotify-tools)
#   make clean    - remove build/ directory and main.pdf
#   make wordcount - approximate word count (requires texcount)
#
# All intermediate artefacts (*.aux, *.log, *.bbl, ...) stay inside build/.

MAIN     := main
LATEX    := pdflatex
BIBTEX   := bibtex
BUILDDIR := build
FLAGS    := -interaction=nonstopmode -halt-on-error -output-directory=$(BUILDDIR)

.PHONY: all clean watch wordcount

all: $(BUILDDIR)/$(MAIN).pdf
	@cp $(BUILDDIR)/$(MAIN).pdf $(MAIN).pdf
	@echo "Done -> $(MAIN).pdf"

$(BUILDDIR)/$(MAIN).pdf: $(MAIN).tex sections/*.tex bibliography/references.bib | $(BUILDDIR)
	$(LATEX) $(FLAGS) $(MAIN)
	$(BIBTEX) $(BUILDDIR)/$(MAIN) || true
	$(LATEX) $(FLAGS) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

watch:
	@echo "Watching for changes... (Ctrl-C to stop)"
	@while inotifywait -e modify -r sections/ bibliography/ $(MAIN).tex 2>/dev/null; do \
		$(MAKE) all; \
	done

wordcount:
	@texcount -inc -sum $(MAIN).tex 2>/dev/null || echo "Install texcount: sudo apt install texlive-extra-utils"

clean:
	rm -rf $(BUILDDIR) $(MAIN).pdf

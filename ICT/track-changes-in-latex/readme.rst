======================================
abstract & keywords track changes absent:
======================================

latexdiff --type=UNDERLINE --preamble=suppress-deleted.sty --append-textcmd="abstract,keywords" main.tex main_revised_I.tex > diff.tex


=================================
Abstract & keywords track changes present:
=================================

latexdiff --type=UNDERLINE --preamble suppress-deleted.sty main.tex main_revised_I.tex >> diff.tex

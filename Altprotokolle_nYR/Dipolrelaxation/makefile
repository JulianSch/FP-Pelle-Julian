all: Alphastrahlung.pdf

Alphastrahlung.tex: plot.py Messdaten1_korr.txt Messdaten1.txt Messdaten2_korr.txt Messdaten2.txt Messdaten3.txt
	python plot.py
Alphastrahlung.pdf: content/auswertung.tex
	#content/diskussion.tex content/theorie.tex content/durchfuerung.tex
	lualatex Alphastrahlung.tex

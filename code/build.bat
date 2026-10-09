@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: freqlists from contingency table
:: python .\freqlists.py --top 0 ..\data\3_contingency\ ..\results\freqlists\
:: generate keyword 100
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords100 --vocab content --top 100 --min-doc-len 1000 
:: matrix of distances
:: python .\keywords-matrix.py --metric rbo --vocab content ..\results\keywords100 ..\results\romans19e-100motscles-rbo.tsv
:: plot formulas without parameters
:: python .\pcoa.py ..\results\romans19e-100motscles-rbo.tsv ..\results\romans19e-100motscles-specifs --exclude "CF" "DF" "*α*" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules de spécificité."
:: plots formulas with parameters
:: python .\pcoa-families.py ..\results\romans19e-100motscles-rbo.tsv --output-prefix ..\results\romans19e-100motscles-spec-families --families "subTF-IDF" "TF-IDF" HGT "G²" "χ²" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules paramétrables de spécificité."
:: 1000 keywords 
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords-content --vocab content --top 0 --min-doc-len 1000 --scorer hgta3 hgta2 hgta1.47 txm hgta0.75   g2a1.4 g2 g2a0.76  tfidf tfidfa3  tfidfa1.14 tfidfa0.77 tfidfa0.57  subtfidf subtfidfa0.56 subtfidfa0.4 subtfidfa0.3 subtfidfa0.2   tfidfa16 logdice tf  chi2 chi2a2 chi2a1.5 df cf
:: python .\keywords-average.py -o ..\results\keywords-average\keywords-content-average.txt ..\results\keywords-content\*.txt
:: plot keywords dispersion
:: python .\keywords-dispersion.py ..\data\3_contingency\ "..\results\keywords-verne\*.txt" verne1870a-42 --output-dir ..\results\keywords-verne\ --freq cf  --top 820 --cmap inferno_r
:: for stopwords dispersion
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords-stops --vocab nocaps --top 0 --min-doc-len 1000 --scorer hgta3 hgta2 hgta1.47 txm hgta0.75   g2a1.4 g2 g2a0.76   tfidf tfidfa3  tfidfa1.14 tfidfa0.77 tfidfa0.57   subtfidf subtfidfa0.56 subtfidfa0.4 subtfidfa0.32 subtfidfa0.2   tfidfa16 logdice tf  chi2 chi2a2 chi2a1.5
:: python .\topwords-dispersion.py ..\data\3_contingency\terms.tsv ..\results\keywords-stops\*.txt --output-dir ..\results\topwords-dispersion\ --cols 3 --words 200 --absent ignore --freq cf --dispersion box --scorer   chi2a0.43 g2 txm   subtfidf tfidf subtfidfa0.32 
:: top words by specif
python words-dispersion-compare.py  ..\data\3_contingency\terms.tsv ..\results\keywords-stops\*.txt --output-dir ../results/topwords-compare --words 100 --width 8  --height 25 --absent ignore --freq cf --scorer txm g2  subtfidfa0.32 tfidf 

:: keywords for an author
:: python .\keywords-average.py -o ..\results\keywords-average\keywords-all-average.txt ..\results\keywords-stops\*.txt

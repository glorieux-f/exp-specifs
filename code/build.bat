@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: freqlists from contingency table
:: python freqlists.py --select "verne*" "balzac*" "sand*" "zola*" "dumas*" --top 1000 ..\data\3_contingency\romans19e- ..\results\freqlists

:: command to get coocs 
:: python cooc.py --corpus ../data/3_contingency/romans19e- --pivot indigène --window 3 5 10 30 100 --scorers simplemaths tscore tf mi logratio txm --vocab content --top 30 -o ../results/cooc "../data/2_lemma/verne*.tsv"
:: generate keyword 100
:: python .\keywords.py  ..\data\3_contingency\romans19e-  ..\results\keywords100\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab "all" "content" "nostops"  --top 100 --min-doc-len 1000  --scorer bm25 bm25k100b1 btfidf cf chi2 df extf g2 logdice logratio mi milogf minsens simplemaths simplemathsk1 subtfidf tf tficf tfidf  tscore txm
python .\keywords.py  ..\data\3_contingency\romans19e-  ..\results\keywords100-bm25-smaths\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab "content"  --top 100 --min-doc-len 1000  --scorer tf bm25 bm25k100b1 simplemathsk0.001 simplemathsk0.01 simplemathsk0.1 simplemathsk1 simplemathsk10 simplemathsk100 simplemathsk1000 simplemathsk10000 simplemathsk100000 tfidfa0 tfidfa0.33 tfidfa0.66 tfidfa1 tfidfa2 tfidfa4 tfidfa8 tfidfa16 extf


:: matrix of distances
:: python .\keywords-matrix.py --metric rbo --vocab content ..\results\keywords100 ..\results\keywords-pcoa\romans19e-100motscles-rbo.tsv
:: plot formulas without parameters
:: python .\pcoa.py ..\results\keywords-pcoa\romans19e-100motscles-rbo.tsv ..\results\keywords-pcoa\romans19e-100motscles-specifs --exclude "cf" "df" "*alpha*" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules de spécificité."
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
:: python words-dispersion-compare.py  ..\data\3_contingency\terms.tsv ..\results\keywords-stops\*.txt --output-dir ../results/topwords-compare --words 100 --width 8  --height 25 --absent ignore --freq cf --scorer txm g2  subtfidfa0.32 tfidf 

:: keywords for an author
:: python .\keywords-average.py -o ..\results\keywords-average\keywords-all-average.txt ..\results\keywords-stops\*.txt

:end

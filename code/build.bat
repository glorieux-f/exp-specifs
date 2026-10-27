@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: freqlists from contingency table
:: python freqlists.py --select "verne*" "balzac*" "sand*" "zola*" "dumas*" --top 1000 ..\data\3_contingency\romans19e- ..\results\freqlists

:: command to get coocs 
:: python cooc.py --corpus ../data/3_contingency/romans19e- --pivot indigène --window 3 5 10 30 100 --scorers simplemaths tscore tf mi logratio txm --vocab content --top 30 -o ../results/cooc "../data/2_lemma/verne*.tsv"

:: plot non parametric formula keyword 100
python .\keywords.py  ..\data\3_contingency\romans19e-  ..\results\keywords100\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab "all" "content" "nostops"  --top 100 --min-doc-len 1000  --scorer bm25 bm25k1000b1 btfidf cf chi2 df extf g2 logdice logratio lrc mi milogf minsens poissondev simplemaths simplemathsk1 subtfidf subtfidfa0.27 tf tficf tfidf tfidfa0.77 tscore txm
python .\keywords-matrix.py --metric rbo --vocab content ..\results\keywords100 ..\results\keywords-pcoa\romans19e-keywords100-rbo.tsv
python .\pcoa.py ..\results\keywords-pcoa\romans19e-keywords100-rbo.tsv --exclude "cf" "df" "*alpha*" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Spectre des formules de spécificité."

:: plot parametric formula keyword 100
:: python .\keywords.py  ..\data\3_contingency\romans19e-  ..\results\keywords100param\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab content --top 100 --min-doc-len 1000  --scorer bm25k10000b1 bm25k1000b1 bm25k100b1 bm25k10b1 bm25k6b1 bm25k3b1  bm25k1b1 bm25k0.1b1 bm25k0.001b1  bm25k0.0001b1  g2a0 g2a0.25 g2a0.5 g2a0.75 g2a1 g2a1.5 g2a2 g2a3 g2a4 g2a8 g2a16 lrca0 lrca0.25 lrca0.5 lrca0.75 lrca1 lrca1.5 lrca2 lrca2.5 lrca3.3 lrca4 lrca8 lrca16 simplemathsk100000 simplemathsk10000 simplemathsk1000 simplemathsk500 simplemathsk170 simplemathsk100 simplemathsk60 simplemathsk20 simplemathsk10 simplemathsk5 simplemathsk1 simplemathsk0.1 simplemathsk0.01 simplemathsk0.001 simplemathsk0.0001 subtfidfa0 subtfidfa0.05 subtfidfa0.1 subtfidfa0.27 subtfidfa0.5 subtfidfa0.75 subtfidfa1 subtfidfa1.5 subtfidfa2 subtfidfa4 subtfidfa8 subtfidfa16 tfidfa0 tfidfa0.25 tfidfa0.5 tfidfa0.77 tfidfa1 tfidfa1.5 tfidfa2 tfidfa3 tfidfa4 tfidfa8 tfidfa16 hgta0 hgta0.25 hgta0.5 hgta0.75 hgta1 hgta1.5 hgta2 hgta3 hgta4 hgta8 hgta16
:: python .\keywords-matrix.py --metric rbo --vocab content ..\results\keywords100param ..\results\keywords-pcoa\romans19e-keywords100param-rbo.tsv
:: python .\pcoa-families.py ..\results\keywords-pcoa\romans19e-keywords100param-rbo.tsv --output-prefix ..\results\keywords-pcoa\romans19e-keywords100param-rbo --families simplemaths subtfidfa bm25 lrca  tfidfa hgta g2a --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules paramétrables de spécificité."

:: Hapax
python .\hapax.py ..\data\3_contingency\romans19e-  ..\results\hapax\  --vocab all content --min-doc-len 1000  --scorer tfidfa100 g2 txm tfidfa0.77 simplemaths subtfidfa0.27 lrcplus tf
python hapax_heatmap.py ..\results\hapax\romans19e-hapax-content.tsv --max-rank 500
python hapax_heatmap.py ..\results\hapax\romans19e-hapax.tsv  --max-rank 500

:: 1000 keywords 
:: python .\keywords.py ..\data\3_contingency\romans19e-  ..\results\keywords1000\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab all content --top 1000 --min-doc-len 1000  --scorer lrc lrcplus g2 txm poissondev tfidfa0.77 subtfidfa0.27 simplemaths tf
:: python .\keywords.py ..\data\3_contingency\romans19e-  ..\results\keywords100spec\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab all content --top 100 --min-doc-len 1000  --scorer lrc lrcplus g2 txm poissondev tfidfa0.77 subtfidfa0.27  simplemaths tf
:: python .\keywords.py ..\data\3_contingency\romans19e-  ..\results\keywords-diag\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab all --diagnose --min-doc-len 1000  --scorer lrc lrcplus g2 txm poissondev tfidfa0.77 subtfidfa0.27  simplemaths tf txmneg 
:: python .\keywords-average.py -o ..\results\keywords-average\keywords1000-all-average.txt ..\results\keywords1000\*-all-*.txt
:: python .\keywords-average.py -o ..\results\keywords-average\keywords1000-content-average.txt ..\results\keywords1000\*-content-*.txt
:: python .\keywords-average.py -o ..\results\keywords-average\keywords100-all-average.txt ..\results\keywords100spec\*-all-*.txt


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

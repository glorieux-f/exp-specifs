@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: freqlists from contingency table
:: python freqlists.py --select "verne*" "balzac*" "sand*" "zola*" "dumas*" --top 1000 ..\data\3_contingency\romans19e- ..\results\freqlists

:: command to get coocs 
:: python cooc.py --corpus ../data/3_contingency/romans19e- --pivot indigène --window 3 5 10 30 100 --scorers simplemaths tscore tf mi logratio txm --vocab content --top 30 -o ../results/cooc "../data/2_lemma/verne*.tsv"

:: plot non parametric formula keyword 100
:: python .\keywords.py  ..\data\3_contingency\romans19e-  ..\results\keywords100\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab all content nostops nocaps  --top 100 --min-doc-len 1000  --scorer bm25 bm25k1000b1 btfidf cf chi2 df extf g2 logdice logratio lrc lrcplus mi milogf minsens poissondev simplemaths simplemathsk1 subtfidf subtfidfa0.37 tf tficf tfidf tfidfa0.77 tscore txm tfidfa100
:: python .\keywords-matrix.py --metric rbo --p 0.97 --vocab content ..\results\keywords100 ..\results\keywords-pcoa\romans19e-keywords100-content.tsv
:: python .\keywords-matrix.py --metric rbo --p 0.97 --vocab all ..\results\keywords100 ..\results\keywords-pcoa\romans19e-keywords100-all.tsv
:: python .\pcoa.py ..\results\keywords-pcoa\romans19e-keywords100-all.tsv --exclude df cf btfidf --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Spectre des formules de spécificité."

:: python .\scorers_benchmark.py  ..\data\3_contingency\romans19e- --scorer tf tfidfa0.77 subtfidfa0.37 g2 txm poissondev simplemaths lrc  --vocab content --repeat 3  --output ../results/scorers-benchmark.tsv

:: plot parametric formula keyword 100
:: python .\keywords.py  ..\data\3_contingency\romans19e-  ..\results\keywords100param\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab content all nocaps --top 100 --min-doc-len 1000  --scorer bm25k1000b1 bm25k10b1 bm25k6b1 bm25k3b1  bm25k1b1 bm25k0.1b1 bm25k0.00001b1 g2a0 g2a0.5 g2a0.75 g2a1 g2a1.5 g2a2 g2a5 g2a10 g2a20 lrca0 lrca1.5 lrca0.3 lrca0.5 lrca1 lrca1.5 lrca2 lrca2.5 lrca3.3 lrca5 lrca10 simplemathsk100000 simplemathsk1000 simplemathsk170 simplemathsk50 simplemathsk20 simplemathsk10 simplemathsk5 simplemathsk2.5 simplemathsk1 simplemathsk0.1 simplemathsk0.0001 subtfidfa0 subtfidfa0.05 subtfidfa0.1 subtfidfa0.37 subtfidfa0.5 subtfidfa0.75 subtfidfa1 subtfidfa1.5 subtfidfa2 subtfidfa5 subtfidfa10  tfidfa0 tfidfa0.5 tfidfa0.77 tfidfa1 tfidfa1.5 tfidfa2 tfidfa3 tfidfa5 tfidfa10 hgta0 hgta0.5 hgta0.75 hgta1 hgta1.5 hgta2 hgta3 hgta4 hgta8 hgta16
:: python .\keywords-matrix.py --metric rbo --p 0.97 --vocab content ..\results\keywords100param ..\results\keywords-pcoa\romans19e-keywords100param-content.tsv
:: python .\pcoa-families.py ..\results\keywords-pcoa\romans19e-keywords100param-content.tsv --output-prefix ..\results\keywords-pcoa\romans19e-keywords100param-content --families simplemaths subtfidfa bm25 lrca  tfidfa hgta g2a --flip 1 --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules paramétrables de spécificité."

:: python .\keywords.py ..\data\3_contingency\romans19e-  ..\results\keywords-diag\  --select "verne*" "balzac*" "sand*" "zola*" "dumas*"  --vocab all --diagnose --min-doc-len 1000  --scorer lrc lrcplus g2 txm poissondev tfidfa0.77 subtfidfa0.37  simplemaths tf txmneg 

:: Hapax
python .\hapax.py ..\data\3_contingency\romans19e-  ..\results\hapax\  --vocab all content --min-doc-len 1000  --scorer tfidfa100 g2 txm tfidfa0.77 simplemaths subtfidfa0.37 lrcplus lrc tf
python hapax_heatmap.py ..\results\hapax\romans19e-hapax-content.tsv --max-rank 100
:: python hapax_heatmap.py ..\results\hapax\romans19e-hapax.tsv  --max-rank 100

:: Topwords
:: python .\ranks.py ..\data\3_contingency\romans19e- ..\results\topwords\ --dfrank 1 200 --min-doc-len 1000 --vocab all --scorer tfidfa100 g2 txm lrcplus tfidfa0.77 subtfidfa0.37  simplemaths tf --force
:: python .\ranks_heatmap.py  ..\results\topwords\romans19e-ranks-all-min1000-dfrank1-200.tsv --max-rank 100


:end

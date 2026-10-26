#!/usr/bin/env python3
"""Calculer les rangs des termes exclusifs (df=1) par formule.

Entrées attendues dans le même dossier :
    romans19e-docs(1).tsv
    romans19e-terms(1).tsv
    romans19e-postings(1).tsv
    scorers.py

Sorties :
    zone-specificite-log-documents.tsv
    zone-specificite-logscale-df1.tsv

Le filtre documentaire est volontairement simple :
    on garde tous les documents contenant au moins un terme de df=1.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

BASE = Path(__file__).resolve().parent
CODES = [
    'tfidfa16',
    'lrcplus',
    'g2',
    'txm',
    'tfidfa0.77',
    'poissondev',
    'subtfidfa0.27',
    'simplemaths',
    'tf',
]


class Corpus:
    """Statistiques minimales requises par scorers.py."""

    def __init__(self, documents: pd.DataFrame, termes: pd.DataFrame) -> None:
        self.n_docs = len(documents)
        self.doc_ids = np.arange(self.n_docs + 1, dtype=np.int64)
        self.doc_len = np.r_[0, documents.doc_len.to_numpy(dtype=np.int64)]
        self.collection_len = int(self.doc_len.sum())
        self.avg_doc_len = self.collection_len / self.n_docs
        self.cf = np.r_[0, termes.cf.to_numpy(dtype=np.int64)]
        self.df = np.r_[0, termes.df.to_numpy(dtype=np.int64)]

    def tf(self, term_id: int, doc_id: int) -> int:
        raise NotImplementedError


def load_scorers_module(base: Path):
    """Charger scorers.py depuis le dossier de travail."""
    spec = importlib.util.spec_from_file_location('scorers_local', base / 'scorers.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['scorers_local'] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    scorers = load_scorers_module(BASE)
    make_scorer = scorers.make_scorer

    documents = pd.read_csv(BASE / 'romans19e-docs(1).tsv', sep='\t')
    termes = pd.read_csv(BASE / 'romans19e-terms(1).tsv', sep='\t')
    occurrences = pd.read_csv(BASE / 'romans19e-postings(1).tsv', sep='\t', dtype=np.int32)

    matrice = csr_matrix(
        (occurrences.tf.to_numpy(), (occurrences.doc_id.to_numpy(), occurrences.term_id.to_numpy())),
        shape=(len(documents) + 1, len(termes) + 1),
    )
    del occurrences
    matrice.sort_indices()
    corpus = Corpus(documents, termes)

    selection = []
    for doc_id in range(1, len(documents) + 1):
        debut, fin = matrice.indptr[doc_id:doc_id + 2]
        term_ids = matrice.indices[debut:fin]
        if term_ids.size and np.any(corpus.df[term_ids] == 1):
            selection.append(doc_id)

    lignes = []
    infos = []
    for code in CODES:
        scorer = make_scorer(corpus, code)
        infos.append({
            'formule': code,
            'name': scorer.name,
            'label': scorer.label,
            'auto_parameter_note': (
                f'k={scorer.k:g}' if hasattr(scorer, 'k') else ''
            ),
        })
        for doc_id in selection:
            debut, fin = matrice.indptr[doc_id:doc_id + 2]
            term_ids = matrice.indices[debut:fin]
            tf = matrice.data[debut:fin]
            scores = scorer.score_terms(doc_id, term_ids, tf)
            valid = ~np.isnan(scores)
            term_ids = term_ids[valid]
            tf = tf[valid]
            scores = scores[valid]
            exclusifs = corpus.df[term_ids] == 1
            if not np.any(exclusifs):
                continue

            # Reprise de la logique utilisée plus haut :
            # TF a un second critère spécifique, les autres partagent la même logique.
            if code == 'tf':
                ordre = np.lexsort((term_ids, corpus.cf[term_ids], -tf, -scores))
            else:
                ordre = np.lexsort((term_ids, -tf, -scores))

            rangs = np.flatnonzero(exclusifs[ordre]) + 1
            lignes.append((doc_id, code, scorer.name, scorer.label, int(rangs[0]), int(rangs[-1])))

    detail = pd.DataFrame(
        lignes,
        columns=['doc_id', 'formule', 'name', 'label', 'rang_premier_df1', 'rang_dernier_df1'],
    )
    detail.to_csv(BASE / 'zone-specificite-log-documents.tsv', sep='\t', index=False)

    resume = []
    for code in CODES:
        groupe = detail[detail.formule == code]
        resume.append({
            'formule': code,
            'name': groupe['name'].iloc[0],
            'label': groupe['label'].iloc[0],
            'nb_documents': len(groupe),
            'p10_premier': groupe['rang_premier_df1'].quantile(0.10),
            'p25_premier': groupe['rang_premier_df1'].quantile(0.25),
            'p50_premier': groupe['rang_premier_df1'].quantile(0.50),
            'p50_dernier': groupe['rang_dernier_df1'].quantile(0.50),
            'p75_dernier': groupe['rang_dernier_df1'].quantile(0.75),
            'p90_dernier': groupe['rang_dernier_df1'].quantile(0.90),
        })
    pd.DataFrame(resume).to_csv(
        BASE / 'zone-specificite-logscale-df1.tsv',
        sep='\t',
        index=False,
        float_format='%.3f',
    )
    pd.DataFrame(infos).to_csv(BASE / 'zone-specificite-formules.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()

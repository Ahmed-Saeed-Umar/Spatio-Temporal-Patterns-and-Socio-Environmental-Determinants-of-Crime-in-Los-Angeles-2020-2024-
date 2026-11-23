"""Discover modus-operandi (MO) patterns from the `Mocodes` column.

`Mocodes` is a space-separated list of numeric LAPD MO codes, e.g. "0416 0334 2004 1822".
The codes are tags (a *set*, order carries little meaning) and the dataset does not ship their
text descriptions, so a language model has nothing to read. We therefore treat each crime as a
"document" whose "words" are its MO codes and use a classic text-mining pipeline:

    TF-IDF  ->  TruncatedSVD (dense low-dimensional space)  ->  K-Means

TF-IDF down-weights codes that appear in almost every crime (e.g. "1822"), so clusters are
driven by the *distinctive* combinations of codes.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

NO_MO = -1  # label for crimes without any MO code


@dataclass
class MOModel:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    normalizer: Normalizer
    kmeans: MiniBatchKMeans
    k: int
    silhouette_scan: pd.DataFrame

    def embed(self, mocodes: pd.Series) -> np.ndarray:
        X = self.vectorizer.transform(mocodes)
        return self.normalizer.transform(self.svd.transform(X))

    def predict(self, mocodes: pd.Series) -> np.ndarray:
        """Cluster label for every crime; crimes with no MO code get NO_MO (-1)."""
        labels = np.full(len(mocodes), NO_MO, dtype=int)
        has_mo = mocodes.str.len().to_numpy() > 0
        if has_mo.any():
            labels[has_mo] = self.kmeans.predict(self.embed(mocodes[has_mo]))
        return labels


def fit_mo_model(mocodes: pd.Series, k_range=range(4, 11), n_components=15,
                 sample_size=150_000, random_state=42) -> MOModel:
    """Fit TF-IDF + SVD + K-Means on a random sample; choose k by silhouette score."""
    mo = mocodes[mocodes.str.len() > 0]
    sample = mo.sample(min(sample_size, len(mo)), random_state=random_state)

    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=50, sublinear_tf=True)
    X = vec.fit_transform(sample)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    norm = Normalizer(copy=False)
    Z = norm.fit_transform(svd.fit_transform(X))

    rng = np.random.default_rng(random_state)
    sil_idx = rng.choice(len(Z), size=min(8_000, len(Z)), replace=False)
    rows, best = [], None
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=random_state, batch_size=4096)
        lab = km.fit_predict(Z)
        with config_context(working_memory=128):      # keep the distance matrix small
            score = silhouette_score(Z[sil_idx], lab[sil_idx])
        rows.append({"k": k, "silhouette": round(float(score), 4), "inertia": float(km.inertia_)})
        if best is None or score > best[0]:
            best = (score, k, km)
    return MOModel(vec, svd, norm, best[2], best[1], pd.DataFrame(rows))


def describe_clusters(model: MOModel, df: pd.DataFrame, labels: np.ndarray, top_codes=6):
    """One row per cluster: size, most distinctive MO codes, dominant crime types, arrest rate.

    "Distinctive" codes = codes whose share inside the cluster is highest compared with their
    share in all MO-tagged crimes (lift), restricted to codes common inside the cluster.
    """
    d = df.assign(mo_cluster=labels)
    d = d[d["mo_cluster"] != NO_MO]
    vocab = np.array(model.vectorizer.get_feature_names_out())
    binary = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, use_idf=False,
                             norm=None, vocabulary=vocab)
    B = binary.fit_transform(d["mocodes"]).tocsr()
    overall = np.asarray(B.mean(axis=0)).ravel()

    out = []
    for c in sorted(d["mo_cluster"].unique()):
        idx = np.where(d["mo_cluster"].to_numpy() == c)[0]
        share = np.asarray(B[idx].mean(axis=0)).ravel()
        lift = share / np.maximum(overall, 1e-9)
        ok = np.where(share >= 0.15)[0]                      # common enough inside the cluster
        ranked = ok[np.argsort(-lift[ok])][:top_codes]
        sub = d.iloc[idx]
        grp = sub["crime_group"].value_counts(normalize=True).head(2)
        out.append({
            "cluster": c,
            "crimes": len(sub),
            "share_%": round(100 * len(sub) / len(d), 1),
            "distinctive_MO_codes": " ".join(vocab[ranked]),
            "top_crime_groups": "; ".join(f"{g} ({p:.0%})" for g, p in grp.items()),
            "arrest_rate_%": round(100 * sub["arrest"].mean(), 1),
            "median_age": sub["vict_age"].median(),
        })
    return pd.DataFrame(out)

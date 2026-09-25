"""Cluster-aware candidate selection: a measured and removed stage.

This module is not part of the pipeline. It is kept because two
retention rules were built, measured and reported, and the code that
produced those numbers should be available alongside them.

The idea was to preserve diversity in the selected set by retaining
candidates per cluster rather than globally. Two rules were measured:

    score_scaled  retention scaled by cluster quality. It reproduced
                  global ranking almost exactly, because a small number
                  of clusters supplied most of the final pool.

    uniform       a fixed fraction from every cluster. It preserved
                  diversity but collapsed quality to barely above
                  applying no selection at all.

The stage was removed because diversity is resolved upstream by corpus
deduplication, which costs nothing at selection time and carries no
quality penalty. If output diversity is unsatisfactory, the corpus is
the place to intervene.

Note that the ordinary runs also contain clustering cells, but with
retention set to 0.99 for every cluster, which keeps essentially
everything and is equivalent to global ranking. Only the two rules here
change the outcome.
"""

import numpy as np
import pandas as pd


def cluster_embeddings(
    embeddings,
    pca_dim=50,
    umap_dim=15,
    min_cluster_size=30,
    min_samples=5,
    seed=42,
):
    """PCA whitening, UMAP reduction, then HDBSCAN.

    Returns (labels, info). Points HDBSCAN cannot assign are labelled -1
    and handled separately by the retention rules.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(embeddings)
    pca = PCA(n_components=min(pca_dim, X.shape[1]), whiten=True, random_state=seed)
    Xp = pca.fit_transform(X)
    explained = float(pca.explained_variance_ratio_.sum())

    try:
        import umap

        reducer = umap.UMAP(n_components=umap_dim, random_state=seed)
        Xu = reducer.fit_transform(Xp)
    except ImportError:
        print("[warning] umap-learn not installed; clustering on PCA output")
        Xu = Xp

    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        ).fit_predict(Xu)
    except ImportError:
        from sklearn.cluster import KMeans

        print("[warning] hdbscan not installed; falling back to KMeans")
        labels = KMeans(n_clusters=20, random_state=seed).fit_predict(Xu)

    n_clusters = len({c for c in labels if c != -1})
    n_noise = int((labels == -1).sum())
    info = {
        "pca_explained": explained,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_frac": n_noise / max(1, len(labels)),
    }
    print(
        f"clustering | {n_clusters} clusters | "
        f"{n_noise} unassigned ({info['noise_frac']*100:.1f}%)"
    )
    return labels, info


def cluster_stats(scored_df, label_col="cluster", score_col="combined_score"):
    """Mean score, count and range per cluster."""
    g = scored_df.groupby(label_col)[score_col]
    stats = pd.DataFrame(
        {
            "cluster": g.mean().index,
            "avg_score": g.mean().values,
            "count": g.count().values,
            "min": g.min().values,
            "max": g.max().values,
        }
    ).sort_values("avg_score", ascending=False)
    return stats.reset_index(drop=True)


def retain_score_scaled(
    scored_df,
    stats,
    min_keep=0.05,
    max_keep=1.00,
    steepness=3.0,
    noise_percentile=0.50,
    label_col="cluster",
    score_col="combined_score",
):
    """Retention scaled by where a cluster sits in this run's own score range.

    Scaling within the run rather than against absolute thresholds is
    deliberate: the composite score is min-max normalised per run, so
    fixed thresholds go stale as soon as the score range shifts.
    """
    avg = dict(zip(stats["cluster"], stats["avg_score"]))
    real = sorted(c for c in avg if c != -1)
    scores = np.array([avg[c] for c in real])

    lo, hi = scores.min(), scores.max()
    pos = np.ones_like(scores) if hi - lo < 1e-9 else (scores - lo) / (hi - lo)
    keep_fracs = min_keep + (max_keep - min_keep) * (pos ** steepness)

    parts = []
    for cid, frac in zip(real, keep_fracs):
        chunk = scored_df[scored_df[label_col] == cid].sort_values(
            score_col, ascending=False
        )
        n_keep = max(1, int(round(len(chunk) * frac)))
        parts.append(chunk.head(n_keep))

    noise = scored_df[scored_df[label_col] == -1]
    if len(noise):
        floor = float(scored_df[score_col].quantile(noise_percentile))
        parts.append(noise[noise[score_col] >= floor])

    out = pd.concat(parts).reset_index(drop=True)
    print(f"score-scaled retention | kept {len(out)} of {len(scored_df)}")
    return out


def retain_uniform(
    scored_df,
    stats,
    keep_frac=0.20,
    noise_floor=0.30,
    label_col="cluster",
    score_col="combined_score",
):
    """A fixed top fraction from every cluster, regardless of quality.

    This preserves diversity by construction and is the reason the rule
    was tried. It also admits the top of clusters that are uniformly
    poor, which is why quality collapses.
    """
    avg = dict(zip(stats["cluster"], stats["avg_score"]))
    real = sorted(c for c in avg if c != -1)

    parts = []
    for cid in real:
        chunk = scored_df[scored_df[label_col] == cid].sort_values(
            score_col, ascending=False
        )
        n_keep = max(1, int(len(chunk) * keep_frac))
        parts.append(chunk.head(n_keep))

    noise = scored_df[scored_df[label_col] == -1]
    if len(noise):
        parts.append(noise[noise[score_col] >= noise_floor])

    out = pd.concat(parts).reset_index(drop=True)
    print(f"uniform {keep_frac*100:.0f}% retention | kept {len(out)} of {len(scored_df)}")
    return out


def pool_concentration(selected_df, label_col="cluster", top_n=2):
    """Share of the selected pool drawn from the largest few clusters.

    This is the diagnostic that explains the score-scaled result: when a
    couple of clusters supply most of the pool, per-cluster retention
    cannot differ much from global ranking.
    """
    counts = selected_df[label_col].value_counts()
    top = counts.head(top_n).sum()
    return {
        "top_n": top_n,
        "from_top_n": int(top),
        "total": int(len(selected_df)),
        "fraction": float(top / max(1, len(selected_df))),
    }

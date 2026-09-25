"""Composite score and global selection.

    combined = bio_weight * bio_score + pll_weight * normalised_PLL

with 0.3 / 0.7. The two terms play different roles: the biological score
is a cheap filter that removes obvious abnormalities, and the likelihood
term does the finer ranking among what survives.

Selection is a global top-K over the scored pool. Cluster-aware
retention was built and measured; it is kept in cluster_ablation.py and
is not part of this path.

Note on interpreting the score: it is min-max normalised within a run,
so the same numeric value means different things in different runs. It
supports ranking within a pool and nothing else. To compare quality
across experiments, fold an unselected sample instead.
"""

import numpy as np
import pandas as pd


def combine(bio_scores, plddt_proxy, bio_weight=0.3, pll_weight=0.7):
    return bio_weight * np.asarray(bio_scores) + pll_weight * np.asarray(plddt_proxy)


def score_frame(sequences, bio_scores, plddt_proxy, bio_weight=0.3, pll_weight=0.7):
    combined = combine(bio_scores, plddt_proxy, bio_weight, pll_weight)
    return pd.DataFrame(
        {
            "sequence": sequences,
            "bio_score": np.asarray(bio_scores),
            "plddt_proxy": np.asarray(plddt_proxy),
            "combined_score": combined,
        }
    )


def select_top_k(scored_df, k=2000, score_col="combined_score"):
    """Global top-K by score. This is the selection stage used in the
    final design."""
    out = scored_df.sort_values(score_col, ascending=False).head(k).copy()
    out = out.reset_index(drop=True)
    print(
        f"selected {len(out)} of {len(scored_df)} | "
        f"score {out[score_col].min():.4f}-{out[score_col].max():.4f}"
    )
    return out


def unselected_sample(scored_df, n=2000, seed=42, score_col="combined_score"):
    """A matched random sample, for measuring what selection contributes.

    Generation and selection are separate stages, so each trained model
    is evaluated twice: once on the selected set and once on an
    unfiltered sample of the same size.
    """
    return scored_df.sample(n=min(n, len(scored_df)), random_state=seed).reset_index(
        drop=True
    )


def rank_correlation(scored_df, plddt_col="plddt", score_col="combined_score"):
    """Spearman correlation between the composite score and true pLDDT.

    This is the quantity that says how well the score identifies foldable
    sequences before folding them. Spearman rather than Pearson because
    selection only ever ranks.
    """
    from scipy.stats import pearsonr, spearmanr

    sub = scored_df.dropna(subset=[plddt_col, score_col])
    rho = spearmanr(sub[score_col], sub[plddt_col]).correlation
    r = pearsonr(sub[score_col], sub[plddt_col])[0]
    return {"spearman": float(rho), "pearson": float(r), "n": len(sub)}

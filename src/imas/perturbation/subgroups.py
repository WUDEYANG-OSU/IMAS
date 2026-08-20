from __future__ import annotations
import numpy as np


def balanced_merge_microclusters(centroids: np.ndarray, sizes: np.ndarray, n_groups: int = 3,
                                 balance_penalty: float = 6.0, max_iterations: int = 200, seed: int = 42):
    """Greedy balanced merge of response-aware microclusters into final response modes."""
    rng=np.random.default_rng(seed); centroids=np.asarray(centroids,float); sizes=np.asarray(sizes,float)
    seeds=[rng.integers(len(centroids))]
    while len(seeds)<n_groups:
        distance=np.min([[np.linalg.norm(c-centroids[s]) for s in seeds] for c in centroids],axis=1); seeds.append(int(np.argmax(distance)))
    group_centers=centroids[seeds].copy(); assignment=np.zeros(len(centroids),int); target=sizes.sum()/n_groups
    for _ in range(max_iterations):
        previous=assignment.copy(); totals=np.zeros(n_groups)
        for i in np.argsort(-sizes):
            costs=[np.linalg.norm(centroids[i]-group_centers[g])+balance_penalty*max(0,(totals[g]+sizes[i]-target)/max(target,1)) for g in range(n_groups)]
            assignment[i]=int(np.argmin(costs)); totals[assignment[i]]+=sizes[i]
        for g in range(n_groups):
            mask=assignment==g
            if mask.any(): group_centers[g]=np.average(centroids[mask],axis=0,weights=sizes[mask])
        if np.array_equal(previous,assignment):break
    return assignment

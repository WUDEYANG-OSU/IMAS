from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors


def project_cell_pseudotime(cell_cluster_ids, cell_embeddings: np.ndarray, cluster_time: dict[str, float], k: int = 15):
    base=np.asarray([cluster_time[str(c)] for c in cell_cluster_ids],np.float32)
    if len(base)<=1:return base
    nn=NearestNeighbors(n_neighbors=min(k+1,len(base)),metric="cosine").fit(cell_embeddings)
    dist,ind=nn.kneighbors(cell_embeddings); out=np.zeros_like(base)
    for i in range(len(base)):
        weights=np.exp(-dist[i,1:]); out[i]=0.6*base[i]+0.4*np.sum(weights*base[ind[i,1:]])/(weights.sum()+1e-8)
    return (out-out.min())/(out.max()-out.min()+1e-12)

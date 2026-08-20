from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from imas.utils.stats import zscore_safe
from imas.utils.naming import normalize_cluster


def aggregate_cluster_cci(interactions: pd.DataFrame, cluster_sizes: pd.DataFrame, score_column: str = "score", aggregation: str = "sum"):
    frame = interactions.copy(); frame["sender_cluster"] = frame["sender_cluster"].map(normalize_cluster); frame["receiver_cluster"] = frame["receiver_cluster"].map(normalize_cluster)
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0)
    agg = getattr(frame.groupby(["sender_cluster", "receiver_cluster"])[score_column], aggregation)().reset_index(name="edge_score")
    clusters = sorted(set(cluster_sizes.iloc[:,0].map(normalize_cluster)) | set(agg.sender_cluster) | set(agg.receiver_cluster))
    node = pd.DataFrame({"cluster_norm": clusters})
    send = agg.groupby("sender_cluster").edge_score.sum(); recv = agg.groupby("receiver_cluster").edge_score.sum()
    node["sender_strength_raw"] = node.cluster_norm.map(send).fillna(0.0); node["receiver_strength_raw"] = node.cluster_norm.map(recv).fillna(0.0)
    node["sender_strength"] = zscore_safe(np.log1p(node.sender_strength_raw)); node["receiver_strength"] = zscore_safe(np.log1p(node.receiver_strength_raw))
    node["net_flow"] = zscore_safe(node.receiver_strength_raw - node.sender_strength_raw)
    node["io_ratio"] = zscore_safe(np.log1p(node.receiver_strength_raw) - np.log1p(node.sender_strength_raw))
    return node, agg


def build_cluster_graph(node_table: pd.DataFrame, edge_table: pd.DataFrame, centroids: np.ndarray,
                        similarity_k: int = 3, use_similarity: bool = True):
    clusters = node_table.cluster_norm.astype(str).tolist(); index = {c:i for i,c in enumerate(clusters)}; n=len(clusters)
    sender = node_table.sender_strength.to_numpy(np.float32); receiver=node_table.receiver_strength.to_numpy(np.float32)
    tf = node_table.get("receiver_tf_response", pd.Series(np.zeros(n))).to_numpy(np.float32)
    rows=[]
    edge_scores = zscore_safe(np.log1p(edge_table.edge_score.to_numpy(np.float32))) if len(edge_table) else np.zeros(0)
    for row, score in zip(edge_table.itertuples(), edge_scores):
        if str(row.sender_cluster) in index and str(row.receiver_cluster) in index:
            i,j=index[str(row.sender_cluster)],index[str(row.receiver_cluster)]
            rows.append((i,j,float(score),0.,sender[i],receiver[j],tf[i],tf[j]))
    if use_similarity and n>1:
        nn=NearestNeighbors(n_neighbors=min(similarity_k+1,n),metric="cosine").fit(centroids)
        dist, ind=nn.kneighbors(centroids)
        for i in range(n):
            for j,d in zip(ind[i,1:],dist[i,1:]): rows.append((i,int(j),0.,float(1-d),sender[i],receiver[int(j)],tf[i],tf[int(j)]))
    groups=defaultdict(list)
    for row in rows: groups[row[0]].append(row)
    k=max([len(v) for v in groups.values()] or [1]); nbr=np.zeros((n,k),np.int64); mask=np.zeros((n,k),bool); feat=np.zeros((n,k,6),np.float32)
    for i in range(n):
        current=groups.get(i,[]) or [(i,i,0.,0.,sender[i],receiver[i],tf[i],tf[i])]
        for position,row in enumerate(current):
            nbr[i,position]=row[1]; mask[i,position]=True; feat[i,position]=row[2:]
    return nbr, mask, feat

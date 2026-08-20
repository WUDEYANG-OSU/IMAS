from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import PCA, TruncatedSVD
from imas.utils.stats import rank01, robust01


@dataclass
class NDSWeights:
    expression: float = 0.34
    position: float = 0.33
    field: float = 0.33
    field_in: float = 0.45
    field_out: float = 0.45
    field_expression: float = 0.10


def build_nds_network(deg: pd.DataFrame, candidates: pd.DataFrame, weights: NDSWeights = NDSWeights()):
    nodes=sorted(set(candidates.ligand.astype(str))|set(candidates.receptor.astype(str))|set(candidates.tf.astype(str)))
    index={node:i for i,node in enumerate(nodes)}; n=len(nodes); adjacency=np.zeros((n,n),float)
    score_col=next(c for c in ["candidate_score","score","edge_score"] if c in candidates.columns)
    for row in candidates.itertuples():
        d=row._asdict(); chain=[str(d["ligand"]),str(d["receptor"]),str(d["tf"])]
        for source,target in zip(chain[:-1],chain[1:]): adjacency[index[source],index[target]]+=float(d[score_col])
    expr_lookup=deg.set_index("gene").log2FC_tumor_vs_normal.to_dict(); expr=np.asarray([abs(float(expr_lookup.get(node,0))) for node in nodes])
    incoming=adjacency.sum(0); outgoing=adjacency.sum(1); field=weights.field_in*robust01(incoming)+weights.field_out*robust01(outgoing)+weights.field_expression*robust01(expr)
    joint=np.concatenate([adjacency,adjacency.T],axis=1); position=PCA(n_components=min(2,n)).fit_transform(joint) if n<=4000 else TruncatedSVD(n_components=2).fit_transform(sparse.csr_matrix(joint))
    pos=np.linalg.norm(position-position.mean(0),axis=1); nds=weights.expression*robust01(expr)+weights.position*robust01(pos)+weights.field*robust01(field)
    node_table=pd.DataFrame({"node":nodes,"expression_component":robust01(expr),"position_component":robust01(pos),"field_component":robust01(field),"NDS":nds})
    return node_table, adjacency


def build_cross_sample_consensus(sample_tables: dict[str,pd.DataFrame], mean_weight=0.55,max_weight=0.25,coverage_weight=0.20):
    rows=[]
    for sample,table in sample_tables.items():
        rows.extend({"sample_id":sample,"node":r.node,"NDS":r.NDS} for r in table.itertuples())
    long=pd.DataFrame(rows); grouped=long.groupby("node").NDS
    out=pd.DataFrame({"mean_nds":grouped.mean(),"max_nds":grouped.max(),"n_samples":grouped.count()}).reset_index(); out["coverage"]=out.n_samples/max(1,len(sample_tables))
    out["consensus_nds"]=mean_weight*out.mean_nds+max_weight*out.max_nds+coverage_weight*out.coverage
    return out.sort_values("consensus_nds",ascending=False)

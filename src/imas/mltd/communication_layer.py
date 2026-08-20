from __future__ import annotations
from collections import defaultdict, Counter
import pandas as pd
import networkx as nx


def _column(frame, tokens):
    for col in frame.columns:
        low=str(col).lower()
        if any(token in low for token in tokens):return col
    return None


def build_communication_annotations(target_gene: str, graph: nx.DiGraph, interactions: pd.DataFrame,
                                    receptor_tf_signature: pd.DataFrame, top_n: int = 16):
    ligand=_column(interactions,["ligand"]); receptor=_column(interactions,["receptor"]); score=_column(interactions,["score","confidence","expr_only","weight"])
    if ligand is None or receptor is None:return []
    frame=interactions.copy(); frame["_score"]=pd.to_numeric(frame[score],errors="coerce").fillna(0.0) if score else 1.0
    frame=frame[(frame[ligand].astype(str)==target_gene)|(frame[receptor].astype(str)==target_gene)].nlargest(top_n,"_score")
    sig_rec=_column(receptor_tf_signature,["receptor"]); sig_tf=_column(receptor_tf_signature,["tf"]); sig_score=_column(receptor_tf_signature,["corr","score","weight"])
    mapping=defaultdict(list)
    if sig_rec and sig_tf:
        for row in receptor_tf_signature.itertuples(index=False):
            d=row._asdict(); mapping[str(d[sig_rec])].append((str(d[sig_tf]),float(d.get(sig_score,1.0))))
    target_edges=[edge for edge in graph.edges if target_gene in edge] or list(graph.edges); used=Counter(); out=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        d=row._asdict(); rec=str(d[receptor]); tfs=[x[0] for x in sorted(mapping.get(rec,[]),key=lambda x:x[1],reverse=True)]
        candidates=[(tf,target_gene) for tf in tfs if graph.has_edge(tf,target_gene)] or target_edges
        anchor=sorted(candidates,key=lambda edge:(used[edge],edge))[i%len(candidates)]; used[anchor]+=1
        out.append({"anchor_u":anchor[0],"anchor_v":anchor[1],"ligand":str(d[ligand]),"receptor":rec,"tf":tfs[0] if tfs else None,"score":float(d["_score"])})
    return out

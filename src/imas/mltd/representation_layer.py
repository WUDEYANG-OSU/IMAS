from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx


def add_representation_layer(graph: nx.DiGraph, target_gene: str, target_high_token_scores: dict[str,float], top_genes: int = 12):
    out=nx.DiGraph(graph)
    ranked=sorted(target_high_token_scores.items(),key=lambda item:(item[0] not in graph,-item[1]))[:top_genes]
    for gene,score in ranked:
        if gene==target_gene:continue
        out.add_node(gene,kind="gene")
        weight=float(max(0.8,min(5.0,np.log1p(max(0.0,score)))))
        if out.has_edge(target_gene,gene): out[target_gene][gene]["stage2_boost"]=weight
        else: out.add_edge(target_gene,gene,weight=weight,edge_type="representation_similarity")
    return out

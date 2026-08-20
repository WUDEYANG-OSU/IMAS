from __future__ import annotations
from dataclasses import dataclass, field
import networkx as nx
import pandas as pd
from .prior_layer import build_prior_layer
from .representation_layer import add_representation_layer
from .communication_layer import build_communication_annotations
from .temporal_layer import add_temporal_annotations


@dataclass
class MLTDResult:
    target_gene: str
    graph: nx.DiGraph
    communication_annotations: list[dict] = field(default_factory=list)
    temporal_annotations: list[dict] = field(default_factory=list)


class MLTDBuilder:
    def __init__(self, target_gene: str, prior_table: pd.DataFrame):
        self.target_gene=target_gene; self.graph=build_prior_layer(target_gene,prior_table); self.comm=[]; self.temporal=[]
    def add_representation(self, token_scores: dict[str,float], top_genes: int = 12):
        self.graph=add_representation_layer(self.graph,self.target_gene,token_scores,top_genes); return self
    def add_communication(self, interactions: pd.DataFrame, receptor_tf_signature: pd.DataFrame, top_n: int = 16):
        self.comm=build_communication_annotations(self.target_gene,self.graph,interactions,receptor_tf_signature,top_n); return self
    def add_temporal(self, edge_timing: pd.DataFrame, cluster_time: pd.DataFrame | None = None, top_n: int = 16):
        self.temporal=add_temporal_annotations(self.comm,edge_timing,cluster_time,top_n); return self
    def build(self): return MLTDResult(self.target_gene,self.graph,self.comm,self.temporal)

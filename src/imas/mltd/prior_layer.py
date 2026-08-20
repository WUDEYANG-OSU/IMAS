from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from imas.utils.io import read_vocab_txt


def load_tf_gene_prior(priors_dir: str | Path, tf_vocab_path: str | Path, gene_vocab_path: str | Path,
                       prefer_full: bool = True, decay_after_cap: float = 0.35, cap_index: int = 256) -> pd.DataFrame:
    priors_dir=Path(priors_dir); tf_vocab=read_vocab_txt(tf_vocab_path); gene_vocab=read_vocab_txt(gene_vocab_path)
    path=(priors_dir/"tf_gene_f.full.npz") if prefer_full and (priors_dir/"tf_gene_f.full.npz").exists() else (priors_dir/"tf_gene_f.cap256.npz")
    z=np.load(path); rows=[]
    for source,dest in zip(z["src"].astype(int),z["dst"].astype(int)):
        if 0<=source<len(tf_vocab) and 0<=dest<len(gene_vocab):
            rows.append({"tf":tf_vocab[source],"gene":gene_vocab[dest],"weight":1.0 if dest<cap_index else float(decay_after_cap),"is_after_cap":int(dest>=cap_index)})
    return pd.DataFrame(rows)


def build_prior_layer(target_gene: str, prior: pd.DataFrame, top_tfs: int = 8, top_genes: int = 12) -> nx.DiGraph:
    target_rows=prior[prior.gene.astype(str)==str(target_gene)]
    keep_tfs=set(target_rows.groupby("tf").weight.sum().nlargest(top_tfs).index.astype(str))
    neighborhood=prior[prior.tf.astype(str).isin(keep_tfs)]
    genes=[target_gene]+[g for g in neighborhood.groupby("gene").weight.sum().sort_values(ascending=False).index.astype(str) if g!=target_gene][:max(0,top_genes-1)]
    graph=nx.DiGraph(); graph.add_node(target_gene,kind="target_gene")
    for row in neighborhood[neighborhood.gene.astype(str).isin(genes)].itertuples():
        graph.add_node(str(row.tf),kind="tf"); graph.add_node(str(row.gene),kind="target_gene" if str(row.gene)==target_gene else "gene")
        graph.add_edge(str(row.tf),str(row.gene),weight=float(row.weight),edge_type="prior_tf_gene")
    return graph

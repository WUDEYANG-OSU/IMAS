from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import networkx as nx


def _scaled(mapping):
    if not mapping:return {}
    keys=list(mapping); values=np.asarray([math.log1p(max(0.0,mapping[k])) for k in keys])
    lo,hi=values.min(),values.max(); scaled=np.full_like(values,0.5) if hi-lo<1e-12 else (values-lo)/(hi-lo)
    return dict(zip(keys,scaled.tolist()))


def _comm_support(annotations):
    values=defaultdict(float)
    for item in annotations: values[(str(item["anchor_u"]),str(item["anchor_v"]))]+=max(0.0,float(item.get("score",0.0)))
    return _scaled(values)


def _temporal_support(annotations):
    values={}
    for item in annotations:
        edge=(str(item["anchor_u"]),str(item["anchor_v"])); dt=item.get("delta_t_receiver_minus_sender",item.get("delta_t",np.nan)); tf=item.get("receiver_tf_response",np.nan)
        dt_score=math.exp(-abs(float(dt)-0.15)/0.45) if np.isfinite(dt) else 0.; tf_score=max(0.,float(tf)) if np.isfinite(tf) else 0.
        values[edge]=max(values.get(edge,0.),0.65*dt_score+0.35*tf_score)
    return _scaled(values)


def build_field_edge_table(graph: nx.DiGraph, communication_annotations: list[dict], temporal_annotations: list[dict],
                           node_expression: dict[str,float] | None = None, weights=(1.0,0.8,1.15,1.0)):
    base={(str(u),str(v)):float(a.get("weight",0.5)) for u,v,a in graph.edges(data=True)}
    rep={(str(u),str(v)):float(a.get("stage2_boost",a.get("weight",0.0) if a.get("edge_type")=="representation_similarity" else 0.0)) for u,v,a in graph.edges(data=True)}
    base,rep,comm,temp=_scaled(base),_scaled(rep),_comm_support(communication_annotations),_temporal_support(temporal_annotations)
    rows=[]
    for u,v,a in graph.edges(data=True):
        edge=(str(u),str(v)); p,r,c,t=base.get(edge,0.),rep.get(edge,0.),comm.get(edge,0.),temp.get(edge,0.)
        effective=math.exp(sum(w*math.log1p(max(0.,x)) for w,x in zip(weights,(p,r,c,t))))-1
        rows.append({"src":edge[0],"dst":edge[1],"edge_type":a.get("edge_type",""),"base_prior":p,"repr_support":r,"comm_support":c,"temporal_support":t,"w_eff":effective})
    frame=pd.DataFrame(rows)
    if len(frame): frame["w_eff_norm"]=list(_scaled({(r.src,r.dst):r.w_eff for r in frame.itertuples()}).values())
    return frame


def build_node_state(node_expression: dict[str,float], alpha: float = 1.6, beta: float = 1.25, epsilon: float = 0.05):
    names=list(node_expression); raw=np.asarray([node_expression[name] for name in names],float); lo,hi=raw.min(),raw.max(); norm=np.full_like(raw,0.5) if hi-lo<1e-12 else (raw-lo)/(hi-lo)
    return pd.DataFrame({"node":names,"expr_raw":raw,"expr_norm":norm,"field_strength":norm**alpha,"susceptibility":np.clip(1-norm,0,1)**beta+epsilon,"state_init":norm})

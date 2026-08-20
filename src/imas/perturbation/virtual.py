from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PerturbationConfig:
    mode: str = "ko"
    strength: float = 0.75
    alpha: float = 1.6
    beta: float = 1.25
    epsilon: float = 0.05
    kappa: float = 0.55
    update_rate: float = 0.35
    max_iterations: int = 20
    tolerance: float = 1e-4
    use_two_hop: bool = True


def _perturb_state(state: pd.DataFrame, target: str, config: PerturbationConfig):
    out=state.copy(); mask=out.node.astype(str)==str(target)
    if mask.any():
        value=float(out.loc[mask,"state_init"].iloc[0])
        value=value*(1-config.strength) if config.mode=="ko" else min(1.,value+config.strength*(1-value))
        out.loc[mask,"state_init"]=value
    return out


def propagate(edge_table: pd.DataFrame, state: pd.DataFrame, config: PerturbationConfig):
    out=state.copy(); nodes=out.node.astype(str).tolist(); index={n:i for i,n in enumerate(nodes)}; n=len(nodes)
    initial=out.state_init.to_numpy(float); current=initial.copy(); matrix=np.zeros((n,n),float)
    for row in edge_table.itertuples():
        if str(row.src) in index and str(row.dst) in index: matrix[index[str(row.src)],index[str(row.dst)]]=max(matrix[index[str(row.src)],index[str(row.dst)]],float(row.w_eff_norm))
    for _ in range(config.max_iterations):
        field=np.clip(current,0,1)**config.alpha; susceptibility=np.clip(1-current,0,1)**config.beta+config.epsilon
        incoming=(matrix.T@field)*susceptibility
        if config.use_two_hop: incoming+=(matrix.T@(matrix.T@field))*0.35*susceptibility
        target=np.clip(initial+config.kappa*incoming,0,1); new=np.clip((1-config.update_rate)*current+config.update_rate*target,0,1)
        if np.max(np.abs(new-current))<config.tolerance: current=new; break
        current=new
    out["state_final"]=current; out["field_strength_final"]=current**config.alpha; out["susceptibility_final"]=np.clip(1-current,0,1)**config.beta+config.epsilon; out["delta_state"]=current-initial
    return out


def edge_flux(edge_table: pd.DataFrame, state: pd.DataFrame):
    lookup=state.set_index(state.node.astype(str))[["field_strength_final","susceptibility_final"]].to_dict("index"); rows=[]
    for row in edge_table.to_dict("records"):
        source,target=str(row["src"]),str(row["dst"]); flux=float(row["w_eff_norm"])*lookup.get(source,{}).get("field_strength_final",0.)*lookup.get(target,{}).get("susceptibility_final",0.)
        rows.append({**row,"edge_flux":flux})
    return pd.DataFrame(rows)


def compute_delta_flux(edge_table, baseline_state, perturbed_state):
    base=edge_flux(edge_table,baseline_state).rename(columns={"edge_flux":"flux_baseline"}); pert=edge_flux(edge_table,perturbed_state)[["src","dst","edge_flux"]].rename(columns={"edge_flux":"flux_perturbed"})
    out=base.merge(pert,on=["src","dst"]); out["delta_flux"]=out.flux_perturbed-out.flux_baseline; out["abs_delta_flux"]=out.delta_flux.abs(); return out


class VirtualPerturbation:
    def __init__(self, edge_table: pd.DataFrame, node_state: pd.DataFrame, target: str, config: PerturbationConfig = PerturbationConfig()):
        self.edge_table,self.node_state,self.target,self.config=edge_table,node_state,target,config
    def run(self):
        baseline=propagate(self.edge_table,self.node_state,self.config)
        perturbed=propagate(self.edge_table,_perturb_state(self.node_state,self.target,self.config),self.config)
        return {"baseline_state":baseline,"perturbed_state":perturbed,"delta_flux":compute_delta_flux(self.edge_table,baseline,perturbed)}

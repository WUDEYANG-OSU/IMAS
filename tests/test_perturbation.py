import pandas as pd
from imas.perturbation import VirtualPerturbation, PerturbationConfig

def test_virtual_perturbation_runs():
    edges=pd.DataFrame([{"src":"A","dst":"B","w_eff_norm":1.0}])
    nodes=pd.DataFrame([{"node":"A","state_init":0.8},{"node":"B","state_init":0.2}])
    result=VirtualPerturbation(edges,nodes,"A",PerturbationConfig(max_iterations=2)).run()
    assert "delta_flux" in result

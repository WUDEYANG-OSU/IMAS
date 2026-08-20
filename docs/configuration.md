# Configuration and portable paths

IMAS resolves paths in this order:

1. A specific environment variable, such as `IMAS_RUNS_ROOT`.
2. `configs/project.yaml` (or the file named by `IMAS_CONFIG`).
3. A path relative to `IMAS_PROJECT_ROOT`.
4. The repository root for lightweight tests.

## Environment variables

| Variable | Purpose |
|---|---|
| `IMAS_CONFIG` | Alternate YAML configuration file |
| `IMAS_PROJECT_ROOT` | Workspace root |
| `IMAS_WORLD_ROOT` | Dataset corpus root |
| `IMAS_RUNS_ROOT` | Generated run outputs |
| `IMAS_GLOBAL_ROOT` | Global projection/network cache |
| `IMAS_MAPS_ROOT` | Global vocabulary maps |
| `IMAS_PRIORS_ROOT` | Prior cache |
| `IMAS_CHECKPOINT_ROOT` | Foundation checkpoints |
| `IMAS_RESOURCES_ROOT` | Pathway and auxiliary resources |
| `IMAS_VERIFY_PC_ROOT` | PC9 perturbation benchmark data |
| `IMAS_SIG_LLM_ROOT` | Optional external legacy output root |

Dataset-specific checkpoint variables are also accepted. For example,
`IMAS_CKPT_CRC_TISSUE=/path/to/checkpoint.pt`.

## Python API

```python
from imas.config import ProjectPaths, get_dataset_checkpoint

paths = ProjectPaths.discover()
print(paths.world_root)
print(get_dataset_checkpoint("CRC-Tissue"))
```

The repository includes `configs/project.default.yaml` and dataset templates
under `configs/datasets/`. Local `configs/project.yaml` is ignored by Git.

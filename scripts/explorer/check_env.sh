#!/bin/bash
# Quick check that the Explorer environment can run Gate B. Run on a compute node:
#   srun -p short -t 00:15:00 --mem=8G -c 1 bash scripts/explorer/check_env.sh
set -euo pipefail
module load explorer anaconda3/2024.06 >/dev/null 2>&1
source "${GNNFLU_ENV:-$HOME/envs/gnn-flu}/bin/activate"
cd "${AREAFLU_ROOT:-/scratch/$USER/areaFluForecast}"
export PYTHONPATH="$PWD/src" AREAFLU_RESEARCH_REPO="${AREAFLU_RESEARCH_REPO:-/scratch/$USER/gnn-flu}"
export PYTHONWARNINGS=ignore
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import torch, torch_geometric, pandas, numpy, scipy
print("versions", torch.__version__, torch_geometric.__version__, pandas.__version__, numpy.__version__)
from research_bridge import boston_from_research, city_from_research
print(boston_from_research().summary().splitlines()[0])
print(city_from_research("columbus", with_static=False).summary().splitlines()[0])
PY
python scripts/parity/run_parity.py list | tail -1

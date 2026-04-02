#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --job-name=pdearena_rebuttal_costs
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PDEARENA_REPO:-}" ]]; then
  PDEARENA_REPO="${PDEARENA_REPO}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  PDEARENA_REPO="${SLURM_SUBMIT_DIR}"
else
  PDEARENA_REPO="${DEFAULT_REPO_ROOT}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-${PDEARENA_REPO}/outputs/train_cost}"
COST_SCRIPT="${PDEARENA_REPO}/scripts/eff_collect_train_cost_cluster.sh"
TABLE_PATH="${OUTPUT_BASE}/rebuttal_training_cost_table.md"

if [[ ! -x "${COST_SCRIPT}" ]]; then
  echo "Cost collection script not found or not executable: ${COST_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}"

run_case() {
  local run_name="$1"
  local config_path="$2"

  echo
  echo "=== Running ${run_name} ==="
  PDEARENA_REPO="${PDEARENA_REPO}" \
  RUN_NAME="${run_name}" \
  OUTPUT_ROOT="${OUTPUT_BASE}/${run_name}" \
  bash "${COST_SCRIPT}" "${config_path}"
}

run_case "convection_unet_cost"   "${PDEARENA_REPO}/configs/eff_convection_parametric.yaml"
run_case "convection_fno_cost"    "${PDEARENA_REPO}/configs/eff_convection_fno.yaml"
run_case "helmholtz_unet_cost"    "${PDEARENA_REPO}/configs/eff_helmholtz2D_parametric.yaml"
run_case "helmholtz_fno_cost"     "${PDEARENA_REPO}/configs/eff_helmholtz2D_fno.yaml"

python - <<PY
import json
from pathlib import Path

output_base = Path(${OUTPUT_BASE@Q})
table_path = Path(${TABLE_PATH@Q})

cases = {
    "U-Net": [
        ("Convection", output_base / "convection_unet_cost" / "training_cost_summary.json"),
        ("Helmholtz-2D", output_base / "helmholtz_unet_cost" / "training_cost_summary.json"),
    ],
    "FNO": [
        ("Convection", output_base / "convection_fno_cost" / "training_cost_summary.json"),
        ("Helmholtz-2D", output_base / "helmholtz_fno_cost" / "training_cost_summary.json"),
    ],
}

def fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.1f} min"

def fmt_gb(mb: float) -> str:
    return f"{mb / 1024.0:.1f} GB"

def fmt_gpu_hours(hours: float) -> str:
    return f"{hours:.2f}"

lines = ["# PDEArena Training Cost Summary", ""]

for model_name, model_cases in cases.items():
    lines.append(f"## {model_name}")
    lines.append("")
    lines.append("| Equation | Time | Peak memory | GPU-hours |")
    lines.append("| --- | ---: | ---: | ---: |")
    for equation, summary_path in model_cases:
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        data = json.loads(summary_path.read_text())
        lines.append(
            f"| {equation} | `{fmt_minutes(float(data['elapsed_seconds']))}` | "
            f"`{fmt_gb(float(data['peak_gpu_memory_mb']))}` | "
            f"`{fmt_gpu_hours(float(data['gpu_hours']))}` |"
        )
    lines.append("")

table_path.write_text("\n".join(lines).rstrip() + "\n")
print(f"Wrote {table_path}")
PY

echo
echo "Finished all four runs."
echo "Markdown table: ${TABLE_PATH}"

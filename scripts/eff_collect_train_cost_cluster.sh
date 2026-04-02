#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=pdearena_cost
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

usage() {
  cat <<'EOF'
Submit or run a PDEArena U-Net/FNO training job while collecting wall time,
peak GPU memory, peak CPU RSS, and GPU-hours.

Typical usage:
  CONFIG_PATH=configs/eff_convection_fno.yaml \
  sbatch scripts/eff_collect_train_cost_cluster.sh

Optional environment variables:
  PDEARENA_REPO     Repo root. Default: ~/Documents/Projects/git/pdearena
  CONFIG_PATH       Required YAML config path
  MODEL_TYPE        Optional: fno | unet. Inferred from config name if omitted
  RUN_NAME          Optional run name. Default: <config-stem>_<timestamp>
  OUTPUT_ROOT       Optional output dir. Default: <repo>/outputs/train_cost/<RUN_NAME>
  PYTHON_EXECUTABLE Optional python path. Default: <repo>/.venv/bin/python or python
  POLL_SECONDS      GPU polling interval. Default: 2
  TRAIN_SUBCOMMAND  Lightning subcommand. Default: fit
  EXTRA_ARGS        Extra CLI overrides, e.g.:
                    "trainer.max_epochs=50 data.n_train=10"

Outputs:
  <OUTPUT_ROOT>/time_verbose.txt
  <OUTPUT_ROOT>/gpu_memory_samples.csv
  <OUTPUT_ROOT>/training_cost_summary.json
  <OUTPUT_ROOT>/training_cost_summary.csv
  <OUTPUT_ROOT>/original_config.yaml
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PDEARENA_REPO:-}" ]]; then
  PDEARENA_REPO="${PDEARENA_REPO}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  PDEARENA_REPO="${SLURM_SUBMIT_DIR}"
else
  PDEARENA_REPO="${DEFAULT_REPO_ROOT}"
fi
CONFIG_PATH="${CONFIG_PATH:-${1:-}}"
MODEL_TYPE="${MODEL_TYPE:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-}"
POLL_SECONDS="${POLL_SECONDS:-2}"
TRAIN_SUBCOMMAND="${TRAIN_SUBCOMMAND:-fit}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "CONFIG_PATH is required." >&2
  usage
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH/#\~/$HOME}"
PDEARENA_REPO="${PDEARENA_REPO/#\~/$HOME}"

if [[ ! -d "${PDEARENA_REPO}" ]]; then
  echo "PDEARENA_REPO does not exist: ${PDEARENA_REPO}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  if [[ -f "${PDEARENA_REPO}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${PDEARENA_REPO}/${CONFIG_PATH}"
  else
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
fi

if [[ -z "${PYTHON_EXECUTABLE}" ]]; then
  if [[ -x "${PDEARENA_REPO}/.venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="${PDEARENA_REPO}/.venv/bin/python"
  else
    PYTHON_EXECUTABLE="python"
  fi
fi

CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"
CONFIG_STEM="${CONFIG_BASENAME%.yaml}"

if [[ -z "${MODEL_TYPE}" ]]; then
  if [[ "${CONFIG_BASENAME}" == *"_fno.yaml" ]]; then
    MODEL_TYPE="fno"
  else
    MODEL_TYPE="unet"
  fi
fi

case "${MODEL_TYPE}" in
  fno)
    TRAIN_SCRIPT="scripts/eff_parametric_fno_train.py"
    ;;
  unet|u-net|parametric)
    TRAIN_SCRIPT="scripts/eff_parametric_train.py"
    ;;
  *)
    echo "Unsupported MODEL_TYPE: ${MODEL_TYPE}" >&2
    exit 1
    ;;
esac

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${CONFIG_STEM}_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${PDEARENA_REPO}/outputs/train_cost/${RUN_NAME}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}"
mkdir -p "${OUTPUT_ROOT}"

TIME_LOG="${OUTPUT_ROOT}/time_verbose.txt"
GPU_LOG="${OUTPUT_ROOT}/gpu_memory_samples.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/training_cost_summary.json"
SUMMARY_CSV="${OUTPUT_ROOT}/training_cost_summary.csv"
COMMAND_LOG="${OUTPUT_ROOT}/command.txt"

cp "${CONFIG_PATH}" "${OUTPUT_ROOT}/original_config.yaml"

collect_descendant_pids() {
  local root_pid="$1"
  local pending=("${root_pid}")
  local descendants=()
  while ((${#pending[@]})); do
    local pid="${pending[0]}"
    pending=("${pending[@]:1}")
    local children
    children="$(pgrep -P "${pid}" || true)"
    if [[ -n "${children}" ]]; then
      while read -r child; do
        [[ -z "${child}" ]] && continue
        descendants+=("${child}")
        pending+=("${child}")
      done <<< "${children}"
    fi
  done
  printf '%s\n' "${descendants[@]:-}"
}

monitor_gpu_memory_tree() {
  local root_pid="$1"
  local log_path="$2"
  local poll_seconds="$3"
  echo "timestamp,used_gpu_memory_mb" > "${log_path}"
  while kill -0 "${root_pid}" 2>/dev/null; do
    local pid_list
    pid_list="$(collect_descendant_pids "${root_pid}" | tr '\n' ' ' | xargs || true)"
    local used="0"
    if [[ -n "${pid_list}" ]]; then
      used="$(
        nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v pids="${pid_list}" '
            BEGIN {
              split(pids, arr, " ")
              for (i in arr) want[arr[i]] = 1
            }
            {
              gsub(/ /, "", $1)
              gsub(/ /, "", $2)
              if ($1 in want) sum += $2
            }
            END { print sum + 0 }'
      )"
    fi
    printf '%s,%s\n' "$(date --iso-8601=seconds)" "${used}" >> "${log_path}"
    sleep "${poll_seconds}"
  done
}

get_num_gpus() {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "${SLURM_GPUS_ON_NODE}" | sed 's/[^0-9].*$//'
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  echo "1"
}

read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"

CMD=(
  "${PYTHON_EXECUTABLE}"
  "${PDEARENA_REPO}/${TRAIN_SCRIPT}"
  "${TRAIN_SUBCOMMAND}"
  "--config" "${CONFIG_PATH}"
  "--trainer.default_root_dir=${OUTPUT_ROOT}"
  "--data.data_dir=${PDEARENA_REPO}/datasets"
  "--model.metrics_out_path=${OUTPUT_ROOT}/metrics.json"
  "--model.plot_out_path=${OUTPUT_ROOT}/gt_pred_abs_error.png"
)
if ((${#EXTRA_ARGS_ARR[@]})); then
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

printf '%q ' "${CMD[@]}" > "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"

START_TS="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s)"

(
  cd "${PDEARENA_REPO}"
  /usr/bin/time -v -o "${TIME_LOG}" "${CMD[@]}"
) &
TRAIN_WRAPPER_PID=$!

monitor_gpu_memory_tree "${TRAIN_WRAPPER_PID}" "${GPU_LOG}" "${POLL_SECONDS}" &
MONITOR_PID=$!

wait "${TRAIN_WRAPPER_PID}"
TRAIN_EXIT=$?

wait "${MONITOR_PID}" || true

END_TS="$(date --iso-8601=seconds)"
END_EPOCH="$(date +%s)"
ELAPSED_S=$((END_EPOCH - START_EPOCH))
NUM_GPUS="$(get_num_gpus)"
GPU_HOURS="$(python - <<PY
elapsed = ${ELAPSED_S}
num_gpus = float(${NUM_GPUS})
print(f"{elapsed / 3600.0 * num_gpus:.6f}")
PY
)"
PEAK_GPU_MB="$(awk -F',' 'NR>1 && ($2+0) > max { max = $2+0 } END { print max + 0 }' "${GPU_LOG}")"
MAX_RSS_KB="$(awk -F': *' '/Maximum resident set size \\(kbytes\\)/ {print $2}' "${TIME_LOG}" | tail -n 1)"
MAX_RSS_KB="${MAX_RSS_KB:-0}"

cat > "${SUMMARY_JSON}" <<EOF
{
  "config_path": "$(realpath "${CONFIG_PATH}")",
  "model_type": "${MODEL_TYPE}",
  "train_script": "${TRAIN_SCRIPT}",
  "output_root": "$(realpath "${OUTPUT_ROOT}")",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "start_time": "${START_TS}",
  "end_time": "${END_TS}",
  "elapsed_seconds": ${ELAPSED_S},
  "elapsed_minutes": $(python - <<PY
print(f"{${ELAPSED_S} / 60.0:.6f}")
PY
),
  "num_gpus": ${NUM_GPUS},
  "gpu_hours": ${GPU_HOURS},
  "peak_gpu_memory_mb": ${PEAK_GPU_MB},
  "max_rss_kb": ${MAX_RSS_KB},
  "time_log": "$(realpath "${TIME_LOG}")",
  "gpu_log": "$(realpath "${GPU_LOG}")",
  "exit_code": ${TRAIN_EXIT}
}
EOF

{
  echo "config_path,model_type,elapsed_seconds,elapsed_minutes,num_gpus,gpu_hours,peak_gpu_memory_mb,max_rss_kb,exit_code"
  echo "\"$(realpath "${CONFIG_PATH}")\",${MODEL_TYPE},${ELAPSED_S},$(python - <<PY
print(f"{${ELAPSED_S} / 60.0:.6f}")
PY
),${NUM_GPUS},${GPU_HOURS},${PEAK_GPU_MB},${MAX_RSS_KB},${TRAIN_EXIT}"
} > "${SUMMARY_CSV}"

echo "Wrote training-cost summary to ${SUMMARY_JSON}"
echo "Peak GPU memory (MB): ${PEAK_GPU_MB}"
echo "Elapsed seconds: ${ELAPSED_S}"
echo "GPU-hours: ${GPU_HOURS}"

exit "${TRAIN_EXIT}"

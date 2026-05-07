#!/usr/bin/env bash
# =============================================================================
# monitor.sh  — sample disk I/O, CPU, RAM, and GPU metrics while a command runs.
# Usage:
#   ./monitor.sh results/hardware_${SLURM_JOB_ID}.csv python offline_phase.py
# =============================================================================

set -euo pipefail

OUTFILE="${1:?Usage: $0 <outfile.csv> <command...>}"
shift
COMMAND=("$@")

INTERVAL=1
DEVICE="nvme0n1"

mkdir -p "$(dirname "$OUTFILE")"

# ── Tool checks
HAS_IOSTAT=false;  command -v iostat     &>/dev/null && HAS_IOSTAT=true
HAS_MPSTAT=false;  command -v mpstat     &>/dev/null && HAS_MPSTAT=true
HAS_SMI=false;     command -v nvidia-smi &>/dev/null && HAS_SMI=true
HAS_DCGMI=false;   command -v dcgmi      &>/dev/null && HAS_DCGMI=true
# free is available on all Linux systems — no check needed

echo "Monitor starting — interval=${INTERVAL}s  device=${DEVICE}"
echo "  iostat=$([ "$HAS_IOSTAT"  = true ] && echo yes || echo no)" \
     " mpstat=$([ "$HAS_MPSTAT"  = true ] && echo yes || echo no)" \
     " nvidia-smi=$([ "$HAS_SMI" = true ] && echo yes || echo no)" \
     " dcgmi=$([ "$HAS_DCGMI"   = true ] && echo yes || echo no)"
echo "  output: $OUTFILE"
echo ""

# ── CSV header
{
  printf "timestamp,"
  printf "cpu_util_pct,"
  printf "ram_used_mb,ram_total_mb,ram_avail_mb,ram_used_pct,"
  printf "disk_r_s,disk_w_s,disk_rkB_s,disk_wkB_s,disk_r_await_ms,disk_w_await_ms,disk_util_pct,"
  printf "gpu_mem_used_mb,gpu_mem_total_mb,gpu_mem_free_mb,gpu_util_pct"
  $HAS_DCGMI && printf ",dcgmi_power_w,dcgmi_gpu_util_pct,dcgmi_nvlink_tx_mbs,dcgmi_nvlink_rx_mbs,dcgmi_pcie_tx_mbs,dcgmi_pcie_rx_mbs"
  printf "\n"
} > "$OUTFILE"

# ── Shared state
SAMPLES_FILE=$(mktemp)
echo 0 > "$SAMPLES_FILE"
T_START=$(date +%s)
MONITOR_PID=""

cleanup() {
  [[ -n "$MONITOR_PID" ]] && { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
  echo ""
  echo "Monitor stopped — $(cat "$SAMPLES_FILE") samples over $(( $(date +%s) - T_START ))s written to $OUTFILE"
  rm -f "$SAMPLES_FILE"
}
trap cleanup EXIT

# ── Sampling loop
_monitor_loop() {
  while true; do
    TS=$(date +"%Y-%m-%dT%H:%M:%S")

    # ── CPU (mpstat: 100 - %idle across all cores)
    CPU_FIELDS=""
    if $HAS_MPSTAT; then
      CPU_FIELDS=$(mpstat 1 1 2>/dev/null \
        | awk '/all/ && /^[0-9]/ { printf "%.1f", 100-$NF }')
    fi
    CPU_FIELDS="${CPU_FIELDS:-}"

    # ── RAM (free -m: used, total, available, used%)
    # $7 = available column (more meaningful than free — excludes buff/cache)
    RAM_FIELDS=$(free -m 2>/dev/null \
      | awk '/^Mem:/ { printf "%.0f,%.0f,%.0f,%.1f", $3,$2,$7,$3*100/$2 }')
    RAM_FIELDS="${RAM_FIELDS:-,,,}"

    # ── Disk I/O (iostat)
    DISK_FIELDS=""
    if $HAS_IOSTAT; then
      DISK_FIELDS=$(iostat -dx 1 1 "$DEVICE" 2>/dev/null \
        | awk -v dev="$DEVICE" '$1 == dev { printf "%s,%s,%s,%s,%s,%s,%s", $2,$8,$3,$9,$6,$12,$NF }')
    fi
    DISK_FIELDS="${DISK_FIELDS:-,,,,,,}"

    # ── GPU (nvidia-smi: averaged across all GPUs)
    GPU_FIELDS=""
    if $HAS_SMI; then
      GPU_FIELDS=$(nvidia-smi \
        --query-gpu=memory.used,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ u+=$1; t+=$2; f+=$3; g+=$4; n++ }
                     END { if(n) printf "%.1f,%.1f,%.1f,%.1f", u/n,t/n,f/n,g/n }')
    fi
    GPU_FIELDS="${GPU_FIELDS:-,,,}"

    # ── DCGMI (optional extended GPU metrics)
    DCGMI_FIELDS=""
    if $HAS_DCGMI; then
      DCGMI_FIELDS=$(dcgmi dmon -e 203,204,155,1001,1002,1009,1010 -c 1 2>/dev/null \
        | awk 'NR>2 { pw+=$4; gu+=$2; ntx+=$5; nrx+=$6; ptx+=$7; prx+=$8; n++ }
               END  { if(n) printf ",%.1f,%.1f,%.1f,%.1f,%.1f,%.1f", pw/n,gu/n,ntx/n,nrx/n,ptx/n,prx/n }')
    fi

    printf "%s,%s,%s,%s,%s%s\n" \
      "$TS" "$CPU_FIELDS" "$RAM_FIELDS" "$DISK_FIELDS" "$GPU_FIELDS" "$DCGMI_FIELDS" \
      >> "$OUTFILE"
    echo $(( $(cat "$SAMPLES_FILE") + 1 )) > "$SAMPLES_FILE"
    sleep "$INTERVAL"
  done
}

_monitor_loop &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"
echo ""

# ── Run command
if [[ ${#COMMAND[@]} -gt 0 ]]; then
  echo "Running : ${COMMAND[*]}"
  echo "Started : $(date +%Y-%m-%dT%H:%M:%S)"
  echo ""
  "${COMMAND[@]}"
  echo ""
  echo "Finished: $(date +%Y-%m-%dT%H:%M:%S)"
else
  echo "No command given — monitoring until Ctrl+C"
  wait
fi
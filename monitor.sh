#!/usr/bin/env bash
# =============================================================================
# monitor.sh
# Runs the RAG pipeline and simultaneously samples disk I/O + GPU metrics,
# writing one CSV row per interval with a wall-clock timestamp.
# The monitor starts just before the pipeline and stops when it finishes.
#
# Usage:
#   chmod +x monitor.sh
#   ./monitor.sh
#   ./monitor.sh -d "nvme0n1,nvme1n1" -i 1 -o results/hardware.csv -D
#
# Options:
#   -d  comma-separated disk devices to monitor (default: nvme0n1)
#   -i  sampling interval in seconds            (default: 1)
#   -o  output CSV path                         (default: results/hardware.csv)
#   -D  enable dcgmi (only if available)
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
INTERVAL=1                          # seconds between samples
DEVICES="nvme0n1"                   # comma-separated disk devices
OUTFILE="results/hardware.csv"
USE_DCGMI=false                     # set true if dcgmi is available

# ── Argument parsing ──────────────────────────────────────────────
while getopts "i:d:o:D" opt; do
  case $opt in
    i) INTERVAL="$OPTARG" ;;
    d) DEVICES="$OPTARG" ;;
    o) OUTFILE="$OPTARG" ;;
    D) USE_DCGMI=true ;;
    *) echo "Usage: $0 [-i interval] [-d devices] [-o outfile] [-D use_dcgmi]"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))   # remove parsed flags, leaving the command to run
COMMAND=("$@")          # e.g. ("python" "-m" "rag_pipeline.example")

mkdir -p "$(dirname "$OUTFILE")"

# ── Tool availability checks ──────────────────────────────────────
HAS_IOSTAT=false;    command -v iostat    &>/dev/null && HAS_IOSTAT=true
HAS_NVIDIASMI=false; command -v nvidia-smi &>/dev/null && HAS_NVIDIASMI=true
HAS_DCGMI=false;     command -v dcgmi    &>/dev/null && $USE_DCGMI && HAS_DCGMI=true

echo "Monitor starting  — interval=${INTERVAL}s  devices=${DEVICES}"
echo "  iostat    : $HAS_IOSTAT"
echo "  nvidia-smi: $HAS_NVIDIASMI"
echo "  dcgmi     : $HAS_DCGMI"
echo "  output    : $OUTFILE"
echo ""

# ── Write CSV header ──────────────────────────────────────────────
{
  printf "timestamp,"
  printf "disk_r_s,disk_w_s,disk_rkB_s,disk_wkB_s,disk_r_await_ms,disk_w_await_ms,disk_util_pct,"
  printf "gpu_mem_used_mb,gpu_mem_total_mb,gpu_mem_free_mb,gpu_util_pct"
  if $HAS_DCGMI; then
    printf ",dcgmi_power_w,dcgmi_gpu_util_pct,dcgmi_nvlink_tx_mbs,dcgmi_nvlink_rx_mbs,dcgmi_pcie_tx_mbs,dcgmi_pcie_rx_mbs"
  fi
  printf "\n"
} > "$OUTFILE"

# ── Cleanup on exit ───────────────────────────────────────────────
SAMPLES=0
T_START=$(date +%s)
MONITOR_PID=""

cleanup() {
  # Stop the background monitor loop
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  T_END=$(date +%s)
  ELAPSED=$((T_END - T_START))
  echo ""
  echo "Monitor stopped — ${SAMPLES} samples over ${ELAPSED}s written to ${OUTFILE}"
}
trap cleanup EXIT

# =================================================================
# Launch the hardware sampling loop in the background
# =================================================================
_monitor_loop() {
  while true; do
  TIMESTAMP=$(date +"%Y-%m-%dT%H:%M:%S")

  # ── Disk (iostat) ──────────────────────────────────────────
  DISK_R_S="";  DISK_W_S=""
  DISK_RKB="";  DISK_WKB=""
  DISK_RAWAIT=""; DISK_WAWAIT=""
  DISK_UTIL=""

  if $HAS_IOSTAT; then
    # Run one snapshot (-c 1) and parse the device line(s)
    # Average across all requested devices if more than one
    declare -a R_S=() W_S=() RKB=() WKB=() RAWAIT=() WAWAIT=() UTIL=()

    IFS=',' read -ra DEV_LIST <<< "$DEVICES"
    IOSTAT_OUT=$(iostat -dx 1 1 "${DEV_LIST[@]}" 2>/dev/null || true)

    while IFS= read -r line; do
      DEV=$(echo "$line" | awk '{print $1}')
      # Check if this line is one of our devices
      for dev in "${DEV_LIST[@]}"; do
        if [[ "$DEV" == "$dev" ]]; then
          R_S+=($(echo "$line"    | awk '{print $2}'))
          W_S+=($(echo "$line"    | awk '{print $8}'))
          RKB+=($(echo "$line"    | awk '{print $3}'))
          WKB+=($(echo "$line"    | awk '{print $9}'))
          RAWAIT+=($(echo "$line" | awk '{print $6}'))
          WAWAIT+=($(echo "$line" | awk '{print $12}'))
          UTIL+=($(echo "$line"   | awk '{print $NF}'))
        fi
      done
    done <<< "$IOSTAT_OUT"

    # Average across devices using awk
    avg_array() {
      local arr=("$@")
      local n=${#arr[@]}
      if [[ $n -eq 0 ]]; then echo ""; return; fi
      local sum=0
      for v in "${arr[@]}"; do sum=$(awk "BEGIN{print $sum + $v}"); done
      awk "BEGIN{printf \"%.2f\", $sum / $n}"
    }

    DISK_R_S=$(avg_array "${R_S[@]+"${R_S[@]}"}")
    DISK_W_S=$(avg_array "${W_S[@]+"${W_S[@]}"}")
    DISK_RKB=$(avg_array "${RKB[@]+"${RKB[@]}"}")
    DISK_WKB=$(avg_array "${WKB[@]+"${WKB[@]}"}")
    DISK_RAWAIT=$(avg_array "${RAWAIT[@]+"${RAWAIT[@]}"}")
    DISK_WAWAIT=$(avg_array "${WAWAIT[@]+"${WAWAIT[@]}"}")
    DISK_UTIL=$(avg_array "${UTIL[@]+"${UTIL[@]}"}")
  fi

  # ── GPU (nvidia-smi) ───────────────────────────────────────
  GPU_MEM_USED=""; GPU_MEM_TOTAL=""; GPU_MEM_FREE=""; GPU_UTIL=""

  if $HAS_NVIDIASMI; then
    SMI_OUT=$(nvidia-smi \
      --query-gpu=memory.used,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)

    if [[ -n "$SMI_OUT" ]]; then
      # Average across GPUs
      GPU_MEM_USED=$(echo "$SMI_OUT"  | awk -F',' '{s+=$1;n++} END{if(n>0)printf "%.1f",s/n}')
      GPU_MEM_TOTAL=$(echo "$SMI_OUT" | awk -F',' '{s+=$2;n++} END{if(n>0)printf "%.1f",s/n}')
      GPU_MEM_FREE=$(echo "$SMI_OUT"  | awk -F',' '{s+=$3;n++} END{if(n>0)printf "%.1f",s/n}')
      GPU_UTIL=$(echo "$SMI_OUT"      | awk -F',' '{s+=$4;n++} END{if(n>0)printf "%.1f",s/n}')
    fi
  fi

  # ── GPU (dcgmi) ────────────────────────────────────────────
  DCGMI_POWER=""; DCGMI_GPU_UTIL=""
  DCGMI_NVLINK_TX=""; DCGMI_NVLINK_RX=""
  DCGMI_PCIE_TX="";   DCGMI_PCIE_RX=""

  if $HAS_DCGMI; then
    DCGMI_OUT=$(dcgmi dmon -e 203,204,155,1001,1002,1009,1010 -c 1 2>/dev/null || true)
    if [[ -n "$DCGMI_OUT" ]]; then
      # Skip header line, average numeric columns across GPUs
      DCGMI_POWER=$(echo    "$DCGMI_OUT" | awk 'NR>1{s+=$4;n++} END{if(n>0)printf "%.1f",s/n}')
      DCGMI_GPU_UTIL=$(echo "$DCGMI_OUT" | awk 'NR>1{s+=$2;n++} END{if(n>0)printf "%.1f",s/n}')
      DCGMI_NVLINK_TX=$(echo "$DCGMI_OUT"| awk 'NR>1{s+=$5;n++} END{if(n>0)printf "%.1f",s/n}')
      DCGMI_NVLINK_RX=$(echo "$DCGMI_OUT"| awk 'NR>1{s+=$6;n++} END{if(n>0)printf "%.1f",s/n}')
      DCGMI_PCIE_TX=$(echo  "$DCGMI_OUT" | awk 'NR>1{s+=$7;n++} END{if(n>0)printf "%.1f",s/n}')
      DCGMI_PCIE_RX=$(echo  "$DCGMI_OUT" | awk 'NR>1{s+=$8;n++} END{if(n>0)printf "%.1f",s/n}')
    fi
  fi

  # ── Write CSV row ──────────────────────────────────────────
  {
    printf "%s," "$TIMESTAMP"
    printf "%s,%s,%s,%s,%s,%s,%s," \
      "$DISK_R_S" "$DISK_W_S" "$DISK_RKB" "$DISK_WKB" \
      "$DISK_RAWAIT" "$DISK_WAWAIT" "$DISK_UTIL"
    printf "%s,%s,%s,%s" \
      "$GPU_MEM_USED" "$GPU_MEM_TOTAL" "$GPU_MEM_FREE" "$GPU_UTIL"
    if $HAS_DCGMI; then
      printf ",%s,%s,%s,%s,%s,%s" \
        "$DCGMI_POWER" "$DCGMI_GPU_UTIL" \
        "$DCGMI_NVLINK_TX" "$DCGMI_NVLINK_RX" \
        "$DCGMI_PCIE_TX"   "$DCGMI_PCIE_RX"
    fi
    printf "\n"
  } >> "$OUTFILE"

    SAMPLES=$((SAMPLES + 1))
    sleep "$INTERVAL"
  done
}

# Start the monitor loop as a background process
_monitor_loop &
MONITOR_PID=$!
echo "Monitor PID : ${MONITOR_PID}"
echo ""

# =================================================================
# Run the command — monitor stops automatically when this exits
# =================================================================
if [[ ${#COMMAND[@]} -gt 0 ]]; then
    echo "Running    : ${COMMAND[*]}"
    echo "Started at : $(date +%Y-%m-%dT%H:%M:%S)"
    echo ""
    "${COMMAND[@]}"
    echo ""
    echo "Finished at: $(date +%Y-%m-%dT%H:%M:%S)"
else
    echo "No command given — monitoring until Ctrl+C"
    wait   # sleep until killed
fi
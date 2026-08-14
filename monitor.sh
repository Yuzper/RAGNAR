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

# ── Tool checks
HAS_IOSTAT=false;  command -v iostat     &>/dev/null && HAS_IOSTAT=true
HAS_MPSTAT=false;  command -v mpstat     &>/dev/null && HAS_MPSTAT=true
HAS_SMI=false;     command -v nvidia-smi &>/dev/null && HAS_SMI=true
HAS_DCGMI=false;   command -v dcgmi      &>/dev/null && HAS_DCGMI=true
# free is available on all Linux systems — no check needed

# ── Block device backing $OUTFILE's filesystem.
# Hard-coding a device name means a node that calls its disk anything else
# silently writes blank disk columns for the whole run, and you only find out
# when you go to plot them. Detect it, and say plainly when detection failed.
mkdir -p "$(dirname "$OUTFILE")"

detect_device() {
  local src base
  src=$(findmnt -no SOURCE --target "$(dirname "$OUTFILE")" 2>/dev/null) || true
  if [[ -n "$src" && "$src" == /dev/* ]]; then
    # Partition (nvme0n1p2) → parent disk (nvme0n1), which is what iostat rows key on.
    base=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1) || true
    [[ -n "$base" ]] && { echo "$base"; return; }
    basename "$src"; return
  fi
  # Network filesystem (NFS/BeeGFS/Lustre) or detection failed — fall back to the
  # first local disk so the columns are at least populated, but the numbers then
  # describe node-local I/O, NOT the traffic to $OUTFILE's filesystem.
  lsblk -dno NAME -e 7,11 2>/dev/null | head -1 || true
}

DEVICE="${MONITOR_DEVICE:-$(detect_device)}"
DEVICE_NOTE=""
if [[ -z "$DEVICE" ]]; then
  DEVICE_NOTE=" (NONE FOUND — disk columns will be blank)"
  HAS_IOSTAT=false
elif [[ -n "${MONITOR_DEVICE:-}" ]]; then
  DEVICE_NOTE=" (from \$MONITOR_DEVICE)"
else
  src=$(findmnt -no SOURCE --target "$(dirname "$OUTFILE")" 2>/dev/null) || true
  [[ "$src" == /dev/* ]] || DEVICE_NOTE=" (GUESS — $(dirname "$OUTFILE") is on '${src:-unknown}', not a local block device)"
fi

# ── GPUs visible to this job. Averaging over every GPU on a shared node reports
# other people's idle cards as our utilisation.
GPU_SELECT="${CUDA_VISIBLE_DEVICES:-}"
GPU_NOTE="all GPUs on node (CUDA_VISIBLE_DEVICES unset)"
[[ -n "$GPU_SELECT" ]] && GPU_NOTE="CUDA_VISIBLE_DEVICES=$GPU_SELECT"

echo "Monitor starting — interval≈${INTERVAL}s  device=${DEVICE:-none}${DEVICE_NOTE}"
echo "  iostat=$([ "$HAS_IOSTAT"  = true ] && echo yes || echo no)" \
     " mpstat=$([ "$HAS_MPSTAT"  = true ] && echo yes || echo no)" \
     " nvidia-smi=$([ "$HAS_SMI" = true ] && echo yes || echo no)" \
     " dcgmi=$([ "$HAS_DCGMI"   = true ] && echo yes || echo no)"
echo "  gpus  : $GPU_NOTE"
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
MP_TMP=$(mktemp)
IO_TMP=$(mktemp)
echo 0 > "$SAMPLES_FILE"
T_START=$(date +%s)
MONITOR_PID=""

cleanup() {
  [[ -n "$MONITOR_PID" ]] && { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
  local n elapsed
  n=$(cat "$SAMPLES_FILE")
  elapsed=$(( $(date +%s) - T_START ))
  echo ""
  # Report the ACHIEVED cadence, not the requested one: mpstat and iostat each
  # block for their own interval, so the real spacing between rows is always
  # larger than $INTERVAL and any rate derived from these rows needs the true
  # figure.
  echo "Monitor stopped — $n samples over ${elapsed}s written to $OUTFILE"
  [[ "$n" -gt 1 ]] && echo "  effective sample interval: $(awk -v e="$elapsed" -v n="$n" 'BEGIN{printf "%.2f", e/(n-1)}')s"
  rm -f "$SAMPLES_FILE" "$MP_TMP" "$IO_TMP"
}
trap cleanup EXIT

# ── Sampling loop
_monitor_loop() {
  while true; do
    TS=$(date +"%Y-%m-%dT%H:%M:%S")

    # ── Sample CPU and disk CONCURRENTLY.
    # Both tools' first report is a since-boot cumulative average — useless here,
    # it describes the node's whole uptime rather than this job. `$INTERVAL 2`
    # makes them emit a second report covering only the last $INTERVAL seconds,
    # and the parsers below take the LAST report, never the first. Running them
    # in parallel keeps one interval of blocking rather than two.
    MP_PID=""; IO_PID=""
    if $HAS_MPSTAT; then
      mpstat "$INTERVAL" 2 > "$MP_TMP" 2>/dev/null & MP_PID=$!
    fi
    if $HAS_IOSTAT; then
      iostat -dx "$INTERVAL" 2 "$DEVICE" > "$IO_TMP" 2>/dev/null & IO_PID=$!
    fi
    [[ -n "$MP_PID" ]] && { wait "$MP_PID" 2>/dev/null || true; }
    [[ -n "$IO_PID" ]] && { wait "$IO_PID" 2>/dev/null || true; }

    # ── CPU (mpstat: 100 - %idle across all cores, last report)
    # The trailing "Average:" line also matches /all/, but does not start with a
    # digit, so the ^[0-9] guard drops it — as it does the since-boot first line.
    CPU_FIELDS=""
    if $HAS_MPSTAT; then
      CPU_FIELDS=$(awk '/all/ && /^[0-9]/ { v = 100 - $NF }
                        END { if (v != "") printf "%.1f", v }' "$MP_TMP" 2>/dev/null || true)
    fi
    CPU_FIELDS="${CPU_FIELDS:-}"

    # ── RAM (free -m: used, total, available, used%)
    # $7 = available column (more meaningful than free — excludes buff/cache)
    RAM_FIELDS=$(free -m 2>/dev/null \
      | awk '/^Mem:/ { printf "%.0f,%.0f,%.0f,%.1f", $3,$2,$7,$3*100/$2 }')
    RAM_FIELDS="${RAM_FIELDS:-,,,}"

    # ── Disk I/O (iostat, last report only)
    # Keeping only the LAST matching row is also the guard against emitting more
    # than seven fields: the old version printed once per matching line, so a
    # second report appended 7 extra columns to the row and silently shifted
    # every GPU column after it.
    DISK_FIELDS=""
    if $HAS_IOSTAT; then
      DISK_FIELDS=$(awk -v dev="$DEVICE" '
          $1 == dev { last = $0 }
          END {
            if (last != "") {
              n = split(last, f, " ")
              printf "%s,%s,%s,%s,%s,%s,%s", f[2], f[8], f[3], f[9], f[6], f[12], f[n]
            }
          }' "$IO_TMP" 2>/dev/null || true)
    fi
    DISK_FIELDS="${DISK_FIELDS:-,,,,,,}"

    # ── GPU (nvidia-smi, averaged over the GPUs THIS JOB was allocated)
    # index and uuid are both queried because CUDA_VISIBLE_DEVICES may hold
    # either form. Empty/unset means no restriction, so take every GPU.
    GPU_FIELDS=""
    if $HAS_SMI; then
      GPU_FIELDS=$(nvidia-smi \
        --query-gpu=index,uuid,memory.used,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v vis="$GPU_SELECT" '
            BEGIN {
              nsel = split(vis, want, /,[ ]*/)
              for (i = 1; i <= nsel; i++) { gsub(/^[ \t]+|[ \t]+$/, "", want[i]); sel[want[i]] = 1 }
            }
            {
              idx = $1; uuid = $2
              gsub(/^[ \t]+|[ \t]+$/, "", idx); gsub(/^[ \t]+|[ \t]+$/, "", uuid)
              if (vis == "" || (idx in sel) || (uuid in sel)) { u+=$3; t+=$4; f+=$5; g+=$6; n++ }
            }
            END { if (n) printf "%.1f,%.1f,%.1f,%.1f", u/n, t/n, f/n, g/n }')
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
    # mpstat/iostat already blocked for ~$INTERVAL above; sleeping again on top
    # of that is what stretched the real cadence to ~3s while the header claimed
    # 1s. Only pace manually when neither tool ran.
    if ! $HAS_MPSTAT && ! $HAS_IOSTAT; then
      sleep "$INTERVAL"
    fi
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
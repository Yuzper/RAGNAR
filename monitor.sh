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

# CUDA_VISIBLE_DEVICES may hold indices OR UUIDs; nvidia-smi accepts either, but
# `dcgmi dmon` only ever prints indices. Resolve to indices once here so the
# dcgmi sampler below can filter at all. Empty means "no restriction" to that
# sampler, so a failed resolve degrades to whole-node averaging -- wrong, but
# loudly announced -- rather than to silently empty columns.
GPU_INDEX_SELECT=""
if $HAS_SMI && [[ -n "$GPU_SELECT" ]]; then
  GPU_INDEX_SELECT=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v vis="$GPU_SELECT" '
        BEGIN {
          nsel = split(vis, want, /,[ ]*/)
          for (i = 1; i <= nsel; i++) { gsub(/^[ \t]+|[ \t]+$/, "", want[i]); sel[want[i]] = 1 }
        }
        {
          idx = $1; uuid = $2
          gsub(/^[ \t]+|[ \t]+$/, "", idx); gsub(/^[ \t]+|[ \t]+$/, "", uuid)
          if ((idx in sel) || (uuid in sel)) out = (out == "" ? idx : out "," idx)
        }
        END { print out }')
  if [[ -z "$GPU_INDEX_SELECT" ]]; then
    echo "  WARNING: could not resolve CUDA_VISIBLE_DEVICES='$GPU_SELECT' to GPU" \
         "indices - dcgmi_* columns will average over ALL GPUs on this node."
  fi
fi

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
  $HAS_DCGMI && printf ",dcgmi_power_w,dcgmi_gpu_util_pct,dcgmi_mem_copy_util_pct,dcgmi_gr_active,dcgmi_sm_active,dcgmi_pcie_tx_bytes,dcgmi_pcie_rx_bytes"
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

    # ── DCGMI (optional extended GPU metrics), averaged over the GPUs THIS JOB
    # was allocated — the same restriction the nvidia-smi block above applies.
    # Unfiltered, dmon reports every GPU on the node and the mean dilutes our one
    # busy card with the node's idle ones: an exclusive single-GPU job on a
    # 4-GPU node reads roughly 4x low on power. The dilution factor is the node's
    # GPU count, so unfiltered columns are not comparable across nodes either.
    #
    # dmon prints the entity as two whitespace-separated fields ("GPU" and the
    # index), so $1="GPU", $2=index, and the -e metrics follow from $3 IN THE
    # ORDER REQUESTED:
    #   $3=203  gpu_util         $4=204  mem_copy_util   $5=155  power_usage
    #   $6=1001 gr_engine_active $7=1002 sm_active
    #   $8=1009 pcie_tx_bytes    $9=1010 pcie_rx_bytes
    DCGMI_FIELDS=""
    if $HAS_DCGMI; then
      DCGMI_FIELDS=$(dcgmi dmon -e 203,204,155,1001,1002,1009,1010 -c 1 2>/dev/null \
        | awk -v vis="$GPU_INDEX_SELECT" '
            BEGIN {
              nsel = split(vis, want, /,[ ]*/)
              for (i = 1; i <= nsel; i++) { gsub(/^[ \t]+|[ \t]+$/, "", want[i]); sel[want[i]] = 1 }
            }
            $1 == "GPU" && (vis == "" || ($2 in sel)) {
              gu += $3; mu += $4; pw += $5; gr += $6; sm += $7; ptx += $8; prx += $9; n++
            }
            END {
              if (n) printf ",%.1f,%.1f,%.1f,%.3f,%.3f,%.1f,%.1f", \
                            pw/n, gu/n, mu/n, gr/n, sm/n, ptx/n, prx/n
            }')
      # Seven empty fields, matching the header, so a dmon that returns no rows
      # shortens no row and shifts no column.
      DCGMI_FIELDS="${DCGMI_FIELDS:-,,,,,,,}"
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
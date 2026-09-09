#!/usr/bin/env bash
# =============================================================================
# monitor.sh  — sample disk I/O, CPU, RAM, and GPU metrics while a command runs.
# Usage:
#   ./monitor.sh results/hardware_${SLURM_JOB_ID}.csv python offline_phase.py
#
# CADENCE
# -------
# Samples every $INTERVAL seconds, and actually achieves it. The previous version
# asked for 1s and delivered ~3.0s, because every tick shelled out to tools that
# BLOCK for a full interval before returning a number:
#
#   mpstat  $INTERVAL 2   → blocks ~1s
#   iostat  $INTERVAL 2   → blocks ~1s   (run in parallel with mpstat, so ~1s total)
#   nvidia-smi one-shot   → ~0.1-0.8s, serial
#   dcgmi dmon -c 1       → ~0.5s,     serial
#
# That is not a pacing bug that a smaller sleep can fix — the blocking IS the
# interval. An 8-second index build got 2 samples and a 2.7-second one got zero,
# which is not enough to characterise anything.
#
# The fix is to stop asking external tools to time-average for us:
#   * CPU  — /proc/stat deltas       (instant read, no subprocess)
#   * RAM  — /proc/meminfo           (instant read, replaces `free`)
#   * disk — /proc/diskstats deltas  (instant read, replaces iostat)
#   * GPU  — nvidia-smi --loop-ms streams in the background; each tick reads the
#            most recent block instead of launching a new process
#   * dcgmi — still a per-tick one-shot, the only subprocess left on the critical
#            path. See the note at its call site.
#
# Rates are computed against the MEASURED elapsed time between ticks, not the
# nominal interval, so a late tick reports a correct rate rather than a rate
# inflated by the assumption that it was on time.
# =============================================================================

set -euo pipefail

OUTFILE="${1:?Usage: $0 <outfile.csv> <command...>}"
shift
COMMAND=("$@")

INTERVAL=1
INTERVAL_MS=$((INTERVAL * 1000))

# ── Tool checks
HAS_SMI=false;     command -v nvidia-smi &>/dev/null && HAS_SMI=true
HAS_DCGMI=false;   command -v dcgmi      &>/dev/null && HAS_DCGMI=true
# CPU, RAM and disk now come from /proc, which is always present on Linux, so
# mpstat/iostat/free are no longer needed or checked for.
HAS_PROCSTAT=false;  [[ -r /proc/stat      ]] && HAS_PROCSTAT=true
HAS_MEMINFO=false;   [[ -r /proc/meminfo   ]] && HAS_MEMINFO=true
HAS_DISKSTATS=false; [[ -r /proc/diskstats ]] && HAS_DISKSTATS=true

# ── Block device backing $OUTFILE's filesystem.
# Hard-coding a device name means a node that calls its disk anything else
# silently writes blank disk columns for the whole run, and you only find out
# when you go to plot them. Detect it, and say plainly when detection failed.
mkdir -p "$(dirname "$OUTFILE")"

detect_device() {
  local src base
  src=$(findmnt -no SOURCE --target "$(dirname "$OUTFILE")" 2>/dev/null) || true
  if [[ -n "$src" && "$src" == /dev/* ]]; then
    # Partition (nvme0n1p2) → parent disk (nvme0n1), which is what /proc/diskstats
    # rows key on for whole-device counters.
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
  HAS_DISKSTATS=false
elif [[ -n "${MONITOR_DEVICE:-}" ]]; then
  DEVICE_NOTE=" (from \$MONITOR_DEVICE)"
else
  src=$(findmnt -no SOURCE --target "$(dirname "$OUTFILE")" 2>/dev/null) || true
  [[ "$src" == /dev/* ]] || DEVICE_NOTE=" (GUESS — $(dirname "$OUTFILE") is on '${src:-unknown}', not a local block device)"
fi
# A device we cannot find in /proc/diskstats yields blank columns forever; say so
# now rather than after the run.
if $HAS_DISKSTATS && ! awk -v d="$DEVICE" '$3 == d { found = 1 } END { exit !found }' /proc/diskstats; then
  DEVICE_NOTE="$DEVICE_NOTE (NOT PRESENT in /proc/diskstats — disk columns will be blank)"
  HAS_DISKSTATS=false
fi

# ── GPUs visible to this job. Averaging over every GPU on a shared node reports
# other people's idle cards as our utilisation.
# MONITOR_GPUS overrides, mirroring MONITOR_DEVICE. It takes physical nvidia-smi
# indices or GPU UUIDs, and exists for the ambiguous case diagnosed below.
GPU_SELECT="${MONITOR_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_NOTE=""   # resolved once NGPU is known

# ── Shared state
SAMPLES_FILE=$(mktemp)
SMI_STREAM=$(mktemp)
echo 0 > "$SAMPLES_FILE"
T_START=$(date +%s)
MONITOR_PID=""
SMI_PID=""

# ── Background nvidia-smi stream.
# --loop-ms makes one process emit a block of NGPU lines every interval, so a tick
# costs a `tail` instead of a process spawn. stdbuf forces line buffering: writing
# to a file (not a tty) nvidia-smi would otherwise sit on a 4KB block buffer and
# the stream would arrive in bursts, minutes late.
NGPU=0
if $HAS_SMI; then
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || echo 0)
fi

# ── Decide whether the GPU columns can be trusted to describe THIS job.
# The filter below matches a row's nvidia-smi index OR its UUID against the
# entries in $GPU_SELECT. Matching on UUID is unambiguous. Matching on INDEX is
# only safe when the two numbering schemes agree, and they do not always:
# CUDA_VISIBLE_DEVICES is JOB-RELATIVE (a single-GPU job gets "0" whichever
# physical card SLURM assigned), while nvidia-smi reports PHYSICAL indices.
# Under SLURM cgroup device isolation nvidia-smi only sees the job's own cards
# and renumbers them from 0, so the two agree and everything is fine — that is
# the case we can detect, by nvidia-smi reporting exactly as many GPUs as the
# job was given. When it reports MORE, the index is ambiguous and we may be
# averaging a neighbour's card while reporting it as ours.
NSEL=0
[[ -n "$GPU_SELECT" ]] && NSEL=$(awk -v v="$GPU_SELECT" 'BEGIN { print split(v, a, /,[ ]*/) }')
GPU_WARN=""
if [[ -z "$GPU_SELECT" ]]; then
  GPU_NOTE="ALL ${NGPU} GPU(s) on node (CUDA_VISIBLE_DEVICES and MONITOR_GPUS both unset)"
  [[ "$NGPU" -gt 1 ]] && GPU_WARN="every GPU column is a node-wide average over ${NGPU} cards, including any this job does not own"
elif [[ "$GPU_SELECT" == *GPU-* ]]; then
  GPU_NOTE="${NSEL} GPU(s) selected by UUID (unambiguous): $GPU_SELECT"
elif [[ "$NGPU" -eq "$NSEL" ]]; then
  GPU_NOTE="${NSEL} of ${NGPU} visible GPU(s): $GPU_SELECT — nvidia-smi is already scoped to this job, indices agree"
else
  GPU_NOTE="${NSEL} of ${NGPU} visible GPU(s): $GPU_SELECT"
  GPU_WARN="AMBIGUOUS GPU SELECTION — nvidia-smi sees ${NGPU} GPUs but this job holds ${NSEL}. CUDA_VISIBLE_DEVICES indices are job-relative while nvidia-smi indices are physical, so '$GPU_SELECT' may select a card this job does not own. Set MONITOR_GPUS to the physical index or the GPU UUID to be certain."
fi
if $HAS_SMI && [[ "$NGPU" -gt 0 ]]; then
  BUF=""; command -v stdbuf &>/dev/null && BUF="stdbuf -oL"
  # shellcheck disable=SC2086
  $BUF nvidia-smi \
      --query-gpu=index,uuid,memory.used,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --loop-ms=$INTERVAL_MS \
      > "$SMI_STREAM" 2>/dev/null & SMI_PID=$!
  # Confirm the stream is actually producing before trusting it for the whole run.
  # If this nvidia-smi build ignores --loop-ms, or buffering defeats us anyway, the
  # file stays empty and we fall back to a per-tick one-shot rather than writing a
  # run's worth of blank GPU columns.
  sleep 1
  if [[ ! -s "$SMI_STREAM" ]]; then
    kill "$SMI_PID" 2>/dev/null || true
    wait "$SMI_PID" 2>/dev/null || true
    SMI_PID=""
  fi
fi
SMI_MODE="one-shot per tick"
[[ -n "$SMI_PID" ]] && SMI_MODE="streaming (--loop-ms=$INTERVAL_MS)"

echo "Monitor starting — interval=${INTERVAL}s  device=${DEVICE:-none}${DEVICE_NOTE}"
echo "  cpu/ram/disk : /proc (stat=$($HAS_PROCSTAT && echo yes || echo no)," \
     "meminfo=$($HAS_MEMINFO && echo yes || echo no)," \
     "diskstats=$($HAS_DISKSTATS && echo yes || echo no))"
echo "  nvidia-smi   : $($HAS_SMI && echo "yes, $SMI_MODE" || echo no)   gpus=$NGPU"
echo "  dcgmi        : $($HAS_DCGMI && echo "yes (one-shot per tick)" || echo no)"
echo "  gpus         : $GPU_NOTE"
[[ -n "$GPU_WARN" ]] && echo "  WARNING      : $GPU_WARN"
echo "  output       : $OUTFILE"
echo ""

# ── CSV header
# Column names and order are unchanged from the runs already in results/, so
# existing analysis keeps working. dcgmi emits SEVEN values for the seven -e
# fields requested below; the labels here name them in the order dcgmi returns.
{
  printf "timestamp,"
  printf "cpu_util_pct,"
  printf "ram_used_mb,ram_total_mb,ram_avail_mb,ram_used_pct,"
  printf "disk_r_s,disk_w_s,disk_rkB_s,disk_wkB_s,disk_r_await_ms,disk_w_await_ms,disk_util_pct,"
  printf "gpu_mem_used_mb,gpu_mem_total_mb,gpu_mem_free_mb,gpu_util_pct"
  $HAS_DCGMI && printf ",dcgmi_power_w,dcgmi_gpu_util_pct,dcgmi_mem_copy_util_pct,dcgmi_gr_active,dcgmi_sm_active,dcgmi_pcie_tx_bytes,dcgmi_pcie_rx_bytes"
  printf "\n"
} > "$OUTFILE"

cleanup() {
  [[ -n "$MONITOR_PID" ]] && { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
  [[ -n "$SMI_PID"     ]] && { kill "$SMI_PID"     2>/dev/null || true; wait "$SMI_PID"     2>/dev/null || true; }
  local n elapsed
  n=$(cat "$SAMPLES_FILE")
  elapsed=$(( $(date +%s) - T_START ))
  echo ""
  # Report the ACHIEVED cadence, not the requested one. With the blocking tools
  # gone this should now sit within a few percent of $INTERVAL; if it does not,
  # dcgmi is the remaining per-tick subprocess and is the thing to drop.
  echo "Monitor stopped — $n samples over ${elapsed}s written to $OUTFILE"
  # Measured from the ROW TIMESTAMPS, not from $elapsed: T_START precedes the
  # nvidia-smi stream probe and the last row precedes process exit, so dividing
  # total wall time by row count reports a cadence slower than the one actually
  # achieved. The gap between first and last row is the real spacing.
  if [[ "$n" -gt 1 ]]; then
    local first last f_s l_s
    first=$(awk -F, 'NR == 2 { print $1; exit }' "$OUTFILE" 2>/dev/null || true)
    last=$(tail -n 1 "$OUTFILE" 2>/dev/null | cut -d, -f1 || true)
    f_s=$(date -d "$first" +%s 2>/dev/null || echo "")
    l_s=$(date -d "$last"  +%s 2>/dev/null || echo "")
    if [[ -n "$f_s" && -n "$l_s" ]]; then
      echo "  effective sample interval: $(awk -v e="$((l_s - f_s))" -v n="$n"         'BEGIN { printf "%.2f", e / (n - 1) }')s  (requested ${INTERVAL}s, over $n rows)"
    fi
  fi
  rm -f "$SAMPLES_FILE" "$SMI_STREAM"
}
trap cleanup EXIT

# ── Sampling loop
_monitor_loop() {
  # Previous counter values, for delta arithmetic. Seeded from a first read so the
  # very first emitted row already covers a real interval instead of reporting
  # since-boot averages — the exact trap the old `mpstat N 2` two-report dance
  # existed to avoid.
  local prev_cpu_tot=0 prev_cpu_idle=0 prev_ms=0
  local prev_disk="" cur_disk=""

  if $HAS_PROCSTAT; then
    read -r prev_cpu_tot prev_cpu_idle < <(
      awk '/^cpu / { t = 0; for (i = 2; i <= 9; i++) t += $i; print t, $5 + $6; exit }' /proc/stat)
  fi
  if $HAS_DISKSTATS; then
    # fields: 4 reads done, 6 sectors read, 7 ms reading, 8 writes done,
    #        10 sectors written, 11 ms writing, 13 ms doing I/O
    prev_disk=$(awk -v d="$DEVICE" '$3 == d { print $4, $6, $7, $8, $10, $11, $13; exit }' /proc/diskstats)
  fi
  prev_ms=$(date +%s%3N)

  local tick=0
  local t0_ms; t0_ms=$(date +%s%3N)

  while true; do
    # Pace FIRST: the seed reads above just happened, so an immediate sample would
    # cover a ~0s window and divide by it.
    tick=$((tick + 1))
    local target now delta
    target=$((t0_ms + tick * INTERVAL_MS))
    now=$(date +%s%3N)
    delta=$((target - now))
    if (( delta > 0 )); then
      sleep "$(awk -v d="$delta" 'BEGIN { printf "%.3f", d / 1000 }')"
    fi

    TS=$(date +"%Y-%m-%dT%H:%M:%S")
    now=$(date +%s%3N)
    local dt_ms=$((now - prev_ms))
    (( dt_ms <= 0 )) && dt_ms=1
    prev_ms=$now

    # ── CPU: 100 * busy/total over the interval just elapsed.
    CPU_FIELDS=""
    if $HAS_PROCSTAT; then
      local cur_tot cur_idle
      read -r cur_tot cur_idle < <(
        awk '/^cpu / { t = 0; for (i = 2; i <= 9; i++) t += $i; print t, $5 + $6; exit }' /proc/stat)
      CPU_FIELDS=$(awk -v ct="$cur_tot" -v ci="$cur_idle" -v pt="$prev_cpu_tot" -v pi="$prev_cpu_idle" '
        BEGIN {
          dt = ct - pt; di = ci - pi
          if (dt <= 0) exit                       # no time passed, or counters reset
          v = 100 * (dt - di) / dt
          # CPU hotplug or suspend/resume can move idle non-monotonically, which
          # puts v outside [0,100]. Clamp: a wrong-but-bounded value is easier to
          # spot in a plot than a 150% spike that looks like a real event.
          if (v < 0) v = 0; if (v > 100) v = 100
          printf "%.1f", v
        }')
      prev_cpu_tot=$cur_tot; prev_cpu_idle=$cur_idle
    fi
    CPU_FIELDS="${CPU_FIELDS:-}"

    # ── RAM from /proc/meminfo, in the same four columns `free -m` gave:
    # used = total - free - buffers - (cached + reclaimable), matching procps, and
    # available = MemAvailable (excludes cache that is not actually reclaimable).
    RAM_FIELDS=""
    if $HAS_MEMINFO; then
      RAM_FIELDS=$(awk '
        /^MemTotal:/     { tot = $2 }
        /^MemFree:/      { fre = $2 }
        /^MemAvailable:/ { av  = $2 }
        /^Buffers:/      { buf = $2 }
        /^Cached:/       { cac = $2 }
        /^SReclaimable:/ { rec = $2 }
        END {
          if (tot > 0) {
            used = (tot - fre - buf - cac - rec) / 1024
            printf "%.0f,%.0f,%.0f,%.1f", used, tot / 1024, av / 1024, used * 100 / (tot / 1024)
          }
        }' /proc/meminfo 2>/dev/null || true)
    fi
    RAM_FIELDS="${RAM_FIELDS:-,,,}"

    # ── Disk from /proc/diskstats deltas. Sectors are always 512 B in this file
    # regardless of the device's real sector size, so kB = sectors / 2.
    # await is per-completed-request; with no completions in the window it is 0,
    # not a division by zero.
    DISK_FIELDS=""
    if $HAS_DISKSTATS; then
      cur_disk=$(awk -v d="$DEVICE" '$3 == d { print $4, $6, $7, $8, $10, $11, $13; exit }' /proc/diskstats)
      if [[ -n "$cur_disk" && -n "$prev_disk" ]]; then
        DISK_FIELDS=$(awk -v cur="$cur_disk" -v prv="$prev_disk" -v dtms="$dt_ms" '
          BEGIN {
            split(cur, c, " "); split(prv, p, " ")
            dt = dtms / 1000
            dr = c[1] - p[1]; drs = c[2] - p[2]; drm = c[3] - p[3]
            dw = c[4] - p[4]; dws = c[5] - p[5]; dwm = c[6] - p[6]
            dio = c[7] - p[7]
            # A counter that went backwards means the device was reset or renamed;
            # emit blanks rather than a negative rate.
            if (dr < 0 || dw < 0) exit
            printf "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                   dr / dt, dw / dt, (drs / 2) / dt, (dws / 2) / dt,
                   (dr > 0 ? drm / dr : 0), (dw > 0 ? dwm / dw : 0),
                   (dio / dtms) * 100
          }')
      fi
      [[ -n "$cur_disk" ]] && prev_disk="$cur_disk"
    fi
    DISK_FIELDS="${DISK_FIELDS:-,,,,,,}"

    # ── GPU (averaged over the GPUs THIS JOB was allocated).
    # index and uuid are both queried because CUDA_VISIBLE_DEVICES may hold
    # either form. Empty/unset means no restriction, so take every GPU.
    GPU_FIELDS=""
    if $HAS_SMI; then
      local gpu_raw=""
      if [[ -n "$SMI_PID" ]]; then
        # Latest block from the background stream: one line per GPU.
        gpu_raw=$(tail -n "$NGPU" "$SMI_STREAM" 2>/dev/null || true)
      else
        gpu_raw=$(nvidia-smi \
          --query-gpu=index,uuid,memory.used,memory.total,memory.free,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null || true)
      fi
      GPU_FIELDS=$(printf '%s\n' "$gpu_raw" \
        | awk -F',' -v vis="$GPU_SELECT" '
            BEGIN {
              nsel = split(vis, want, /,[ ]*/)
              for (i = 1; i <= nsel; i++) { gsub(/^[ \t]+|[ \t]+$/, "", want[i]); sel[want[i]] = 1 }
            }
            NF >= 6 {
              idx = $1; uuid = $2
              gsub(/^[ \t]+|[ \t]+$/, "", idx); gsub(/^[ \t]+|[ \t]+$/, "", uuid)
              if (vis == "" || (idx in sel) || (uuid in sel)) { u+=$3; t+=$4; f+=$5; g+=$6; n++ }
            }
            END { if (n) printf "%.1f,%.1f,%.1f,%.1f", u/n, t/n, f/n, g/n }')
    fi
    GPU_FIELDS="${GPU_FIELDS:-,,,}"

    # ── DCGMI (optional extended GPU metrics).
    # The only subprocess left on the critical path. `dmon -c 1` returns promptly
    # (it does not time-average the way mpstat/iostat did), but if the achieved
    # interval printed at shutdown drifts above $INTERVAL this is the cause —
    # either drop dcgmi or convert it to a background `dmon -d $INTERVAL_MS`
    # stream the way nvidia-smi is handled above.
    DCGMI_FIELDS=""
    if $HAS_DCGMI; then
      DCGMI_FIELDS=$(dcgmi dmon -e 203,204,155,1001,1002,1009,1010 -c 1 2>/dev/null \
        | awk 'NR>2 { pw+=$4; gu+=$2; mc+=$3; gr+=$5; sm+=$6; ptx+=$7; prx+=$8; n++ }
               END  { if(n) printf ",%.1f,%.1f,%.1f,%.3f,%.3f,%.1f,%.1f", pw/n,gu/n,mc/n,gr/n,sm/n,ptx/n,prx/n }')
    fi

    printf "%s,%s,%s,%s,%s%s\n" \
      "$TS" "$CPU_FIELDS" "$RAM_FIELDS" "$DISK_FIELDS" "$GPU_FIELDS" "$DCGMI_FIELDS" \
      >> "$OUTFILE"
    echo $(( $(cat "$SAMPLES_FILE") + 1 )) > "$SAMPLES_FILE"
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

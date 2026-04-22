"""
hardware_monitor.py
-------------------
Background hardware monitor that runs all 4 commands the supervisor asked for:
  top, nvidia-smi, dcgmi, disk stats (iostat + df)

It also accepts event markers so you can align hardware activity with
pipeline stage timings from RunTrace.latency_ms.

Usage:
    monitor = HardwareMonitor(log_dir="logs/hardware", interval=2)
    monitor.start()
    monitor.mark("index:START")
    pipeline.DB_build_index(...)
    monitor.mark("index:END")
    ...
    monitor.stop()
    monitor.save()           # → logs/hardware/hw_YYYYMMDD_HHMMSS.json
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime


class HardwareMonitor:

    def __init__(self, log_dir: str = "logs/hardware", interval: int = 2):
        self.log_dir = log_dir
        self.interval = interval
        self._running = False
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._records: dict[str, list] = {
            "meta":      [],
            "events":    [],   # pipeline stage markers
            "cpu":       [],
            "gpu_smi":   [],
            "gpu_dcgmi": [],
            "disk":      [],
        }
        os.makedirs(log_dir, exist_ok=True)

    def start(self) -> "HardwareMonitor":
        self._running = True
        self._records["meta"].append({"event": "start", "wall": datetime.now().isoformat(), "t": time.time()})
        self.mark("MONITOR:START")

        for name, fn in [
            ("cpu",       self._poll_cpu),
            ("gpu_smi",   self._poll_gpu_smi),
            ("gpu_dcgmi", self._poll_gpu_dcgmi),
            ("disk",      self._poll_disk),
        ]:
            t = threading.Thread(target=fn, daemon=True, name=f"hw_{name}")
            t.start()
            self._threads.append(t)

        print(f"[HardwareMonitor] Started (interval={self.interval}s, log_dir={self.log_dir})")
        return self

    def stop(self) -> "HardwareMonitor":
        self.mark("MONITOR:STOP")
        self._running = False
        for t in self._threads:
            t.join(timeout=self.interval + 2)
        self._records["meta"].append({"event": "stop", "wall": datetime.now().isoformat(), "t": time.time()})
        print("[HardwareMonitor] Stopped")
        return self

    def mark(self, label: str) -> None:
        """
        Stamp a named event. Call around every major pipeline stage so
        hardware timeline can be correlated with component activity.
        e.g. monitor.mark("retrieval:START") ... monitor.mark("retrieval:END")
        """
        with self._lock:
            self._records["events"].append({"t": time.time(), "label": label})

    def save(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"hw_{ts}.json")
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"[HardwareMonitor] Saved → {path}")
        return path

    # ------------------------------------------------------------------
    # Background pollers
    # ------------------------------------------------------------------

    def _append(self, key: str, record: dict) -> None:
        with self._lock:
            self._records[key].append(record)

    def _poll_cpu(self) -> None:
        """top -bn1: one-shot CPU/memory snapshot every interval."""
        while self._running:
            try:
                raw = subprocess.check_output(
                    ["top", "-bn1"], stderr=subprocess.DEVNULL, text=True
                )
                summary = "\n".join(raw.splitlines()[:5])
                self._append("cpu", {"t": time.time(), "raw": summary})
            except Exception as e:
                self._append("cpu", {"t": time.time(), "error": str(e)})
            time.sleep(self.interval)

    def _poll_gpu_smi(self) -> None:
        """nvidia-smi: GPU util, memory, power, temperature."""
        query = (
            "timestamp,name,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu"
        )
        while self._running:
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, text=True,
                )
                self._append("gpu_smi", {"t": time.time(), "raw": raw.strip()})
            except FileNotFoundError:
                self._append("gpu_smi", {"t": time.time(), "error": "nvidia-smi not found"})
                break
            except Exception as e:
                self._append("gpu_smi", {"t": time.time(), "error": str(e)})
            time.sleep(self.interval)

    def _poll_gpu_dcgmi(self) -> None:
        """dcgmi dmon: DCGM metrics."""
        while self._running:
            try:
                raw = subprocess.check_output(
                    ["dcgmi", "dmon", "-e", "203,204,1001,1002", "-c", "1"],
                    stderr=subprocess.DEVNULL, text=True,
                )
                self._append("gpu_dcgmi", {"t": time.time(), "raw": raw.strip()})
            except FileNotFoundError:
                self._append("gpu_dcgmi", {"t": time.time(), "error": "dcgmi not found"})
                break
            except Exception as e:
                self._append("gpu_dcgmi", {"t": time.time(), "error": str(e)})
            time.sleep(self.interval)

    def _poll_disk(self) -> None:
        """df + iostat: disk usage and I/O throughput."""
        while self._running:
            record: dict = {"t": time.time()}
            try:
                record["df"] = subprocess.check_output(
                    ["df", "-h", "/"], stderr=subprocess.DEVNULL, text=True
                ).strip()
            except Exception as e:
                record["df_error"] = str(e)
            try:
                record["iostat"] = subprocess.check_output(
                    ["iostat", "-x", "1", "1"], stderr=subprocess.DEVNULL, text=True
                ).strip()
            except FileNotFoundError:
                record["iostat"] = "iostat not available (install sysstat)"
            except Exception as e:
                record["iostat_error"] = str(e)
            self._append("disk", record)
            time.sleep(self.interval)

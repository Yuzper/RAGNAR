"""
Run configuration — one source of truth for what a run was.

Both halves of the old problem come from the same root: configuration lived in
Python source. That made a sweep impossible without editing code (no job
arrays), and it made provenance unreliable, because the description saved into a
report was hand-written text maintained separately from the values that actually
built the pipeline. The two drift, and a saved result stops being traceable.

Here the loaded config is what constructs the components *and* what is recorded,
so the two cannot disagree. With one configuration per SLURM job, the config
file plus the git SHA is the entire identity of a run.

Index-defining vs query-time
----------------------------
`index_fingerprint()` returns the subset that determines the *vectors*: the
embedder, the metric, the index geometry, the corpus, and whether titles were
prepended. It is written into the saved index at build time and checked when an
online run loads it. Everything else — nprobe, top_k, reranker, generator — is
query-time and free to vary across runs sharing one build, which is what makes
"build once, evaluate many configurations" safe rather than merely convenient.
"""

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

# Dotted paths whose values determine the contents of the index. Changing any of
# these invalidates an existing build; changing anything else does not.
INDEX_DEFINING_KEYS = (
    "embedder.model",
    "embedder.metric",
    "index.type",
    "index.nlist",
    "index.m_pq",
    "index.nbits_pq",
    "index.train_size",
    "offline.data_path",
    "offline.prepend_titles",
    # The chunker determines what a vector IS, so it belongs here as much as the
    # embedder does. Without it, two builds differing only in chunking would
    # fingerprint identically and an online run could query one believing it was
    # the other — the exact failure this mechanism exists to prevent.
    "chunker.type",
    "chunker.size",
    "chunker.overlap",
    "chunker.min_chunk_words",
)


class ConfigError(Exception):
    """Raised for a malformed config file or a bad --set override."""


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_provenance() -> dict[str, Any]:
    """
    Commit the run was launched from, and whether the tree was clean.

    dirty=True means the recorded SHA does not fully describe the code that ran,
    which is worth knowing before a result goes into a thesis.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


class RunConfig:
    """A loaded, override-resolved configuration."""

    def __init__(self, data: dict, source: str, overrides: list[str] | None = None):
        self._data = data
        self.source = source
        self.overrides = list(overrides or [])

    # ── loading ────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls, path: str | Path | None = None, overrides: list[str] | None = None,
    ) -> "RunConfig":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"Config file {path} did not parse to a mapping")

        cfg = cls(data, source=str(path))
        for override in overrides or []:
            cfg._apply_override(override)
        cfg._validate()
        return cfg

    def _apply_override(self, override: str) -> None:
        """
        Apply one `dotted.key=value` override in place.

        The key must already exist and the new value must keep the existing
        type. A sweep that silently ignored a mistyped key would produce a set
        of runs that look different in the filename and are identical in fact —
        the most expensive kind of mistake available here.
        """
        if "=" not in override:
            raise ConfigError(f"Malformed --set '{override}', expected key=value")
        key, raw = override.split("=", 1)
        key = key.strip()

        parent, leaf = self._resolve_parent(key)
        if leaf not in parent:
            raise ConfigError(
                f"--set '{key}' does not exist in {self.source}. "
                f"Available keys at this level: {sorted(parent)}"
            )

        # yaml.safe_load gives the natural type: 128 -> int, true -> bool,
        # 0.5 -> float, everything else -> str.
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw

        current = parent[leaf]
        if current is not None and value is not None:
            # bool is a subclass of int; treat them as distinct so
            # `--set offline.prepend_titles=1` is rejected rather than silently
            # accepted as True.
            if isinstance(current, bool) != isinstance(value, bool) or (
                not isinstance(current, type(value))
                and not (isinstance(current, float) and isinstance(value, int))
            ):
                raise ConfigError(
                    f"--set '{key}={raw}' has type {type(value).__name__}, "
                    f"but {key} is {type(current).__name__} in {self.source}"
                )
        parent[leaf] = value
        self.overrides.append(override)

    def _resolve_parent(self, key: str) -> tuple[dict, str]:
        parts = key.split(".")
        node: Any = self._data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"--set '{key}': no section '{part}' in {self.source}")
            node = node[part]
        if not isinstance(node, dict):
            raise ConfigError(f"--set '{key}': '{'.'.join(parts[:-1])}' is not a section")
        return node, parts[-1]

    def _validate(self) -> None:
        for section in ("embedder", "index", "offline", "online", "chunker"):
            if section not in self._data:
                raise ConfigError(f"{self.source} is missing the '{section}' section")
        for key in INDEX_DEFINING_KEYS:
            self.get(key)  # raises if absent

        # Checked here rather than only in build_chunker so a bad sweep value
        # fails at launch, not after the embedder has loaded and the job has
        # spent minutes getting to the first batch. Imported locally to keep
        # config.py free of a module-level dependency on the components package.
        from .components.Chunkers.ChunkerHelper import CHUNKER_TYPES

        chunker_type = self.get("chunker.type")
        if chunker_type not in CHUNKER_TYPES:
            raise ConfigError(
                f"chunker.type must be one of {CHUNKER_TYPES}, got '{chunker_type}'"
            )
        size, overlap = self.get("chunker.size"), self.get("chunker.overlap")
        if size <= 0:
            raise ConfigError(f"chunker.size must be positive, got {size}")
        if overlap < 0:
            raise ConfigError(f"chunker.overlap must be >= 0, got {overlap}")
        if overlap >= size:
            # stride <= 0: the window never advances and the build produces
            # chunks forever off a single document.
            raise ConfigError(
                f"chunker.overlap ({overlap}) must be smaller than chunker.size ({size})"
            )

        k = self.get("online.eval.retrieval_k")
        top_k = self.get("online.retriever.top_k")
        if k >= top_k:
            # Not a style rule: when the metric window equals the candidate pool
            # every retrieved-relevant chunk falls inside it and pool_recall_at_k
            # collapses to a constant 1.0.
            raise ConfigError(
                f"online.eval.retrieval_k ({k}) must be well below "
                f"online.retriever.top_k ({top_k})"
            )
        if self.get("online.reranker.top_k") > top_k:
            raise ConfigError(
                f"online.reranker.top_k ({self.get('online.reranker.top_k')}) cannot "
                f"exceed online.retriever.top_k ({top_k})"
            )
        warmup = self.get("online.eval.warmup_queries")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ConfigError(
                f"online.eval.warmup_queries must be a non-negative integer, "
                f"got {warmup!r}"
            )

        rr_type = self.get("online.reranker.type")
        if rr_type not in ("cross_encoder", "passthrough"):
            raise ConfigError(
                f"online.reranker.type must be 'cross_encoder' or 'passthrough', "
                f"got '{rr_type}'"
            )

    # ── access ─────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = ...) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not ...:
                    return default
                raise ConfigError(f"Missing config key '{key}' in {self.source}")
            node = node[part]
        return node

    def section(self, key: str) -> dict:
        value = self.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"Config key '{key}' is not a section")
        return copy.deepcopy(value)

    @property
    def data(self) -> dict:
        return copy.deepcopy(self._data)

    # ── provenance ─────────────────────────────────────────────────────
    def index_fingerprint(self) -> dict[str, Any]:
        """The index-defining subset, flat and comparable."""
        return {key: self.get(key) for key in INDEX_DEFINING_KEYS}

    def provenance(self, extra: dict | None = None) -> dict[str, Any]:
        """
        Everything needed to identify this run, for embedding in a report.

        `resolved` is the config after overrides — the values that actually ran,
        not what the file said before the job array modified it.
        """
        prov = {
            "config_source": self.source,
            "overrides": list(self.overrides),
            **git_provenance(),
            "resolved": self.data,
        }
        if extra:
            prov.update(extra)
        return prov

    def describe(self) -> str:
        lines = [f"  config           : {self.source}"]
        if self.overrides:
            lines.append(f"  overrides        : {', '.join(self.overrides)}")
        git = git_provenance()
        dirty = "  (UNCOMMITTED CHANGES)" if git["git_dirty"] else ""
        lines.append(f"  git              : {git['git_sha']} on {git['git_branch']}{dirty}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RunConfig({self.source}, overrides={self.overrides})"


def compare_fingerprints(built: dict | None, current: dict) -> list[str]:
    """
    Differences between the config an index was built with and the current one.

    Returns a list of human-readable mismatches; empty means compatible. `built`
    is None for indexes saved before fingerprinting existed — the caller decides
    whether to warn or refuse.
    """
    if built is None:
        return []
    diffs = []
    for key in sorted(set(built) | set(current)):
        was, now = built.get(key, "<absent>"), current.get(key, "<absent>")
        if was != now:
            diffs.append(f"{key}: index built with {was!r}, config says {now!r}")
    return diffs


# ── CLI helpers ────────────────────────────────────────────────────────

def add_config_args(parser) -> None:
    """Attach --config/--set to an argparse parser (shared by both phases)."""
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="path to the run config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set online.nprobe=128 (repeatable)",
    )


def main(argv: list[str] | None = None) -> int:
    """
    Read a value out of a config file.

    Exists so shell scripts have the same source of truth as Python — the SLURM
    job needs the generator model name to pull it before the eval starts, and
    hard-coding it there is how a job ends up pulling one model and evaluating
    another.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="python -m rag_pipeline.config")
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--get", help="dotted key to print, e.g. online.generator.model")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="print the resolved config as JSON")
    args = parser.parse_args(argv)

    try:
        cfg = RunConfig.load(args.config, args.overrides)
    except ConfigError as exc:
        print(f"[config] {exc}")
        return 1

    if args.get:
        try:
            print(cfg.get(args.get))
        except ConfigError as exc:
            print(f"[config] {exc}")
            return 1
    elif args.json:
        print(json.dumps(cfg.provenance(), indent=2))
    else:
        print(yaml.safe_dump(cfg.data, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

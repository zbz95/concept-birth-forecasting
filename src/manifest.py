"""Run manifests and the causality gate.

Every artifact this pipeline writes carries a manifest: the config hash, the
code commit, and the hashes of every input artifact it was built from. That
chain is what makes the leakage checklist auditable after the fact rather than
by assertion.

Principle 1 (causality gate) is enforced here, not by convention: `assert_causal`
is called by every phase that produces a time-indexed artifact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-able object (sorted keys, no whitespace drift)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def config_hash(cfg: dict | None = None) -> str:
    return sha256_obj(cfg if cfg is not None else load_config())


def code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        pass
    return "no-git"


# ---------------------------------------------------------------------------
# Causality gate
# ---------------------------------------------------------------------------

class LeakageError(AssertionError):
    """Raised when an artifact indexed by T would absorb information after T."""


def assert_causal(
    artifact: str,
    as_of: date | str,
    max_observed_date: date | str | None,
    *,
    note: str = "",
) -> None:
    """No artifact indexed by time T may use information dated after T.

    `as_of` is the artifact's index date T. `max_observed_date` is the latest
    input datum that actually went into it. The second must not exceed the first.
    """
    if max_observed_date is None:
        return
    t = _as_date(as_of)
    obs = _as_date(max_observed_date)
    if obs > t:
        raise LeakageError(
            f"{artifact}: as_of={t.isoformat()} but absorbed data dated "
            f"{obs.isoformat()} ({(obs - t).days}d after T). {note}"
        )


def _as_date(v: date | str) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    artifact: str
    phase: str
    config_hash: str
    code_commit: str
    created_utc: str
    inputs: dict[str, str] = field(default_factory=dict)   # path -> sha256
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    as_of: str | None = None
    max_observed_date: str | None = None

    @classmethod
    def build(
        cls,
        artifact: str,
        phase: str,
        *,
        inputs: Iterable[str | Path] = (),
        params: dict | None = None,
        stats: dict | None = None,
        as_of: date | str | None = None,
        max_observed_date: date | str | None = None,
        cfg: dict | None = None,
    ) -> "Manifest":
        if as_of is not None:
            assert_causal(artifact, as_of, max_observed_date)
        return cls(
            artifact=artifact,
            phase=phase,
            config_hash=config_hash(cfg),
            code_commit=code_commit(),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            inputs={str(p): sha256_file(p) for p in inputs},
            params=params or {},
            stats=stats or {},
            as_of=None if as_of is None else _as_date(as_of).isoformat(),
            max_observed_date=(
                None if max_observed_date is None
                else _as_date(max_observed_date).isoformat()
            ),
        )

    def write(self, target: str | Path) -> Path:
        """Write alongside the artifact as `<target>.manifest.json`."""
        out = Path(str(target) + ".manifest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return out


# ---------------------------------------------------------------------------
# Append-only logs (kills, merges, flags) — reversible by construction
# ---------------------------------------------------------------------------

def log_event(path: str | Path, event: dict) -> None:
    """Append one JSONL row. Nothing is ever deleted, only superseded.

    Every row carries `undone: false`; an undo is a NEW row referencing the
    original's `event_id`, never an edit or a delete (Principle 3).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code_commit(),
        "undone": False,
        **event,
    }
    row.setdefault("event_id", sha256_obj(row)[:16])
    with open(p, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# LLM-judge filter
# ---------------------------------------------------------------------------

def judge_dropped(cfg: dict | None = None) -> set[str]:
    """Concept labels the LLM judge marked DROP, or empty if the judge is off.

    Verdicts are keyed by the concept label as exported for judging — the union
    of cluster labels across every vocab_T — so applying them at the cluster
    level is exactly what was judged. A term with no verdict is kept: absence of
    a judgement is never evidence for deletion.
    """
    cfg = cfg or load_config()
    if not cfg.get("llm_judge", {}).get("enabled"):
        return set()
    path = Path("data/interim/judge_verdicts.parquet")
    if not path.exists():
        raise FileNotFoundError(
            "llm_judge.enabled is true but data/interim/judge_verdicts.parquet is "
            "missing — run src/extraction/judge_ingest.py first")
    import duckdb
    rows = duckdb.connect().execute(
        f"SELECT term FROM read_parquet('{path}') WHERE verdict = 'DROP'").fetchall()
    return {t for (t,) in rows}

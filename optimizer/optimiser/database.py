"""Append-only JSONL log of every evaluation (the optimiser's memory; enables resume)."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalRecord:
    index: int
    label: str
    x: Dict[str, float]
    status: str                      # ok | failed | infeasible
    metrics: Optional[Dict[str, float]] = None      # cl, cd, cm, ld
    message: str = ""
    score: Optional[float] = None
    terms: Dict[str, float] = field(default_factory=dict)
    mode: str = ""
    wall_s: float = 0.0
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=float)

    @staticmethod
    def from_json(line: str) -> "EvalRecord":
        return EvalRecord(**json.loads(line))


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, rec: EvalRecord) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(rec.to_json() + "\n")

    def load(self) -> List[EvalRecord]:
        if not self.path.is_file():
            return []
        out: List[EvalRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(EvalRecord.from_json(line))
        return out

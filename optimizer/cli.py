"""Shared command-line plumbing for the four stage scripts."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import Config, ConfigError, load_config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "config" / "config.yaml"), help="path to the YAML config")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set flow.speed=80 (repeatable)")
    p.add_argument("--objective", choices=["max_ld", "min_cd", "max_ld_cm", "multi"],
                   help="objective mode for this run (overrides objective.mode)")
    p.add_argument("--analysis", choices=["incompressible", "compressible"],
                   help="SimScale analysis type for this run (overrides flow.analysis_type)")
    p.add_argument("--backend", choices=["simscale", "mock"], default="simscale",
                   help="CFD backend: 'simscale' = real CFD (costs credits), 'mock' = free analytic surrogate")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def load(args: argparse.Namespace) -> Config:
    overrides: List[str] = list(args.overrides)
    if args.objective:
        overrides.append(f"objective.mode={args.objective}")
    if args.analysis:
        overrides.append(f"flow.analysis_type={args.analysis}")
    if args.backend == "mock" and not any(o.startswith("geometry.backend=") for o in overrides):
        overrides.append("geometry.backend=analytic")          # a dry run must not need OpenVSP
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
    try:
        return load_config(args.config, overrides)
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")


def parse(parser: argparse.ArgumentParser, argv: Optional[List[str]]) -> argparse.Namespace:
    return parser.parse_args(argv)

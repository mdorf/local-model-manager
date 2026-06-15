"""Command-line entrypoint for local-model-manager."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lmm.discovery import discover_models
from lmm.gguf import read_gguf
from lmm.hardware import detect_hardware
from lmm.llama import get_supported_flags
from lmm.recommend import recommend_config


def cmd_models(args: argparse.Namespace) -> int:
    models = discover_models(args.root)
    if not models:
        print("No models found.")
        return 0
    for m in sorted(models, key=lambda x: (x.family, x.size_label, x.path.name)):
        flags = " ".join(f for f in ["MTP" if m.has_mtp else ""] if f)
        ctx = f"{m.context_length // 1024}K" if m.context_length is not None else "?"
        print(f"{m.path.name}  [{m.arch} {m.size_label} {m.quant} ctx={ctx}] {flags}".rstrip())
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    matches = [m for m in discover_models(args.root)
               if m.path.name == args.model or str(m.path) == args.model]
    if not matches:
        print(f"Model not found: {args.model}")
        return 1
    model = matches[0]
    metadata = read_gguf(model.shards[0]).metadata
    hardware = detect_hardware()
    supported = get_supported_flags() or None
    cfg = recommend_config(model, metadata, hardware, supported=supported,
                           alias=model.path.stem)
    print(f"Recommended for {model.path.name} "
          f"(ctx {cfg.context}, cache {cfg.cache_type}):")
    print("llama-server " + " ".join(cfg.flags))
    print(f"\nFit: {cfg.fit.level} — {cfg.fit.message}")
    for w in cfg.warnings:
        print(f"warning: {w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmm", description="local-model-manager")
    sub = parser.add_subparsers(dest="command", required=True)
    p_models = sub.add_parser("models", help="list discovered local models")
    p_models.add_argument("--root", action="append", default=None,
                          help="model root dir (repeatable); defaults to ~/models")
    p_models.set_defaults(func=cmd_models)
    p_rec = sub.add_parser("recommend", help="recommend a llama-server config for a model")
    p_rec.add_argument("model", help="model filename (or full path) to recommend for")
    p_rec.add_argument("--root", action="append", default=None,
                       help="model root dir (repeatable); defaults to ~/models")
    p_rec.set_defaults(func=cmd_recommend)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if getattr(args, "root", None) is None:
        args.root = [str(Path.home() / "models")]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

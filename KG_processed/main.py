from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KGAT Knowledge Retrieve Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML config file  (default: config.yaml)",
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        help="Override data.session_path in config",
    )
    parser.add_argument(
        "--info", "-i",
        type=str,
        default=None,
        help="Override data.info_path in config",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override data.output_path in config",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override embedding.model_name in config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override embedding.device in config  (default: cuda)",
    )
    parser.add_argument(
        "--layers", "-l",
        type=int,
        default=None,
        help="Override propagation.num_layers in config",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable cache: recompute graph and embeddings from scratch",
    )
    parser.add_argument(
        "--no-graph-cache",
        action="store_true",
        default=False,
        help="Recompute graph even if cache exists",
    )
    parser.add_argument(
        "--no-embed-cache",
        action="store_true",
        default=False,
        help="Recompute embeddings even if cache exists",
    )

    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Apply CLI argument overrides onto the loaded config dict.
    Only overrides keys that were explicitly provided by the user.
    """
    if args.model:
        config.setdefault("embedding", {})["model_name"] = args.model

    if args.device:
        config.setdefault("embedding", {})["device"] = args.device

    if args.layers is not None:
        config.setdefault("propagation", {})["num_layers"] = args.layers

    if args.no_cache:
        config.setdefault("pipeline", {})["use_cache"] = False

    if args.no_graph_cache:
        # Delete graph cache path so pipeline skips cache for graph only
        config.setdefault("data", {})["graph_cache"] = None

    if args.no_embed_cache:
        # Delete embedding cache path so pipeline skips cache for embeddings only
        config.setdefault("data", {})["embedding_cache"] = None

    return config


def main() -> int:
    args = parse_args()

    # ----------------------------------------------------------------
    # Validate config file exists
    # ----------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    # ----------------------------------------------------------------
    # Import here to keep startup fast when --help is used
    # ----------------------------------------------------------------
    import yaml
    from src.pipeline import Pipeline

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = apply_overrides(config, args)

    # ----------------------------------------------------------------
    # Run pipeline
    # ----------------------------------------------------------------
    pipeline = Pipeline(config)

    try:
        pipeline.run(
            session_path=args.session,
            info_path=args.info,
            output_path=args.output,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
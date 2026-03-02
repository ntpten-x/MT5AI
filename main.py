from __future__ import annotations

import argparse
import json
from contextlib import suppress

from loguru import logger

from config import get_settings
from modules.service import TradingBotService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MT5AI Bot entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize SQLite schema")
    backfill_parser = subparsers.add_parser("backfill", help="Backfill historical bars from MT5")
    backfill_parser.add_argument("--symbol", required=False)
    collect_parser = subparsers.add_parser("collect", help="Collect the latest bars from MT5")
    collect_parser.add_argument("--symbol", required=False)
    subparsers.add_parser("heartbeat", help="Check MT5 connectivity and snapshot equity")
    inspect_parser = subparsers.add_parser("check-symbols", help="Inspect broker symbols for the active universe")
    inspect_parser.add_argument("--symbol", action="append", required=False)
    subparsers.add_parser("refresh-models", help="Backfill and retrain models for all active symbols")
    subparsers.add_parser("run-live", help="Start scheduled live trading loop")

    train_parser = subparsers.add_parser("train-xgb", help="Train the tabular/deep model stack")
    train_parser.add_argument("--symbol", required=False)
    train_parser.add_argument("--side", choices=["long", "short", "both"], default="both")

    backtest_parser = subparsers.add_parser("backtest", help="Run a VectorBT backtest")
    backtest_parser.add_argument("--symbol", required=False)

    walk_parser = subparsers.add_parser("walk-forward", help="Run walk-forward analysis")
    walk_parser.add_argument("--symbol", action="append", required=False)
    walk_parser.add_argument("--all", action="store_true", help="Run walk-forward for all active symbols")
    walk_parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write JSON reports to the configured data directory",
    )
    walk_parser.add_argument(
        "--apply-profile",
        action="store_true",
        help="Promote only walk-forward-accepted profiles into runtime optimization profiles",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()
    service = TradingBotService(settings)

    try:
        if args.command == "init-db":
            service.init_database()
            logger.info("Database initialized at {}", settings.database_path)
            return 0

        if args.command == "backfill":
            summary = service.backfill(symbol=args.symbol)
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "collect":
            summary = service.collect_latest(symbol=args.symbol)
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "heartbeat":
            state = service.heartbeat()
            print(json.dumps(state, indent=2, default=str))
            return 0

        if args.command == "check-symbols":
            report = service.inspect_broker_symbols(symbols=args.symbol)
            print(json.dumps(report, indent=2, default=str))
            return 0

        if args.command == "refresh-models":
            summary = service.refresh_models()
            print(json.dumps(summary, indent=2, default=str))
            return 0

        if args.command == "train-xgb":
            metrics = service.train_xgb(symbol=args.symbol, side=args.side)
            print(json.dumps(metrics, indent=2))
            return 0

        if args.command == "backtest":
            stats = service.backtest(symbol=args.symbol)
            print(json.dumps(stats, indent=2, default=str))
            return 0

        if args.command == "walk-forward":
            if args.all or (args.symbol and len(args.symbol) > 1):
                symbols = args.symbol or None
                report = service.walk_forward_many(
                    symbols=symbols,
                    write_report=args.write_report,
                    apply_profile=args.apply_profile,
                )
            else:
                symbol = args.symbol[0] if args.symbol else None
                raw_report = service.walk_forward(symbol=symbol)
                report = raw_report
                if args.write_report:
                    report = {
                        "report": raw_report,
                        "report_path": str(service.walk_forward_engine.write_report(raw_report["symbol"], raw_report)),
                    }
                if args.apply_profile:
                    profile = service.walk_forward_engine.recommend_profile(raw_report["symbol"], raw_report)
                    wrapped_report = report if isinstance(report, dict) and "report" in report else {"report": raw_report}
                    report = {
                        **wrapped_report,
                        "profile": profile,
                        "profile_path": service._write_runtime_profile(raw_report["symbol"], profile),
                    }
            print(json.dumps(report, indent=2, default=str))
            return 0

        if args.command == "run-live":
            service.run_scheduler()
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    finally:
        with suppress(Exception):
            service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

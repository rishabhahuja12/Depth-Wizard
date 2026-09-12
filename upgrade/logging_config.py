"""Structured, timestamped logging for the upgrade pipeline.

`get_logger(name)` returns a stdlib logger that writes DETAILED records —
timestamp, level, logger name, module:line, message — to BOTH the console and a
per-run file at `upgrade/logs/<name>_<YYYYmmdd_HHMMSS>.log`. Reused across
training / evaluation / inference so a long run leaves an auditable trail even if
the shell redirection is lost. No third-party dependency.

Structured context: pass key/value fields via `log_kv(logger, msg, k=v, ...)` to
append `key=value` pairs to a record (e.g. epoch metrics, config dumps).

    from logging_config import get_logger, log_kv
    log = get_logger("train_metric")
    log_kv(log, "epoch done", epoch=3, val_mae=6.21, lr=2.5e-6)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(module)s:%(lineno)d | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Console + per-run file logger. Idempotent: repeated calls for the same name
    return the already-configured logger (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if getattr(logger, "_upgrade_configured", False):
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    d = Path(log_dir) if log_dir else LOG_DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = d / f"{name}_{ts}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger._upgrade_configured = True  # type: ignore[attr-defined]
        logger.info("logging to %s", log_path)
    except OSError as e:  # console-only if the log dir isn't writable
        logger._upgrade_configured = True  # type: ignore[attr-defined]
        logger.warning("file logging disabled (%s): %s", type(e).__name__, e)
    return logger


def log_kv(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    """Log `msg` with structured `key=value` context appended."""
    if fields:
        msg = msg + " | " + " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, msg)

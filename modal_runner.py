"""Bounded, single-container Modal jobs with a persistent conservative budget ledger."""
import argparse
import fcntl
import json
import hashlib
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
BUDGET_USD = 20.0
BUILD_AND_MISC_RESERVE_USD = 2.0
# Above listed GPU + 2 CPU cores + 8 GiB RAM prices on 2026-10-09.
UPPER_HOURLY_RATES = {"A10": 1.50, "L40S": 2.50}
FUNCTION_TIMEOUT = 900
STARTUP_TIMEOUT = 600
RESERVED_SECONDS = FUNCTION_TIMEOUT + STARTUP_TIMEOUT + 30


def ledger_update(path, change):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {
            "budget_usd": BUDGET_USD, "misc_reserve_usd": BUILD_AND_MISC_RESERVE_USD, "jobs": []}
        change(ledger)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(path)
        return ledger


def reserve(path, run_id, gpu):
    reservation = RESERVED_SECONDS / 3600 * UPPER_HOURLY_RATES[gpu]
    def change(ledger):
        if any(job["run_id"] == run_id for job in ledger["jobs"]):
            raise ValueError("Run ID already used. Refusing accidental retry / overwrite.")
        allocated = ledger["misc_reserve_usd"] + sum(j["reserved_usd"] for j in ledger["jobs"])
        if allocated + reservation > ledger["budget_usd"]:
            raise RuntimeError(f"Budget exhausted: ${allocated:.2f} allocated; ${reservation:.2f} requested")
        ledger["jobs"].append({"run_id": run_id, "gpu": gpu, "reserved_usd": reservation,
                               "status": "reserved", "created_at": datetime.now(timezone.utc).isoformat()})
    return ledger_update(path, change)


def finish(path, run_id, **fields):
    def change(ledger):
        next(j for j in ledger["jobs"] if j["run_id"] == run_id).update(fields)
    return ledger_update(path, change)


def download_archive(volume, run_id, output, expected_sha256=None):
    import zipfile
    from unittest.mock import patch
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "download.tmp"
    for attempt in range(3):
        try:
            with temporary.open("wb") as stream:
                # Modal 1.6.1 defaults to host-core-count parallel block reads.
                # Reduce concurrent requests; SHA-256 and ZIP checks remain necessary.
                with patch("modal.volume.multiprocessing.cpu_count", return_value=1):
                    volume.read_file_into_fileobj(f"{run_id}/artifacts.zip", stream)
            with temporary.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if expected_sha256 and digest != expected_sha256:
                raise ValueError("Artifact SHA-256 mismatch")
            with zipfile.ZipFile(temporary) as archive:
                if any(Path(n).name != n for n in archive.namelist()):
                    raise ValueError("Unexpected archive path")
                if archive.testzip() is not None:
                    raise ValueError("Artifact CRC check failed")
                archive.extractall(output)
            temporary.replace(output / "artifacts.zip")
            return
        except (ValueError, zipfile.BadZipFile, zlib.error):
            if attempt == 2:
                raise
            print("Artifact integrity check failed; retrying download only", flush=True)


def make_app(gpu):
    app = modal.App("arc-differentiable-selfplay")
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.8.0", "numpy==2.5.3", "pytest==8.4.2",
                          extra_options="--timeout 120 --retries 2")
             .add_local_dir(ROOT / "arcplay", "/root/project/arcplay", copy=True,
                            ignore=["**/__pycache__/**"])
             .add_local_dir(ROOT / "tests", "/root/project/tests", copy=True,
                            ignore=["**/__pycache__/**"]))
    volume = modal.Volume.from_name("arc-differentiable-selfplay-runs", create_if_missing=True)

    @app.function(image=image, gpu=gpu, cpu=(2, 2), memory=(8192, 8192), serialized=True,
                  volumes={"/runs": volume}, max_containers=1, min_containers=0,
                  scaledown_window=2, timeout=FUNCTION_TIMEOUT, startup_timeout=STARTUP_TIMEOUT,
                  retries=0, single_use_containers=True)
    def worker(run_id: str, config: dict, steps: int, seconds: int, run_tests: bool):
        import json
        import hashlib
        import subprocess
        import sys
        import time
        import zipfile
        from pathlib import Path
        sys.path.insert(0, "/root/project")
        from arcplay.config import Config
        from arcplay.train import run
        output = Path("/runs") / run_id
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        if run_tests:
            checks = subprocess.run([sys.executable, "-m", "pytest", "-q", "/root/project/tests/test_core.py"],
                                    cwd="/root/project", capture_output=True, text=True, timeout=90)
            (output / "tests.txt").write_text(checks.stdout + checks.stderr)
            print(checks.stdout, flush=True)
            if checks.returncode:
                volume.commit()
                raise RuntimeError("Remote numerical tests failed; see tests.txt")
        summary = run(Config(**config), output, steps, seconds, "cuda", log_every=100)
        summary["worker_seconds"] = time.perf_counter() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with zipfile.ZipFile(output / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                if path.name != "artifacts.zip":
                    archive.write(path, path.name)
        with (output / "artifacts.zip").open("rb") as stream:
            summary["artifact_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        volume.commit()
        summary["worker_seconds"] = time.perf_counter() - started
        return summary
    return app, worker, volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=("A10", "L40S"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Download an existing run without launching compute")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("Run ID must be a plain filename")
    if not 1 <= args.seconds <= 600:
        raise ValueError("Remote runs are limited to 1–600 training seconds each")
    config = json.loads(args.config.read_text()) if args.config else {}
    from arcplay.config import Config
    Config(**config)  # Validate before any paid work.
    ledger_path = ROOT / "budget-ledger.json"
    if args.recover:
        volume = modal.Volume.from_name("arc-differentiable-selfplay-runs")
        download_archive(volume, args.run_id, ROOT / "runs" / args.run_id)
        print(f"Recovered {args.run_id}; no training was launched")
        return
    ledger = reserve(ledger_path, args.run_id, args.gpu)
    reserved = ledger["misc_reserve_usd"] + sum(j["reserved_usd"] for j in ledger["jobs"])
    print(f"Conservative total allocation: ${reserved:.2f} / ${BUDGET_USD:.2f}", flush=True)
    started = time.perf_counter()
    try:
        app, worker, volume = make_app(args.gpu)
        with modal.enable_output(), app.run():
            summary = worker.remote(args.run_id, config, args.steps, args.seconds, args.run_tests)
        finish(ledger_path, args.run_id, status="compute_completed", worker_seconds=summary["worker_seconds"],
               estimated_worker_usd=summary["worker_seconds"] / 3600 * UPPER_HOURLY_RATES[args.gpu],
               artifact_sha256=summary.get("artifact_sha256"))
        output = ROOT / "runs" / args.run_id
        download_archive(volume, args.run_id, output, summary.get("artifact_sha256"))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finish(ledger_path, args.run_id, status="completed", worker_seconds=summary["worker_seconds"],
               elapsed_seconds=time.perf_counter() - started,
               estimated_worker_usd=summary["worker_seconds"] / 3600 * UPPER_HOURLY_RATES[args.gpu])
        print(json.dumps({"run_id": args.run_id, "steps": summary["steps_completed"],
                          "steps_per_second": summary["steady_steps_per_second"],
                          "worker_seconds": summary["worker_seconds"], "artifacts": str(output)}, indent=2))
    except BaseException as error:
        finish(ledger_path, args.run_id, status="failed", error=str(error),
               elapsed_seconds=time.perf_counter() - started)
        raise


if __name__ == "__main__":
    main()

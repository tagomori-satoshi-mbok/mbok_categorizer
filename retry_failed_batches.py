"""Retry Gemini candidate batches that previously failed.

Usage example:
    python retry_failed_batches.py \
        --input final_categories.json \
        --candidates-dir dictionary/candidates_full \
        --sleep 0.8

The script scans the specified directory for files named
``batch_<start>_<end>.error.log`` and replays the same range by invoking
``generate_dictionary_candidates.py`` with ``--offset`` and ``--limit``.
``"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ERROR_PATTERN = re.compile(r"batch_(\d+)_(\d+)\.error\.log$")


def find_error_logs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("batch_*_*.error.log") if ERROR_PATTERN.search(path.name))


def build_command(
    offset: int,
    end: int,
    *,
    input_path: Path,
    output_dir: Path,
    batch_size: int,
    sleep: float | None,
    model: str | None,
    api_version: str | None,
    temperature: float | None,
    timeout: float | None,
    max_retries: int | None,
    retry_wait: float | None,
    python_executable: str | None,
) -> list[str]:
    limit = end - offset
    if limit <= 0:
        raise ValueError(f"invalid range offset={offset} end={end}")

    cmd: list[str] = [python_executable or sys.executable, "generate_dictionary_candidates.py"]
    cmd.extend(["--input", str(input_path)])
    cmd.extend(["--output-dir", str(output_dir)])
    cmd.extend(["--batch-size", str(batch_size)])
    cmd.extend(["--offset", str(offset)])
    cmd.extend(["--limit", str(limit)])

    if sleep is not None:
        cmd.extend(["--sleep", str(sleep)])
    if model:
        cmd.extend(["--model", model])
    if api_version:
        cmd.extend(["--api-version", api_version])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    if max_retries is not None:
        cmd.extend(["--max-retries", str(max_retries)])
    if retry_wait is not None:
        cmd.extend(["--retry-wait", str(retry_wait)])

    return cmd


def retry_batches(
    error_logs: Iterable[Path],
    *,
    input_path: Path,
    output_dir: Path,
    batch_size: int,
    sleep: float | None,
    model: str | None,
    api_version: str | None,
    temperature: float | None,
    timeout: float | None,
    max_retries: int | None,
    retry_wait: float | None,
    python_executable: str | None,
) -> None:
    for error_log in error_logs:
        match = ERROR_PATTERN.search(error_log.name)
        if not match:
            continue
        offset = int(match.group(1))
        end = int(match.group(2))
        output_path = output_dir / f"batch_{offset}_{end}.json"

        if output_path.exists():
            print(f"skip {error_log.name}: output already exists")
            error_log.unlink(missing_ok=True)
            continue

        cmd = build_command(
            offset,
            end,
            input_path=input_path,
            output_dir=output_dir,
            batch_size=batch_size,
            sleep=sleep,
            model=model,
            api_version=api_version,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            retry_wait=retry_wait,
            python_executable=python_executable,
        )

        print(f"\n=== retrying batch {offset}_{end} ===")
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED: batch {offset}_{end} ({exc})")
            continue

        # 成功したらエラーログを削除
        error_log.unlink(missing_ok=True)
        print(f"completed batch {offset}_{end}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("final_categories.json"))
    parser.add_argument("--candidates-dir", type=Path, default=Path("dictionary/candidates_full"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-wait", type=float, default=None)
    parser.add_argument("--python", dest="python_executable", default=None, help="python executable to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_logs = find_error_logs(args.candidates_dir)

    if not error_logs:
        print("no error logs found")
        return

    print(f"found {len(error_logs)} error logs")
    retry_batches(
        error_logs,
        input_path=args.input,
        output_dir=args.candidates_dir,
        batch_size=args.batch_size,
        sleep=args.sleep,
        model=args.model,
        api_version=args.api_version,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
        python_executable=args.python_executable,
    )


if __name__ == "__main__":
    main()

"""Gemini API を使って辞書拡張候補を抽出するスクリプト。

使い方:
    export GEMINI_API_KEY=...
    python3 generate_dictionary_candidates.py --batch-size 1000 --limit 3000 --output-dir dictionary/candidates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import requests
import unicodedata

from dictionary import utils as dict_utils


DEFAULT_API_VERSION = "v1beta2"
DEFAULT_MODEL = "models/gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 1000
REQUEST_SLEEP = 0.5
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_WAIT = 5.0

PROJECT_ROOT = Path(__file__).parent


def load_categories(path: Path, offset: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if offset:
        data = data[offset:]
    if limit is not None:
        data = data[:limit]
    return data


def call_gemini(
    api_key: str,
    api_version: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait: float = DEFAULT_RETRY_WAIT,
) -> str:
    url = f"https://generativelanguage.googleapis.com/{api_version}/{model}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }

    attempts = max(1, max_retries)
    last_exc: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")
            break
        except (requests.Timeout, requests.ConnectionError, RuntimeError) as exc:
            last_exc = exc
            if attempt >= attempts:
                response = None
                break
            wait_seconds = max(0.0, retry_wait) * attempt
            print(f"request retry {attempt}/{attempts} after error: {exc}", file=sys.stderr)
            if wait_seconds:
                time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(f"Gemini API request failed after {attempts} attempts: {last_exc}") from last_exc

    data = response.json()
    candidates = data.get("candidates")
    if not candidates:
        raise RuntimeError(f"Unexpected response payload: {data}")

    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text")
    if not text:
        raise RuntimeError(f"Gemini response missing text content: {data}")
    return text


def build_prompt(
    existing_dicts: dict[str, Iterable[str]],
    categories: list[dict[str, Any]],
) -> str:
    header = """
あなたはカテゴリ辞書の候補抽出を手伝うアシスタントです。
以下の既存辞書に含まれない新しい語だけを抽出してください。
各リストは最大でも 20 個までにしてください。
JSON フォーマットで回答し、キーは必ず以下を含めてください:
  - target_keywords
  - size_keywords
  - entity_domains
  - domains_with_kana
  - item_as_entity_domains
  - query_target_keywords
  - query_item_keywords
  - query_entity_keywords
  - query_category_group_keywords
該当する語がなければ空配列にしてください。
既存語と同じ語は含めないでください。

【既存語】
""".strip()

    lines = [header]
    for key, values in existing_dicts.items():
        joined = ", ".join(sorted(dict_utils.ensure_strings(values))) or "(なし)"
        lines.append(f"- {key}: {joined}")

    lines.append("\n【解析対象カテゴリ】")
    for entry in categories:
        cat_id = entry.get("category_id", "?")
        raw_path = entry.get("raw_path", "")
        lines.append(f"{cat_id}: {raw_path}")

    lines.append(
        "\nJSON を返す際は以下の例に従ってください:\n"
        '{"target_keywords": [], '
        '"size_keywords": [], '
        '"entity_domains": [], '
        '"domains_with_kana": [], '
        '"item_as_entity_domains": [], '
        '"query_target_keywords": [], '
        '"query_item_keywords": [], '
        '"query_entity_keywords": [], '
        '"query_category_group_keywords": []}'
    )

    return "\n".join(lines)


def parse_response(text: str) -> dict[str, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini response is not valid JSON: {text}") from exc

    result: dict[str, list[str]] = {}
    fields = (
        "target_keywords",
        "size_keywords",
        "entity_domains",
        "domains_with_kana",
        "item_as_entity_domains",
        "query_target_keywords",
        "query_item_keywords",
        "query_entity_keywords",
        "query_category_group_keywords",
    )

    for key in fields:
        values = data.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"Field '{key}' must be a list, got: {values}")
        cleaned = []
        for value in values:
            if isinstance(value, str):
                normalized = unicodedata.normalize("NFKC", value).strip()
                if normalized:
                    cleaned.append(normalized)
        result[key] = cleaned
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "final_categories.json",
        help="解析対象のカテゴリ JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dictionary" / "candidates",
        help="Gemini 応答を保存するディレクトリ",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="使用する Gemini モデル ID",
    )
    parser.add_argument(
        "--api-version",
        default=DEFAULT_API_VERSION,
        help="Generative Language API バージョン (例: v1beta, v1beta2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="カテゴリを何件ずつまとめて送るか",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="処理開始位置",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理件数を制限する場合に指定",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Gemini 生成温度",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=REQUEST_SLEEP,
        help="リクエスト間に挟むスリープ秒数",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Gemini API リクエストのタイムアウト秒数",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Gemini API 呼び出し時の最大リトライ回数",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=DEFAULT_RETRY_WAIT,
        help="リトライ時の基本待機秒数（試行数に比例して増加）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gemini を呼ばずにプロンプト確認のみ行う",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("GEMINI_API_KEY が設定されていないため dry-run モードに切り替えます。", file=sys.stderr)
        args.dry_run = True

    categories = load_categories(args.input, offset=args.offset, limit=args.limit)
    total = len(categories)
    if total == 0:
        print("対象カテゴリがありません。", file=sys.stderr)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_dicts = {
        "target_keywords": dict_utils.load_set("target_keywords.json"),
        "size_keywords": dict_utils.load_set("size_keywords.json"),
        "entity_domains": dict_utils.load_set("entity_domains.json"),
        "domains_with_kana": dict_utils.load_set("domains_with_kana.json"),
        "item_as_entity_domains": dict_utils.load_set("item_as_entity_domains.json"),
        "query_target_keywords": dict_utils.load_set("query_target_keywords.json"),
        "query_item_keywords": dict_utils.load_set("query_item_keywords.json"),
        "query_entity_keywords": dict_utils.load_set("query_entity_keywords.json"),
        "query_category_group_keywords": dict_utils.load_set("query_category_group_keywords.json"),
    }

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = categories[start:end]
        prompt = build_prompt(existing_dicts, batch)

        output_path = args.output_dir / f"batch_{args.offset + start}_{args.offset + end}.json"
        if output_path.exists():
            print(f"skip existing: {output_path}")
            continue

        if args.dry_run:
            print("===== PROMPT PREVIEW =====")
            print(prompt[:1000], "..." if len(prompt) > 1000 else "")
            print("==========================")
            break

        print(f"requesting batch {args.offset + start} - {args.offset + end} ...")
        try:
            response_text = call_gemini(
                api_key,
                args.api_version,
                args.model,
                prompt,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
        except Exception as exc:
            error_log = output_path.with_suffix(".error.log")
            error_log.write_text(str(exc), encoding="utf-8")
            print(f"request failed (skipped): {exc}", file=sys.stderr)
            continue
        try:
            candidates = parse_response(response_text)
        except ValueError as exc:
            print(f"Failed to parse response: {exc}", file=sys.stderr)
            print("Raw response:", response_text)
            continue

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "range": [int(args.offset + start), int(args.offset + end)],
                    "candidates": candidates,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"saved: {output_path}")

        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()


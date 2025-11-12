"""Gemini Embedding API を使って属性別ベクトルを生成するサンプルスクリプト。

環境変数 GEMINI_API_KEY に API キーを設定してから実行してください。
実際のリクエストができない環境では dry-run モードで確認できます。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Set, Tuple

import requests


DEFAULT_MODEL = "models/text-embedding-004"
DEFAULT_API_VERSION = "v1beta"
DEFAULT_BATCH_SIZE = 16


def load_categories(path: Path, limit: int | None = None, offset: int = 0) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if offset:
        data = data[offset:]
    if limit is not None:
        return data[:limit]
    return data


def build_attribute_text(category: dict, attributes: Iterable[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for attr in attributes:
        value = category.get(attr, "") or ""
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        texts[attr] = normalized
    return texts


def load_processed_pairs(output_path: Path) -> Set[Tuple[str, str]]:
    if not output_path.exists():
        return set()

    processed: Set[Tuple[str, str]] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            category_id = record.get("category_id")
            attribute = record.get("attribute")
            if category_id and attribute:
                processed.add((category_id, attribute))
    return processed


def request_embeddings(
    api_key: str,
    api_version: str,
    model: str,
    texts: list[str],
    timeout: float = 60.0,
) -> list[list[float]]:
    if not texts:
        return []

    url = f"https://generativelanguage.googleapis.com/{api_version}/{model}:batchEmbedContents"
    payload = {
        "requests": [
            {
                "model": model,
                "content": {"parts": [{"text": text}]},
            }
            for text in texts
        ],
    }

    response = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text}"
        )

    data = response.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Unexpected response payload: {data}")

    return [embedding.get("values", []) for embedding in embeddings]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("final_categories_first_1000.json"),
        help="入力となるカテゴリ JSON のパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("final_categories_first_1000_embeddings.jsonl"),
        help="生成した埋め込みを書き出す JSON Lines ファイル",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=["entity", "item", "target"],
        help="埋め込みを作成する属性名",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="カテゴリ数を制限したい場合に指定",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="処理開始位置を指定 (既処理分を飛ばしたいときに使用)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="使用する Gemini Embedding モデル (models/... 形式)",
    )
    parser.add_argument(
        "--api-version",
        default=DEFAULT_API_VERSION,
        help="Generative Language API のバージョン (例: v1beta)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API を呼び出さずに処理内容のみ表示",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="バッチ送信間に挟むスリープ秒数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="API へ送信するバッチサイズ",
    )

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("GEMINI_API_KEY が設定されていないため dry-run に切り替えます。", file=sys.stderr)
        args.dry_run = True

    categories = load_categories(args.input, limit=args.limit, offset=args.offset)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    processed_pairs = load_processed_pairs(output)

    total = len(categories)
    print(f"カテゴリ数: {total}")
    print(f"対象属性: {args.attributes}")
    if processed_pairs:
        print(f"既存レコード: {len(processed_pairs)} 件 (category_id, attribute)")

    batch_size = max(1, args.batch_size)

    with output.open("a", encoding="utf-8") as writer:
        for index in range(0, total, batch_size):
            batch = categories[index : index + batch_size]
            for attribute in args.attributes:
                pending = []
                for category in batch:
                    category_id = category.get("category_id")
                    if not category_id:
                        pending.append("")
                        continue
                    if (category_id, attribute) in processed_pairs:
                        pending.append("")
                        continue
                    pending.append(build_attribute_text(category, [attribute]).get(attribute, ""))

                to_request = [text for text in pending if text]
                if not to_request:
                    continue

                if args.dry_run:
                    embeddings = [[0.0] * 3 for _ in to_request]
                else:
                    embeddings = request_embeddings(api_key, args.api_version, args.model, to_request)

                embed_iter = iter(embeddings)
                for category, text in zip(batch, pending):
                    category_id = category.get("category_id")
                    if not category_id or not text:
                        continue
                    embedding = next(embed_iter, [])
                    record = {
                        "category_id": category_id,
                        "attribute": attribute,
                        "text": category.get(attribute, ""),
                        "embedding": embedding,
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_pairs.add((category_id, attribute))

            processed = min(index + batch_size, total)
            print(f"processed {processed}/{total}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"保存完了: {output}")


if __name__ == "__main__":
    main()


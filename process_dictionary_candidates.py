"""候補語ファイルを正規化して正式な辞書に取り込むスクリプト。"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from dictionary import utils as dict_utils


PROJECT_ROOT = Path(__file__).parent
DEFAULT_CANDIDATES_DIR = PROJECT_ROOT / "dictionary" / "candidates"


KEY_TO_FILENAME = {
    "target_keywords": "target_keywords.json",
    "size_keywords": "size_keywords.json",
    "entity_domains": "entity_domains.json",
    "domains_with_kana": "domains_with_kana.json",
    "item_as_entity_domains": "item_as_entity_domains.json",
    "query_target_keywords": "query_target_keywords.json",
    "query_item_keywords": "query_item_keywords.json",
    "query_entity_keywords": "query_entity_keywords.json",
    "query_category_group_keywords": "query_category_group_keywords.json",
}


PAREN_PATTERN = re.compile(r"\(([^)]*)\)")
PUNCT_ONLY_PATTERN = re.compile(r"^[\W_]+$")
SPLIT_PATTERN = re.compile(r"[／/・,、]")


SIZE_KEY_LETTERS = {
    "XS",
    "S",
    "M",
    "L",
    "LL",
    "XL",
    "XXL",
    "XXXL",
    "FREE",
    "F",
    "SS",
    "MB",
    "WINNER",
}


SIZE_KEYWORDS_SUBSTR = (
    "cm",
    "号",
    "インチ",
    "サイズ",
    "カップ",
    "段",
    "人用",
    "mm",
    "畳",
    "坪",
    "ml",
    "l",
)


SIZE_KEYWORDS_WORDS = {
    "フリー",
    "フリーサイズ",
    "シングル",
    "ダブル",
    "セミダブル",
    "クイーン",
    "キング",
    "ベビー",
    "ベッド",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
    normalized = normalized.replace("　", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def split_token(token: str) -> Set[str]:
    token = normalize_text(token)
    if not token:
        return set()

    variants: Set[str] = set()

    # base without parentheses
    base = PAREN_PATTERN.sub("", token).strip()
    if base:
        variants.add(base)

    # contents inside parentheses
    for inner in PAREN_PATTERN.findall(token):
        inner_norm = normalize_text(inner)
        if inner_norm:
            variants.add(inner_norm)

    if not variants:
        variants.add(token)

    final_tokens: Set[str] = set()
    for variant in variants:
        parts = [normalize_text(part) for part in SPLIT_PATTERN.split(variant)]
        parts = [part for part in parts if part]
        if len(parts) > 1:
            final_tokens.update(parts)
        else:
            final_tokens.update(parts or [variant])

    cleaned_tokens: Set[str] = set()
    for token in final_tokens:
        cleaned = normalize_text(token)
        if cleaned:
            cleaned_tokens.add(cleaned)

    return cleaned_tokens


def should_keep(key: str, token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if PUNCT_ONLY_PATTERN.match(token):
        return False

    if key == "size_keywords":
        if re.search(r"\d", token):
            return True
        upper = token.upper()
        if upper in SIZE_KEY_LETTERS:
            return True
        if token in SIZE_KEYWORDS_WORDS:
            return True
        if any(sub in token for sub in SIZE_KEYWORDS_SUBSTR):
            return True
        return False

    return True


def load_candidates(directory: Path) -> List[Path]:
    return sorted(directory.glob("batch_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_DIR,
        help="候補語 JSON が入ったディレクトリ",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="実際に辞書を更新する（指定しない場合はドライラン）",
    )

    args = parser.parse_args()

    candidate_dir = args.candidates_dir
    if not candidate_dir.exists():
        raise FileNotFoundError(f"候補ディレクトリが見つかりません: {candidate_dir}")

    candidate_files = load_candidates(candidate_dir)
    if not candidate_files:
        print("候補ファイルが見つかりません。")
        return

    additions: Dict[str, Set[str]] = defaultdict(set)

    for path in candidate_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        candidates = data.get("candidates", {})
        if not isinstance(candidates, dict):
            continue

        for key, values in candidates.items():
            if key not in KEY_TO_FILENAME:
                continue
            if not isinstance(values, list):
                continue

            for value in values:
                if not isinstance(value, str):
                    continue
                for token in split_token(value):
                    if token and should_keep(key, token):
                        additions[key].add(token)

    if not additions:
        print("追加候補はありませんでした。")
        return

    print("候補抽出結果:")
    for key in KEY_TO_FILENAME:
        words = sorted(additions.get(key, []))
        print(f"  {key}: {len(words)} 件")
        for word in words:
            print(f"    - {word}")

    if not args.apply:
        print("--apply を指定すると辞書ファイルを更新します。")
        return

    for key, filename in KEY_TO_FILENAME.items():
        words = additions.get(key)
        if not words:
            continue

        existing = set(dict_utils.load_set(filename))
        new_words = sorted(existing.union(words))

        target_path = PROJECT_ROOT / "dictionary" / filename
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(new_words, f, ensure_ascii=False, indent=2)

        print(f"更新: {filename} (追加 {len(words - existing)} 件)")


if __name__ == "__main__":
    main()


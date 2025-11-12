"""カテゴリーパスを属性ごとのJSONに変換するスクリプト"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from dictionary.utils import load_list, load_set


def load_final_categories(file_path: Path) -> list[tuple[str, str]]:
    categories: list[tuple[str, str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            cat_id = parts[0].strip()
            path = parts[1].strip()
            if cat_id and path:
                categories.append((cat_id, path))
    return categories


TARGET_KEYWORDS = load_set("target_keywords.json")
SIZE_KEYWORDS = load_set("size_keywords.json")

SIZE_RANGE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)(?:cm|号)$", re.IGNORECASE)
ALPHA_SIZE_PATTERN = re.compile(r"^(?:[X]{0,3}[SML]|XXL|XXXL)(?:サイズ.*)?$", re.IGNORECASE)

ENTITY_DOMAINS = load_set("entity_domains.json")
DOMAINS_WITH_KANA = load_set("domains_with_kana.json")
ITEM_AS_ENTITY_DOMAINS = load_set("item_as_entity_domains.json")
KANA_GROUPS = set(load_list("kana_groups.json"))
CATEGORY_GROUP_FIRST_DOMAINS = {"ブランド"}


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def normalize_path(path: str) -> dict:
    segments = [seg.strip() for seg in path.split(" > ") if seg.strip()]
    normalized_segments = [normalize_text(seg) for seg in segments]

    result: dict[str, str | list[str]] = {
        "category_id": "",
        "top_level": "",
        "domain": "",
        "kana_group": "",
        "entity": "",
        "category_group": "",
        "subcategory": "",
        "target": "",
        "item": "",
        "size": "",
        "extra": [],
        "raw_path": path,
    }

    if not segments:
        return result

    result["top_level"] = segments[0]

    idx = 1
    if idx < len(segments):
        result["domain"] = segments[idx]
        idx += 1

    domain_norm = normalize_text(result["domain"]) if result["domain"] else ""

    if domain_norm == "アニメ/コミック/キャラクター":
        if idx < len(segments) and not result["category_group"]:
            result["category_group"] = segments[idx]
            idx += 1

        if idx < len(segments):
            norm = normalized_segments[idx]
            if norm in KANA_GROUPS and len(norm) == 1:
                result["kana_group"] = segments[idx]
                idx += 1

        if idx < len(segments):
            result["entity"] = segments[idx]
            idx += 1
    else:
        if (
            domain_norm in CATEGORY_GROUP_FIRST_DOMAINS
            and idx < len(segments)
        ):
            candidate = normalized_segments[idx]
            if candidate not in KANA_GROUPS or len(candidate) > 1:
                result["category_group"] = segments[idx]
                idx += 1

        if idx < len(segments):
            norm = normalized_segments[idx]
            if domain_norm in DOMAINS_WITH_KANA and norm in KANA_GROUPS and len(norm) == 1:
                result["kana_group"] = segments[idx]
                idx += 1

        if idx < len(segments) and domain_norm in ENTITY_DOMAINS:
            result["entity"] = segments[idx]
            idx += 1

    remaining_labels: list[str] = []

    for seg, norm in zip(segments[idx:], normalized_segments[idx:]):
        if not result["target"] and norm in TARGET_KEYWORDS:
            result["target"] = seg
            continue

        if not result["size"]:
            if SIZE_RANGE_PATTERN.match(norm):
                result["size"] = seg
                continue

            upper_norm = norm.upper()
            if any(keyword in norm for keyword in SIZE_KEYWORDS) or ALPHA_SIZE_PATTERN.match(upper_norm):
                result["size"] = seg
                continue

        remaining_labels.append(seg)

    labels_for_structure = remaining_labels
    if result["entity"] and labels_for_structure and labels_for_structure[0] == result["entity"]:
        labels_for_structure = labels_for_structure[1:]

    consumed_indices: set[int] = set()

    if labels_for_structure and not result["category_group"]:
        result["category_group"] = labels_for_structure[0]
        consumed_indices.add(0)

    remaining_indices = [i for i in range(len(labels_for_structure)) if i not in consumed_indices]

    if len(remaining_indices) >= 2:
        first_idx = remaining_indices[0]
        result["subcategory"] = labels_for_structure[first_idx]
        consumed_indices.add(first_idx)
        remaining_indices = [i for i in range(len(labels_for_structure)) if i not in consumed_indices]

    remaining_indices = [i for i in range(len(labels_for_structure)) if i not in consumed_indices]

    if remaining_indices:
        last_idx = remaining_indices[-1]
        result["item"] = labels_for_structure[last_idx]
        consumed_indices.add(last_idx)

    if not labels_for_structure and not result["item"] and result["entity"]:
        result["item"] = result["entity"]

    if (not result["entity"] and result["item"]
            and domain_norm in ITEM_AS_ENTITY_DOMAINS):
        result["entity"] = result["item"]

    for idx_label, seg in enumerate(labels_for_structure):
        if idx_label in consumed_indices:
            continue
        result["extra"].append(seg)

    return result


def transform_to_json(categories: list[tuple[str, str]], output_path: Path) -> None:
    transformed = []
    for cat_id, path in categories:
        normalized = normalize_path(path)
        normalized["category_id"] = cat_id
        transformed.append(normalized)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)


def main() -> None:
    base_dir = Path(__file__).parent
    final_categories_file = base_dir / "final_categories.txt"
    output_file = base_dir / "final_categories.json"

    categories = load_final_categories(final_categories_file)
    transform_to_json(categories, output_file)
    print(f"変換完了: {len(categories)} 件 -> {output_file}")


if __name__ == "__main__":
    main()



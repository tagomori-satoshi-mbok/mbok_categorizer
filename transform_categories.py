"""
カテゴリーファイルを階層構造に変換するスクリプト

1. 最終カテゴリー（他の行にIDが出現しないもの）のみを抽出
2. 各最終カテゴリーについて、親カテゴリーの階層をたどって
   「親カテゴリー名 > 親カテゴリー名 > ... > 最終カテゴリー名」の形式に変換
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def load_categories(file_path: str) -> tuple[dict, dict]:
    """
    categoriesファイルを読み込み、階層情報を抽出
    
    Returns:
        categories: {
            category_id: {
                'name': category_name,
                'path_ids': [ancestor_id1, ancestor_id2, ..., category_id]
            }
        }
        id_to_name: {category_id: category_name}
    """

    categories: dict[str, dict] = {}
    id_to_name: dict[str, str] = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split('\t')]
            if len(parts) < 2:
                continue

            category_id = parts[0]
            category_name = parts[1]

            if not category_id or not category_name:
                continue

            id_to_name[category_id] = category_name

            # categoriesファイルでは「80」以降が実際の階層パスになる
            # 例: ... 0 0 80 100000001 100000129 100000867 100000973 100000996 100000997 0 1
            path_ids: list[str] = []
            path_started = False

            for value in parts[2:]:
                if not value:
                    continue

                if not path_started:
                    if value == '80':
                        path_started = True
                    # 80 をパスに含めるかどうかはここで決定する
                    # （80 自体はIDを持っておりカテゴリ名称に変換可能）
                        path_ids.append(value)
                    continue

                if value in {'0', '1'}:
                    break

                path_ids.append(value)

            # 80 が見つからないケース（稀）でも最低限自分自身をパスに含める
            if not path_ids:
                path_ids = [category_id]

            # パスの末尾に自分自身が含まれていなければ追加
            if path_ids[-1] != category_id:
                path_ids.append(category_id)

            categories[category_id] = {
                'name': category_name,
                'path_ids': path_ids,
            }

    return categories, id_to_name


def find_final_categories(categories: dict) -> list[str]:
    """
    最終カテゴリー（他の行の親IDリストに出現しないもの）を抽出
    
    Returns:
        最終カテゴリーIDのリスト
    """
    # 他の行の親IDリストに出現するIDを収集
    # あるカテゴリーIDが、自分以外の他のカテゴリーの親IDリストに含まれているかをチェック
    parent_ids = set()
    for cat_data in categories.values():
        path_ids = cat_data.get('path_ids', [])
        # 最後の要素は自身なので除外して親セットに追加
        parent_ids.update(path_ids[:-1])
    
    print(f"親IDとして出現するID数: {len(parent_ids):,}")
    
    # 他の行の親IDリストに出現しないID = 最終カテゴリー
    final_categories = []
    for cat_id in categories.keys():
        if cat_id not in parent_ids:
            final_categories.append(cat_id)
    
    print(f"最終カテゴリー数: {len(final_categories):,}")
    print(f"除外されたカテゴリー数: {len(categories) - len(final_categories):,}")
    
    # 検証: 最終カテゴリーに含まれるIDが、他の行の親IDリストに含まれていないか再確認
    verification_failed = []
    for cat_id in final_categories[:100]:  # 最初の100件を検証
        for other_cat_id, other_cat_data in categories.items():
            if other_cat_id == cat_id:
                continue
            other_path = other_cat_data.get('path_ids', [])
            if cat_id in other_path[:-1]:
                verification_failed.append((cat_id, other_cat_id))
    
    if verification_failed:
        print(f"警告: 検証で {len(verification_failed)} 件の問題が見つかりました")
        for cat_id, other_cat_id in verification_failed[:5]:
            print(f"  {cat_id} が {other_cat_id} の親IDリストに含まれています")
    else:
        print("検証: 最終カテゴリーは正しく抽出されています")
    
    return final_categories


def build_category_path(category_id: str, categories: dict, id_to_name: dict) -> str:
    """
    カテゴリーの階層パスを構築
    
    Args:
        category_id: カテゴリーID
        categories: カテゴリーデータの辞書
        id_to_name: IDから名称へのマッピング
        
    Returns:
        「親カテゴリー名 > 親カテゴリー名 > ... > 最終カテゴリー名」の形式
    """

    if category_id not in categories:
        return ""

    path_ids = categories[category_id].get('path_ids', [])
    if not path_ids:
        return id_to_name.get(category_id, category_id)

    path_parts = []
    seen_ids = set()

    for path_id in path_ids:
        if path_id in seen_ids:
            continue
        name = id_to_name.get(path_id, path_id)
        if name:
            path_parts.append(name)
            seen_ids.add(path_id)

    return " > ".join(path_parts)


def transform_categories(categories_file: str, output_file: str):
    """
    カテゴリーファイルを変換
    """
    print(f"カテゴリーファイルを読み込み中: {categories_file}")
    categories, id_to_name = load_categories(categories_file)
    print(f"読み込み完了: {len(categories):,} カテゴリー")
    
    print("\n最終カテゴリーを抽出中...")
    final_category_ids = find_final_categories(categories)
    print(f"最終カテゴリー数: {len(final_category_ids):,}")
    
    print(f"\n階層構造に変換中...")
    transformed_categories = []
    
    for i, cat_id in enumerate(final_category_ids, 1):
        if i % 1000 == 0:
            print(f"  進捗: {i:,}/{len(final_category_ids):,} ({i*100/len(final_category_ids):.1f}%)")
        
        category_path = build_category_path(cat_id, categories, id_to_name)
        if category_path:
            transformed_categories.append({
                'category_id': cat_id,
                'category_path': category_path
            })
    
    print(f"\n変換完了: {len(transformed_categories):,} 件")
    
    # 出力ファイルに保存
    print(f"\n結果を保存中: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in transformed_categories:
            f.write(f"{item['category_id']}\t{item['category_path']}\n")
    
    print(f"✓ 保存完了: {output_file}")
    
    # サンプルを表示
    print("\n変換サンプル（最初の5件）:")
    for item in transformed_categories[:5]:
        print(f"  ID: {item['category_id']}")
        print(f"  パス: {item['category_path']}")
        print()


if __name__ == "__main__":
    default_input = Path(__file__).parent / "categories"
    default_output = Path(__file__).parent / "final_categories.txt"

    parser = argparse.ArgumentParser(description="categories ファイルを階層パスに変換します。")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="入力となる categories ファイル（タブ区切り）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="出力する final_categories.txt のパス",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"エラー: {args.input} が見つかりません。")
        sys.exit(1)

    transform_categories(str(args.input), str(args.output))


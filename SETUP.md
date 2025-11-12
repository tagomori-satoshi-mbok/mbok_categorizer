# Categorizer Pipeline Setup

このドキュメントは新しい環境（例: GCP 上のコンテナ）でカテゴリ処理パイプラインを実行するための手順をまとめたものです。

## 1. 必要ファイルの配置
- `categories` (タブ区切りの元データ) を `/app` などコンテナ内のワークディレクトリに配置します。  
- `dictionary/` 配下には、既存の辞書 JSON (`target_keywords.json` など) が含まれています。

## 2. Python スクリプトの概要
- `transform_categories.py`: `categories` を読み込んで `final_categories.txt` を生成します。  
  - `--input` オプションで入力ファイル、`--output` で出力ファイルを指定できます。
- `categories_to_json.py`: `final_categories.txt` から `final_categories.json` に変換します。  
  - `--input` / `--output` オプションあり。
- `generate_dictionary_candidates.py`: Gemini API を使用して辞書候補を生成します。  
  - `--output-dir` で出力先を指定。`dictionary/candidates_full` を推奨。  
  - `--offset`, `--limit`, `--batch-size`, `--sleep` などで制御可能。  
  - `GEMINI_API_KEY` 環境変数が必須。
- `process_dictionary_candidates.py`: 候補を既存辞書にマージします。  
  - `--apply` を付けないとドライランになります。
- `gemini_attribute_embeddings.py`: 属性別に埋め込みを生成し、JSON Lines 形式で保存します。

## 3. 基本的な実行フロー
1. `python transform_categories.py --input categories --output final_categories.txt`
2. `python categories_to_json.py --input final_categories.txt --output final_categories.json`
3. `python generate_dictionary_candidates.py --input final_categories.json --output-dir dictionary/candidates_full`
4. `python process_dictionary_candidates.py --candidates-dir dictionary/candidates_full --apply`
5. `python gemini_attribute_embeddings.py --input final_categories.json --output final_categories_embeddings.jsonl`

## 4. Docker での利用
- `Dockerfile` は Python 3.11 slim ベースです。  
- ビルド: `docker build -t categorizer .`
- 実行時に `categories` や API キーを渡す例:
  ```bash
  docker run --rm -it \
    -v /path/to/categories:/app/categories \
    -e GEMINI_API_KEY=YOUR_KEY \
    categorizer bash
  ```

## 5. 注意事項
- Gemini API へのアクセスにはネットワークが必要です。  
- 辞書 JSON は NFKC 正規化を前提としています。  
- 途中でタイムアウトが発生した場合は、`--offset` などを使って再実行してください。


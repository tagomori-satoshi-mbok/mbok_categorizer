# Categorizer Pipeline

このプロジェクトは、`categories` ファイルを入力として以下の処理を実行できるように構成されています。

1. **final_categories.txt の生成**  
   - `transform_categories.py` を用いて、タブ区切りの `categories` ファイルを階層パス付きの `final_categories.txt` に変換します。  
   - 実行例:  
     ```bash
     python transform_categories.py --input categories --output final_categories.txt
     ```

2. **final_categories.json の生成**  
   - `categories_to_json.py` を用いて、`final_categories.txt` から属性を抽出した JSON (`final_categories.json`) を生成します。  
   - 実行例:  
     ```bash
     python categories_to_json.py --input final_categories.txt --output final_categories.json
     ```

3. **辞書候補の生成**  
   - `generate_dictionary_candidates.py` を用いて、Gemini API から辞書候補を取得します。結果は `dictionary/candidates` または `dictionary/candidates_full` に保存されます。  
   - 実行例:  
     ```bash
     export GEMINI_API_KEY=...
     python generate_dictionary_candidates.py \
       --input final_categories.json \
       --output-dir dictionary/candidates_full \
       --batch-size 100 \
       --sleep 0.8 \
       --offset 0
     ```

4. **辞書の更新**  
   - `process_dictionary_candidates.py` を用いて、候補を既存辞書 (`dictionary/*.json`) にマージします。  
   - 実行例:  
     ```bash
     python process_dictionary_candidates.py --candidates-dir dictionary/candidates_full --apply
     ```

5. **埋め込みの生成**  
   - `gemini_attribute_embeddings.py` を用いて、属性毎に Gemini Embedding API でベクトル化を行い、JSON Lines 形式で出力します。  
   - 実行例:  
     ```bash
     export GEMINI_API_KEY=...
     python gemini_attribute_embeddings.py \
       --input final_categories.json \
       --output final_categories_embeddings.jsonl \
       --attributes entity item target \
       --batch-size 16 --sleep 0.6
     ```

## 必要ファイル
- `categories`: もともとのカテゴリデータ (タブ区切り)  
- `dictionary/*.json`: 辞書ファイル群  

## Docker での利用
プロジェクトには Dockerfile が同梱されています。GCP などでの利用を想定して、Docker イメージをビルド・実行できます。
```bash
cd /Users/satoshi/categorizer
docker build -t categorizer .
```
実行時には `categories` ファイルや API キーをボリュームや環境変数で渡してください。

### docker-compose でのローカル実行

```bash
cd /Users/satoshi/categorizer
docker-compose up --build api
```

ブラウザまたは `curl http://127.0.0.1:8001/healthz` で疎通を確認できます。必要な環境変数は `.env` で管理するか、`docker-compose.yml` の `environment` セクションに直接指定してください。

### Cloud Run へのデプロイ例

1. **コンテナのビルドとプッシュ**
   ```bash
   PROJECT_ID=your-gcp-project
   REGION=asia-northeast1  # 任意
   IMAGE_NAME=categorizer-api

   gcloud builds submit --project "${PROJECT_ID}" --tag "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" .
   ```

2. **Cloud Run サービスの作成/更新**
   ```bash
   gcloud run deploy categorizer-api \
     --project "${PROJECT_ID}" \
     --image "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
     --region "${REGION}" \
     --allow-unauthenticated \
     --set-env-vars "GEMINI_API_KEY=YOUR_KEY" \
     --set-env-vars "EMBEDDINGS_PATH=/app/final_categories_embeddings.jsonl" \
     --set-env-vars "CATEGORIES_PATH=/app/final_categories.json"
   ```

   ※ `GEMINI_API_KEY` など機密情報は、`--set-secrets` や Secret Manager の利用を推奨します。

3. **動作確認**
   デプロイが完了したら、発行された URL に対し `GET /healthz` または `POST /search` を実行して動作を確認します。

Cloud Run ではデフォルトで `PORT` 環境変数が 8080 に設定されます。本プロジェクトの Docker イメージは `PORT` を参照して `uvicorn` を起動するため、追加設定は不要です。


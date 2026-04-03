import json
import os
from typing import List, Dict, Any

import boto3
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#S3 + OpenSearch config.
BUCKET = "arxiv-s3"
PREFIX = "processed/arxiv_oai/records/"  # Folder containing processed JSON files.
INDEX_NAME = "arxiv-papers"  # Target OpenSearch index.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model.

#OpenSearch connection settings.
HOST = "localhost"
PORT = 9200
USERNAME = "admin"
PASSWORD = "[PASSWORD PLACEHOLDER]"

#Local file to track last processed S3 file (for incremental indexing).
STATE_FILE = "/home/ubuntu/arxiv-opensearch/index_state.json"

#Initialize S3 client.
s3 = boto3.client("s3")

#Initailize OpenSearch client.
client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

#Load embedding model once.
model = SentenceTransformer(MODEL_NAME)


def load_state():
    """
    Load the last processed S3 key from local state file.
    If file does not exists, start from beginning.
    """
    if not os.path.exists(STATE_FILE):
        return {"last_processed_key": ""}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(last_key: str):
    """
    Save the latest processed S3 key for future runs.
    """
    with open(STATE_FILE, "w") as f:
        json.dump({"last_processed_key": last_key}, f)


def list_new_keys(last_key: str) -> List[str]:
    """
    List only new JSON files in S3 (keys greater than last processed key).
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    #Scan all objects under prefix.
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue  # Skip non-JSON files.
            if key > last_key:
                keys.append(key)  # Only include new files.

    keys.sort()
    return keys


def load_records(key: str) -> List[Dict[str, Any]]:
    """
    Load JSON records from a given S3 file.
    """
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def build_text(record: Dict[str, Any]) -> str:
    """
    Combine title and abstract into one string for embedding.
    """
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    return f"{title}. {abstract}".strip()


def main():
    """
    Incrementally index new S3 data into OpenSearch.
    Only processes files not previously indexed.
    """
    #Load last processed state.
    state = load_state()
    last_key = state["last_processed_key"]

    #Get only new files.
    new_keys = list_new_keys(last_key)

    if not new_keys:
        print("No new processed JSON files found.")
        return

    print(f"Found {len(new_keys)} new files.")

    latest_key = last_key
    total_indexed = 0

    #Process each new file.
    for key in tqdm(new_keys, desc="New files"):
        records = load_records(key)

        #Prepare text for embedding.
        texts = [build_text(r) for r in records]

        #Generate embeddings for all records.
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        #Build OpenSearch bulk actions.
        actions = []
        for record, emb in zip(records, embeddings):
            arxiv_id = record.get("arxiv_id")
            if not arxiv_id:
                continue  # Skip invalid records.

            actions.append(
                {
                    "_index": INDEX_NAME,
                    "_id": arxiv_id,  # Prevent duplicates (same ID overwrites).
                    "_source": {
                        "arxiv_id": arxiv_id,
                        "title": record.get("title"),
                        "abstract": record.get("abstract"),
                        "link": record.get("link"),
                        "created": record.get("created"),
                        "categories": record.get("categories", []),
                        "combined_text": build_text(record),
                        "embedding": emb.tolist(),
                    },
                }
            )

        #Bulk index into OpenSearch.
        success, failed = helpers.bulk(
            client,
            actions,
            chunk_size=500,
            stats_only=True,
        )

        total_indexed += success
        latest_key = key  # Update checkpoint.

        print(f"Indexed {success} docs from {key}")

    #Save latest processed key.
    save_state(latest_key)

    #Final summary.
    print(f"\nDone. Total indexed: {total_indexed}")
    print(f"Updated state file to: {latest_key}")


if __name__ == "__main__":
    main()

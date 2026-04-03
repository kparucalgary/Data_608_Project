import json
from typing import Iterator, List, Dict, Any

import boto3
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#S3 and OpenSearch config.
BUCKET = "arxiv-s3"   
PREFIX = "processed/arxiv_oai/records/" #Folder containing processed JSON files in S3.
INDEX_NAME = "arxiv-papers" #Target OpenSearch index.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #Embedding model.

#OpenSearch connection settings.
HOST = "localhost"
PORT = 9200
USERNAME = "admin"
PASSWORD = "[PASSWORD PLACEHOLDER]"

#Processing settings.
BATCH_SIZE = 64 #Number of text to embed at once.
BULK_CHUNK_SIZE = 500 #Number of docs per OpenSearch bullk request.
STATE_FILE = "index_state.json" #Local file to track the latest processed S3 key.

#Initialize S3 client.
s3 = boto3.client("s3")

#Initialize OpenSearch client.
client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

#Load embedding model once at startup.
model = SentenceTransformer(MODEL_NAME)


def list_all_processed_keys(bucket: str, prefix: str) -> List[str]:
	"""
	Return all JSON file keys under the given S3 prefix, sorted in order.
	"""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

	#Paginate through all matching S3 objects.
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"): #Only keep JSON files.
                keys.append(key)

    keys.sort()
    return keys


def load_records(bucket: str, key: str) -> List[Dict[str, Any]]:
	"""
	Load and parse one JSON file from S3 into a list of records.
	"""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def build_text(record: Dict[str, Any]) -> str:
	"""
	Combine title and abstract into one text string for embedding.
	"""
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    return f"{title}. {abstract}".strip()


def chunk_list(items: List[Any], size: int) -> Iterator[List[Any]]:
	"""
	Yield smaller chunks from a list.
	"""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_actions(records: List[Dict[str, Any]], embeddings) -> List[Dict[str, Any]]:
	"""
	Build OpenSearch bulk indexing actions from records and their embeddings.
	"""
    actions = []
    for record, emb in zip(records, embeddings):
        arxiv_id = record.get("arxiv_id")
        if not arxiv_id:
            continue #Skip records w/ no unique ID.

        actions.append({
            "_index": INDEX_NAME,
            "_id": arxiv_id, #Use arXiv ID as the document ID.
            "_source": {
                "arxiv_id": arxiv_id,
                "title": record.get("title"),
                "abstract": record.get("abstract"),
                "link": record.get("link"),
                "created": record.get("created"),
                "categories": record.get("categories", []),
                "combined_text": build_text(record),
                "embedding": emb.tolist(), #Convert NumPy vector to regular list.
            }
        })
    return actions


def save_state(last_key: str):
	"""
	Save the most recently processed S3 key to a local state file.
	"""
    with open(STATE_FILE, "w") as f:
        json.dump({"last_processed_key": last_key}, f)


def main():
	"""
	Backfill the OpenSearch index from all processed JSON files in S3.
	"""
    keys = list_all_processed_keys(BUCKET, PREFIX)
    if not keys:
        raise RuntimeError(f"No processed JSON files found under s3://{BUCKET}/{PREFIX}")

    print(f"Found {len(keys)} processed JSON files")

	#Running totals for reporting.
    total_files = 0
    total_records_loaded = 0
    total_indexed = 0
    total_failed = 0

	#Process each JSON file from S3.
    for key in tqdm(keys, desc="Files"):
        records = load_records(BUCKET, key)
        if not records:
            continue #Skip emtpy files.

        total_files += 1
        total_records_loaded += len(records)

		#Prepare text input for embedding model.
        texts = [build_text(r) for r in records]

		#Generate embeddings for all records in the file.
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

		#Convert records + embeddings into OpenSearch bulk actions.
        actions = build_actions(records, embeddings)

		#Send docs to OpenSearch using bulk indexing.
        success, failed = helpers.bulk(
            client,
            actions,
            chunk_size=BULK_CHUNK_SIZE,
            stats_only=True,
            request_timeout=120
        )

        total_indexed += success
        total_failed += failed

		#Print per-file indexing summary.
        print(
            f"Indexed file: s3://{BUCKET}/{key} | "
            f"records={len(records)} | indexed={success} | failed={failed}"
        )

	#Save the latest processed key after all files are done.
    latest_key = keys[-1]
    save_state(latest_key)

	#Final summary.
    print("\nDone.")
    print(f"Files processed: {total_files}")
    print(f"Records loaded: {total_records_loaded}")
    print(f"Indexed: {total_indexed}")
    print(f"Failed: {total_failed}")
    print(f"State file updated with latest key: {latest_key}")


if __name__ == "__main__":
    main()

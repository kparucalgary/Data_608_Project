import json
from datetime import datetime, timezone

import boto3
import streamlit as st

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


# --- S3 Configuration ---
BUCKET_NAME = "data608-arxiv-logs-s3"
QUERY_LOG_PREFIX = "query-logs/"

def log_query_to_s3(query: str) -> None:
    s3 = boto3.client("s3")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") # Zulu(UTC) timestamp

    payload = {
        "query": query,
        "timestamp": timestamp
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{QUERY_LOG_PREFIX}query_{timestamp}.json",
        Body=json.dumps(payload, indent=2), # Indent 2 spaces for readability
        ContentType="application/json" # tells AWS S3 that the body is a JSON object
    )



# --- OpenSearch Configuration ---
HOST = "localhost"
PORT = 9200
USERNAME = "admin"
PASSWORD = "data608ML!!!"

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

model = SentenceTransformer(MODEL_NAME)


def run_search(query: str, top_k: int = 5):
    # Step 1: embed query
    query_vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    # Step 2: search OpenSearch
    body = {
        "size": top_k,
        "_source": ["title", "abstract", "link"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=body)

    # Step 3: format results for Streamlit
    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]

        results.append({
            "title": src.get("title"),
            "authors": [],  # optional for now
            "abstract": src.get("abstract"),
            "url": src.get("link"),
        })

    return results




# --- Streamlit UI ---
st.set_page_config(page_title="arXiv Semantic Search", layout="wide")

st.title("arXiv Semantic Search Assistant")
st.write("Enter a research topic to search for semantically relevant papers.")


# User Input
query = st.text_input("Search query")

if st.button("Search"):
    if query.strip():

        # Log the query to S3
        log_query_to_s3(query)

        # Run the search
        results = run_search(query)

        # Display the results   
        st.subheader("Results")
        for i, paper in enumerate(results, start=1):
            st.markdown(f"### {i}. {paper['title']}")
            st.write("**Authors:**", ", ".join(paper["authors"]))
            st.write("**Abstract:**", paper["abstract"])
            st.markdown(f"[Open paper]({paper['url']})")
            st.markdown("---")
    else:
        st.warning("Please enter a query.")

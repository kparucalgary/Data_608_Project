import json
from datetime import datetime, timezone

import boto3
import streamlit as st

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from langgraph_structure import run_pipeline


# --- S3 Configuration ---
BUCKET_NAME = "data608-arxiv-logs-s3"
QUERY_LOG_PREFIX = "query-logs/"

def log_query_to_s3(query: str, result_count: int, threshold: float) -> None:
    s3 = boto3.client("s3")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") # Zulu(UTC) timestamp

    payload = {
        "query": query,
        "timestamp": timestamp,
        "result_count": result_count,
        "threshold": threshold
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



# --- Streamlit UI ---
st.set_page_config(page_title="arXiv Semantic Search", layout="wide")

st.title("arXiv Semantic Search Assistant")
st.write("Enter a research topic to search for semantically relevant papers.")

with st.form("search_form"):
    query = st.text_input("Search query")

    result_count = st.selectbox(
        "Number of results",
        options=[5, 10, 20, 30],
        index=0
    )
    threshold = st.selectbox(
        "Similarity Threshold",
        options=[0.5, 0.6, 0.7, 0.8, 0.9],
        index=0
    )

    submitted = st.form_submit_button("Search")

if submitted:
    if query.strip():

        # Log the query to S3
        log_query_to_s3(query, result_count, threshold)

        # Run the search
        results = run_pipeline(mode='llm', query=query, numpapers=result_count, threshold=threshold)
        #results = run_search(query, top_k=result_count)

        # Display the results
        st.subheader("Results")
        for i, paper in enumerate(results, start=1):
            st.markdown(f"### {i}. {paper['title']}")
            st.write("**Abstract:**", paper["abstract"])
            st.markdown(f"[Open paper]({paper['url']})")
            st.markdown("---")
    else:
        st.warning("Please enter a query.")

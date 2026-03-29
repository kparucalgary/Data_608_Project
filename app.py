import json
from datetime import datetime, timezone

import boto3
import streamlit as st


# --- S3 Configuration ---
BUCKET_NAME = "data608-arxiv-logs-s3"
QUERY_LOG_PREFIX = "query-logs/"

def log_query_to_s3(query: str) -> None:
    s3 = boto3.client("s3")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    payload = {
        "query": query,
        "timestamp": timestamp
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{QUERY_LOG_PREFIX}query_{timestamp}.json",
        Body=json.dumps(payload, indent=2),
        ContentType="application/json"
    )

'''
Currently a placeholder function for the search functionality

Eventual Replacement:

1. 'sentence-transformers': Use a model to convert 'query' into a vector embedding.
2. 'OpenSearch': Send that vector to your OpenSearch cluster to 'Retrieve Relevant Papers'.
3. Return the actual metadata (title, URL, abstract) stored in your OpenSearch index.
'''
def run_search(query: str):
    return [
        {
            "title": "Placeholder Paper 1",
            "authors": ["Author A", "Author B"],
            "abstract": f"This is a placeholder result for query: {query}",
            "url": "https://arxiv.org"
        },
        {
            "title": "Placeholder Paper 2",
            "authors": ["Author C"],
            "abstract": "Another placeholder abstract until backend integration is ready.",
            "url": "https://arxiv.org"
        }
    ]





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

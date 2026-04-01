from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# -------------------------------
# OpenSearch connection settings
# -------------------------------
HOST = "localhost"
PORT = 9200
USERNAME = "admin"
PASSWORD = "data608ML!!!"

# -------------------------------
# Index + model config
# -------------------------------
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3  # Number of top results to return

# -------------------------------
# Initialize OpenSearch client
# -------------------------------
client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    timeout=60,  # Timeout for connection
)

# -------------------------------
# Load embedding model
# -------------------------------
# Used to convert query text into vectors
model = SentenceTransformer(MODEL_NAME)


def semantic_search(query_text: str, top_k: int = TOP_K):
    """
    Perform semantic (vector-based) search on OpenSearch.
    """

    # Convert user query into embedding vector
    query_vector = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    # Define OpenSearch k-NN query
    body = {
        "size": top_k,  # Number of results to return
        "_source": [  # Fields to include in response
            "arxiv_id",
            "title",
            "abstract",
            "link",
            "created",
            "categories",
        ],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,  # Query embedding
                    "k": top_k,  # Number of nearest neighbors
                }
            }
        },
    }

    # Execute search request
    response = client.search(
        index=INDEX_NAME,
        body=body,
        request_timeout=60,
    )

    # Extract results and attach similarity score
    results = []
    for hit in response["hits"]["hits"]:
        result = hit["_source"].copy()  # Extract document fields
        result["score"] = hit["_score"]  # Add similarity score
        results.append(result)

    return results


def main():
    """
    CLI interface for semantic search.
    """

    # Get user input query
    query = input("Enter your search query: ").strip()

    if not query:
        print("No query entered.")
        return

    # Perform semantic search
    results = semantic_search(query)

    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} results for: {query}\n")

    # Display results
    for i, hit in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Score: {hit.get('score')}")
        print(f"arXiv ID: {hit.get('arxiv_id')}")
        print(f"Title: {hit.get('title')}")
        print(f"Created: {hit.get('created')}")
        print(f"Categories: {hit.get('categories')}")
        print(f"Link: {hit.get('link')}")

        # Clean and truncate abstract for readability
        abstract = (hit.get("abstract") or "").replace("\n", " ")
        print(f"Abstract: {abstract[:500]}...")
        print()


if __name__ == "__main__":
    main()

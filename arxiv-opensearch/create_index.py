from opensearchpy import OpenSearch

#Connection config for the OpenSearch cluster.
HOST = "localhost"  # OpenSearch host - we run locally on EC2.
PORT = 9200  # Default OpenSearch port.
USERNAME = "admin"  # Admin username (from the initial setup).
PASSWORD = "[PASSWORD PLACEHOLDER]"  # Admin password (from the initial setup).
INDEX_NAME = "arxiv-papers"  # Name of the index to create.

#Initialize OpenSearch client w/ authentication and SSL settings.
client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=(USERNAME, PASSWORD),  # Basic authentication.
    use_ssl=True,  # Use HTTPS connection.
    verify_certs=False,  # Disable cert verficiation.
    ssl_assert_hostname=False,  # Disable hostname verification.
    ssl_show_warn=False,  # Suppress SSL warnings.
)

#Define index settings and field mappings.
mapping = {
    "settings": {
        "index": {
            "knn": True  # Enable k-NN vector search capability.
        }
    },
    "mappings": {
        "properties": {
            "arxiv_id": {"type": "keyword"},  # Unique ID (exact match)
            "title": {"type": "text"},  # Full-text searchable field.
            "abstract": {"type": "text"},  # Full-text searchable field.
            "link": {"type": "keyword"},  # URL (exact match).
            "created": {
                "type": "date",
                "format": "strict_date_optional_time||yyyy-MM-dd",
            },  # Publication date w/ flexible format.
            "categories": {"type": "keyword"},  # List of categories (filtering).
            "combined_text": {"type": "text"},  # Title + abstract for embedding input.
            "embedding": {
                "type": "knn_vector",  # Vector field for semantic search.
                "dimension": 384,  # Matches the embedding model output size.
            },
        }
    },
}

#If the index already exists, delete it (fresh build).
if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)

#Create the index with the defined mapping.
client.indices.create(index=INDEX_NAME, body=mapping)

#Confirm creation.
print(f"Created index: {INDEX_NAME}")

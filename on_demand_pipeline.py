from datetime import datetime, timezone
import json
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

import boto3
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

# ======================
# Configuration
# ======================

BUCKET = "data608-arxiv-s3"

# Separate prefix for on-demand pipeline to avoid conflicts
RAW_PREFIX = "ondemand/arxiv_oai/pages/"
PROCESSED_PREFIX = "ondemand/arxiv_oai/records/"

# OpenSearch index
INDEX_NAME = "arxiv-papers"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# OpenSearch connection config
HOST = "localhost"
PORT = 9200
USERNAME = "admin"
PASSWORD = "data608ML!!!"

# arXiv OAI-PMH endpoint
OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"

# arXiv API endpoint
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"


# Default constraints
DEFAULT_UNTIL_DATE = "2024-12-31"
DEFAULT_MAX_RECORDS = 500
DEFAULT_TOP_K = 5

# Performance tuning
BATCH_SIZE = 64
BULK_CHUNK_SIZE = 500

# XML namespace mapping
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}

# ======================
# Initialize clients
# ======================

s3 = boto3.client("s3")

client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

model = SentenceTransformer(MODEL_NAME)

# ======================
# OAI-PMH helpers
# ======================


def build_arxiv_url(search_query="", max_results=10):
    """Build OAI-PMH request URL."""
    params = {
            "search_query": search_query,
            "max_results": max_results,
        }
    return f"{OAI_BASE_URL}?{urllib.parse.urlencode(params)}"

def build_oai_url(metadata_prefix="arXiv", until_date=None, resumption_token=None):
    """Build OAI-PMH request URL."""
    if resumption_token:
        params = {
            "verb": "ListRecords",
            "resumptionToken": resumption_token,
        }
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": metadata_prefix,
        }
        if until_date:
            params["until"] = until_date

    return f"{ARXIV_BASE_URL}?{urllib.parse.urlencode(params)}"

def fetch_oai_page(url):
    """Fetch XML page from arXiv."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DATA608-arXiv-ondemand-ingestion/1.0"},
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def check_oai_error(xml_bytes):
    """Check if OAI response contains an error."""
    root = ET.fromstring(xml_bytes)
    error = root.find(".//oai:error", NS)

    if error is not None:
        raise ValueError(f"OAI Error: {error.text}")


def parse_page_metadata(xml_bytes):
    """Extract pagination metadata."""
    root = ET.fromstring(xml_bytes)

    response_date = root.findtext(
        "oai:responseDate",
        default="",
        namespaces=NS,
    )

    token_elem = root.find(".//oai:resumptionToken", NS)
    resumption_token = (
        token_elem.text.strip()
        if token_elem is not None and token_elem.text
        else None
    )

    records = root.findall(".//oai:record", NS)

    return {
        "response_date": response_date,
        "resumption_token": resumption_token,
        "record_count": len(records),
    }
'''def parse_page_metadata_arxiv_api(xml_bytes):
    """Extract pagination metadata."""
    root = ET.fromstring(xml_bytes)

    response_date = root.findtext(
        "oai:responseDate",
        default="",
        namespaces=NS,
    )

    token_elem = root.find(".//oai:resumptionToken", NS)
    resumption_token = (
        token_elem.text.strip()
        if token_elem is not None and token_elem.text
        else None
    )

    records = root.findall(".//oai:record", NS)

    return {
        "response_date": response_date,
        "resumption_token": resumption_token,
        "record_count": len(records),
    }

'''
def parse_arxiv_atom(xml: str):
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall('atom:entry', ns):
        raw_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        title = entry.findtext('atom:title', default='', namespaces=ns).strip()
        abstract = entry.findtext('atom:summary', default='', namespaces=ns).strip()
        categories = [
            c.attrib.get("term", "").strip()
            for c in entry.findall("atom:category", ns)
        ]
        created = entry.findtext('atom:updated', default="", namespaces=ns).strip()
        papers.append({'arxiv_id': arxiv_id, 'title': title, 'abstract': abstract, 'url': f"https://arxiv.org/abs/{arxiv_id}", 'categories': categories, 'created': created })
    return papers


def parse_arxiv_records(xml_bytes):
    """Parse arXiv records from XML."""
    root = ET.fromstring(xml_bytes)
    parsed_records = []

    for record in root.findall(".//oai:record", NS):
        header = record.find("oai:header", NS)
        if header is not None and header.attrib.get("status") == "deleted":
            continue

        meta = record.find("oai:metadata/arxiv:arXiv", NS)
        if meta is None:
            continue

        arxiv_id = meta.findtext("arxiv:id", "", NS).strip()
        title = meta.findtext("arxiv:title", "", NS).strip()
        abstract = meta.findtext("arxiv:abstract", "", NS).strip()
        created = meta.findtext("arxiv:created", "", NS).strip()

        categories_text = meta.findtext("arxiv:categories", "", NS).strip()
        categories = categories_text.split() if categories_text else []

        if not arxiv_id:
            continue

        parsed_records.append(
            {
                "arxiv_id": arxiv_id,
                "title": " ".join(title.split()),
                "abstract": " ".join(abstract.split()),
                "link": f"https://arxiv.org/abs/{arxiv_id}",
                "categories": categories,
                "created": created,
            }
        )

    return parsed_records


# ======================
# S3 helpers
# ======================


def write_raw_xml_to_s3(bucket_name, xml_bytes, run_time, page_num, run_id):
    """Write raw XML response to S3."""
    key = f"{RAW_PREFIX}{run_time:%Y/%m/%d}/oai_{run_id}_{page_num}.xml"

    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=xml_bytes,
        ContentType="application/xml",
    )
    return key


def write_parsed_json_to_s3(bucket_name, records, run_time, run_id):
    """Write parsed JSON records to S3."""
    key = f"{PROCESSED_PREFIX}{run_time:%Y/%m/%d}/records_{run_id}.json"

    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(records).encode("utf-8"),
        ContentType="application/json",
    )
    return key


# ======================
# Embedding + indexing
# ======================


def build_text(record):
    """Combine title and abstract for embedding."""
    return f"{record.get('title', '')}. {record.get('abstract', '')}".strip()


def embed_and_index(records):
    """Generate embeddings and index records into OpenSearch."""
    if not records:
        return {"indexed": 0, "failed": 0}

    texts = [build_text(record) for record in records]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    actions = []
    for record, emb in zip(records, embeddings):
        actions.append(
            {
                "_op_type": "index",
                "_index": INDEX_NAME,
                "_id": record["arxiv_id"],
                "_source": {
                    "arxiv_id": record["arxiv_id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "link": record["link"],
                    "created": record["created"],
                    "categories": record["categories"],
                    "combined_text": build_text(record),
                    "embedding": emb.tolist(),
                },
            }
        )

    success, failed = helpers.bulk(client, actions, stats_only=True)
    return {"indexed": success, "failed": failed}


# ======================
# Search
# ======================


def semantic_search(query, top_k=DEFAULT_TOP_K):
    """Run semantic vector search in OpenSearch."""
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    body = {
        "size": top_k,
        "_source": ["title", "abstract", "link"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
    }

    response = client.search(index=INDEX_NAME, body=body)

    results = []
    for hit in response["hits"]["hits"]:
        result = hit["_source"].copy()
        result["score"] = hit["_score"]
        results.append(result)

    return results


def format_hits_for_app(hits):
    """Format OpenSearch hits for the app response."""
    return [
        {
            "title": hit.get("title"),
            "authors": [],
            "abstract": hit.get("abstract"),
            "url": hit.get("link"),
            "score": hit.get("score"),
        }
        for hit in hits
    ]


# ======================
# Data collection
# ======================


def collect_ondemand_records(max_records, until_date, bucket_name, search_query):
    """Collect arXiv records on demand and store raw/processed copies in S3."""
    run_time = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:8]
    
    all_records = []
    resumption_token = None
    page_num = 0
    url = build_arxiv_url(search_query, max_records)
    with urllib.request.urlopen(url) as f:
        xml = f.read()
        
    '''
            import urllib.request as libreq
            
    
    
    while len(all_records) < max_records:
        url = build_oai_url(
            until_date=until_date if not resumption_token else None,
            resumption_token=resumption_token,
        )
        
        xml = fetch_oai_page(url)
        check_oai_error(xml)

        page_num += 1
        write_raw_xml_to_s3(bucket_name, xml, run_time, page_num, run_id)

        meta = parse_page_metadata(xml)
        records = parse_arxiv_records(xml)

        remaining = max_records - len(all_records)
        all_records.extend(records[:remaining])

        resumption_token = meta["resumption_token"]
        if not resumption_token:
            break
    '''
    write_raw_xml_to_s3(bucket_name, xml, run_time, page_num, run_id)
    all_records = parse_arxiv_atom(xml)
    json_key = write_parsed_json_to_s3(bucket_name, all_records, run_time, run_id)

    return {
        "records": all_records,
        "count": len(all_records),
        "json_key": json_key,
    }


# ======================
# Main pipeline
# ======================


def run_ondemand_pipeline(
    user_query,
    search_query,
    max_records=DEFAULT_MAX_RECORDS,
    until_date=DEFAULT_UNTIL_DATE,
    top_k=DEFAULT_TOP_K,
    bucket_name=BUCKET,
):
    """Run the full on-demand ingestion, indexing, and search pipeline."""
    if not user_query.strip():
        raise ValueError("Query cannot be empty")

    data = collect_ondemand_records(max_records, until_date, bucket_name, search_query)
    stats = embed_and_index(data["records"])

    hits = semantic_search(user_query, top_k)
    formatted = format_hits_for_app(hits)

    return {
        "records_added": data["count"],
        "indexed": stats["indexed"],
        "failed": stats["failed"],
        "results": formatted,
        "top_score": formatted[0]["score"] if formatted else None,
    }

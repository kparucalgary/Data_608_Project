import json
from datetime import datetime, timezone
import boto3

BUCKET_NAME = "data608-arxiv-logs-s3"

s3 = boto3.client("s3")

payload = {
    "test": "hello from ec2",
    "timestamp": datetime.now(timezone.utc).isoformat()
}

s3.put_object(
    Bucket=BUCKET_NAME,
    Key="query-logs/test.json",
    Body=json.dumps(payload),
    ContentType="application/json"
)

print("SUCCESS")

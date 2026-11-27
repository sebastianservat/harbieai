"""Which Claude models can this account call? Run: uv run python check_models.py"""
import os

import boto3
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_REGION", "ap-southeast-2")
bedrock = boto3.client("bedrock", region_name=REGION)
runtime = boto3.client("bedrock-runtime", region_name=REGION)

# 1. Every Claude inference profile offered in this region
print(f"Claude inference profiles in {REGION}:")
try:
    profiles = bedrock.list_inference_profiles(maxResults=1000)["inferenceProfileSummaries"]
    claude_ids = sorted(p["inferenceProfileId"] for p in profiles if "anthropic" in p["inferenceProfileId"])
    for pid in claude_ids:
        print("  ", pid)
except Exception as e:  # the API key may not be allowed to list
    claude_ids = []
    print("   could not list:", type(e).__name__, str(e)[:120])

# 2. Send a tiny request to each one: the real test of access
candidates = claude_ids or [
    "au.anthropic.claude-haiku-5-5",
    "au.anthropic.claude-sonnet-5",
    "au.anthropic.claude-sonnet-4-6",
    "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
]
print("\nAccess test:")
for model_id in candidates:
    try:
        runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Say ok"}]}],
            inferenceConfig={"maxTokens": 5},
        )
        print(f"   OK      {model_id}")
    except Exception as e:
        print(f"   FAILED  {model_id}  ({type(e).__name__}: {str(e)[:90]})")
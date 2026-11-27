import os
from functools import lru_cache

from langchain_aws import ChatBedrockConverse

@lru_cache
def get_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model_id=os.getenv("BEDROCK_MODEL_ID", "au.anthropic.claude-haiku-4-5-20251001-v1:0"),
        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
        provider="anthropic",
        max_tokens=1024,
    )

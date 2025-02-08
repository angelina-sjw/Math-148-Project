from openai import OpenAI
from typing import Union, List
import base64
from ..data_types import schema1
import requests
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, openai_api_key: str, clip_server_url: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.clip_server_url = clip_server_url

    async def extract_keywords(self, query: str) -> str:
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "system", "content": "You are a helpful assistant that extracts keywords from a user's query."}, {"role": "user", "content": query}],
            max_tokens=100,
            response_format=schema1
        )
        return response.choices[0].message.content

    async def get_embedding(self, input_data: Union[str, bytes], input_type: str) -> List[float]:
        try:
            if input_type == "text":
                response = requests.post(
                    f"{self.clip_server_url}/embed/text",
                    json={"text": input_data}
                )
            elif input_type == "image":
                base64_data = base64.b64encode(input_data).decode('utf-8')
                response = requests.post(
                    f"{self.clip_server_url}/embed/image",
                    json={"image_data": base64_data}
                )
            else:
                raise ValueError("input_type must be either 'text' or 'image'")

            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                logger.error(f"Server error: {response.text}")
                raise Exception(f"Error from CLIP server: {response.text}")
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise Exception(f"Failed to get embedding: {str(e)}")


from openai import OpenAI
from typing import Union, List
import base64
from ..data_types import schema1

class LLMClient:
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)

    async def extract_keywords(self, query: str) -> str:
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "system", "content": "You are a helpful assistant that extracts keywords from a user's query."}, {"role": "user", "content": query}],
            max_tokens=100,
            response_format=schema1
        )
        return response.choices[0].message.content

    async def get_embedding(self, input_data: Union[str, bytes], input_type: str) -> List[float]:

        if input_type == "text":
            response = await self.openai_client.embeddings.create(
                model="clip",
                input=input_data,
                encoding_format="float"
            )
            return response.data[0].embedding
        
        elif input_type == "image":
            base64_image = base64.b64encode(input_data).decode('utf-8')
            response = await self.openai_client.embeddings.create(
                model="clip",
                input=base64_image,
                encoding_format="float"
            )
            return response.data[0].embedding
        
        else:
            raise ValueError("input_type must be either 'text' or 'image'")


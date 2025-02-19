import os
from typing import Optional, List

from openai import OpenAI
from lmdeploy import pipeline, TurbomindEngineConfig
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self,model_id: str, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 stream: bool = False):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.llm = None
        self.stream = stream
        if not api_key:
            backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)
            self.llm = pipeline(model_id,backend_config=backend_config)
            
    def __call__(self, prompt):                
        if self.llm:
            return self.llm(prompt).text
        elif self.api_key:
            client = OpenAI(api_key=self.api_key, base_url= self.base_url)
            
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=self.stream
            )
            
            if self.stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
        else:
            raise RuntimeError("Neither API key nor local model initialized.")
            
    def _stream_response(self, response):
        """Helper function that is only called when streaming"""
        for chunk in response:
            yield chunk.choices[0].delta.content
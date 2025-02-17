import os
import subprocess
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest, ScoredPoint

import pandas as pd
from FlagEmbedding import BGEM3FlagModel,FlagReranker
from dotenv import load_dotenv
from utils import set_proxy

load_dotenv()
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["HF_HOME"] = "/root/autodl-tmp/.cache/huggingface"
# os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
# os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/.cache/modelscope/'
# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value
class Retriever:
    def __init__(self, embedder: str, reranker: str, collection_name: str):
        with set_proxy():
            self.embedder = BGEM3FlagModel(embedder, use_fp16=True)
            self.reranker = FlagReranker(reranker, use_fp16=True)
        self.collection_name = collection_name
        self.client = QdrantClient(path=os.getenv('vs_path'))

    def retrieve(self, query: str, topk: int = 50):
        
        # 1. Generate query embedding
        query_embed = self.embedder.encode([query])["dense_vecs"][0]
        
        # 2.  recall top k docs
        search_result = self.client.query_points(collection_name="sales_qa", query=query_embed.tolist(),using="text",limit = topk).points
        retrieved_docs = [(point.payload["description"], point.score) for point in search_result]

        # 3. reranking
        pairs = [(query, doc) for doc, score in retrieved_docs]
   
        scores = self.reranker.compute_score(pairs)
        
        reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_chunks = [(retrieved_docs[i][0], scores[i]) for i in reranked_indices]
        
        return retrieved_docs, reranked_chunks



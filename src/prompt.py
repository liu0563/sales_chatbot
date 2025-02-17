from typing import Optional, List

past_k_rounds = 5

def generate_prompt(prompt_type: str, query: str, history: List[dict], reranked_chunks: Optional[List] = None):
    if prompt_type == 'rewrite':
        #whole_conv = " ".join([past_convs[i*2] for i in range(len(past_convs)//2)]) + "\n\n" + "User: " + query

        p = f"""
        You are a knowledgeable and helpful sales assistant at a car dealership. Your task is to refine the user's query in a multi-turn conversation to ensure clarity, coherence, and relevance while preserving their exact intent. Use the conversation history to maintain context.
        
        ### Instructions:
        - Review the conversation history and the user's latest query.
        - Rewrite the query to make it clearer and more specific.
        - Do not add or infer additional information!
        
        ### Conversation history:
        {history[-past_k_rounds*2:]}
        
        ### Rewritten query:
        
        """
        
    elif prompt_type == 'generate':
        
        p = f"""

        You are assisting a customer in selecting a car. Below are five retrieved pieces of car-related information relevant to the customer's query, along with up to 10 exchanges of recent conversation history (if available) for context.
        
        Use only the provided documents to answer the customer's query. Do not assume, infer, or generate facts or numbers beyond the given context. If the retrieved information is irrelevant or unhelpful, disregard it.
        
        ### Car Descriptions:
        1. {reranked_chunks[0][0]}
        2. {reranked_chunks[1][0]}
        3. {reranked_chunks[2][0]}
        4. {reranked_chunks[3][0]}
        5. {reranked_chunks[4][0]}
        
        ### Conversation History:
        {history}
        
        ### Customer Query:
        {query}
        
        ### Response:

        """
    
    return p

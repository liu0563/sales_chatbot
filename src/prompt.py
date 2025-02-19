from typing import Optional, List


def generate_prompt(prompt_type: str, query: str, history: List[dict], reranked_chunks: Optional[List] = None, top_k_chunks: int = 5, past_k_rounds: int = 5):
    if prompt_type == 'rewrite':
        # Extract relevant conversation history for rewrite prompt
        user_query_history = "\n".join(
            f"- User: {history[i*2]['content']}" for i in range(min(len(history)//2, past_k_rounds))
        )
        
        # Construct the rewrite prompt
        p = f"""
        You are a knowledgeable and helpful sales assistant at a car dealership. Your task is to refine the user's query in a multi-turn conversation to ensure clarity, coherence, and relevance           while preserving their exact intent. Use the conversation history to maintain context.
        
        ### Instructions:
        - Review the conversation history and the user's latest query.
        - Rewrite the query to make it clearer and more specific.
        - Do not add or infer additional information!
        
        ### Conversation History:
        {user_query_history}
        
        ### Current User Query:
        {query}
        
        ### Rewritten Query:
        """
    
    elif prompt_type == 'generate':
        # Format retrieved chunks for generation prompt
        chunks = "\n".join(
            f"{i+1}. {chunk[0]}" for i, chunk in enumerate(reranked_chunks[:top_k_chunks])
        )
        
        # Construct the generation prompt
        p = f"""
        You are assisting a customer in selecting a car. Below are five retrieved pieces of car-related information relevant to the customer's query, along with up to {past_k_rounds * 2} exchanges of recent conversation history (if available) for context.
        
        ### Retrieved Car Information:
        {chunks}
        
        ### Conversation History:
        {history[-past_k_rounds*2:]}
        
        ### Customer Query:
        {query}

        - Only use information from the provided context. Do not add or assume details not explicitly stated.
        - If the information is not relevant or helpful, disregard it.
        - Keep the response clear and focused on the customer's needs without referencing the sources.

        ### Response:
        """
    
    return p

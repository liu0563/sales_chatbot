import gradio as gr
import os
import sys
import time

from src.retriever import Retriever
from src.LLM_utils import LLM
from src.prompt import generate_prompt

# Initialize the RAG components once
def initialize_rag():
    retriever = Retriever(
        embedder='BAAI/bge-m3',
        reranker='BAAI/bge-reranker-v2-m3',
        collection_name='sales_qa'
    )
    rewriter = LLM('Qwen/Qwen2.5-3B-Instruct')
    gen_llm = LLM(
        'ep-20250213200344-crq6r',
        api_key=os.getenv('ARK_API_KEY'),
        base_url=os.getenv('ARK_BASE_URL'),
        stream=True
    )
    return retriever, rewriter, gen_llm

# Initialize RAG components globally
retriever, rewriter, gen_llm = initialize_rag()

# RAG response generation
def rag_response(query, history):
    # Query rewriting using chat history as context
    rewrite_prompt = generate_prompt('rewrite', query, history, reranked_chunks=None)
    rewritten_query = rewriter(rewrite_prompt)
    
    # Retrieval
    recall_docs, reranked_chunks = retriever.retrieve(rewritten_query)
    
    # Response generation using chat history and retrieved chunks
    generation_prompt = generate_prompt('generate', query, history, reranked_chunks)
    response = gen_llm(generation_prompt)

    return response

# Gradio app
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]
        
    def bot(history: list):
        user_message = history[-1]["content"]  # Get the latest user message
        response = rag_response(user_message, history)
        
        assistant_response = ""
        for chunk in response:  # Stream the response
            assistant_response += chunk
            # Update the assistant's message in place
            history[-1]["content"] = assistant_response
            yield history
        history.append({"role": "assistant", "content": assistant_response})
        time.sleep(0.05)  # Simulate typing
            

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_port = 6006)
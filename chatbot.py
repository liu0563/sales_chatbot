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
    gr.Markdown("# 🚗 Car Dealership AI Assistant")
    gr.Markdown(
        "This chatbot assists with **car dealership inquiries**, including sales, service, and inventory questions. "
        "It uses a **Retrieval-Augmented Generation (RAG)** model to fetch relevant information before responding."
    )

    chatbot = gr.Chatbot(type="messages", label="Chat with AI Assistant")
    msg = gr.Textbox(
        placeholder="Type your question here...",
        label="Your Message",
    )
    clear = gr.Button("Clear Chat")
    
    # Example prompts
    examples = gr.Examples(
        examples=[
            "What cars are available in the inventory?",
            "Can I book a service appointment?",
            "Tell me about the financing options.",
            "What are the dealership hours?",
        ],
        inputs=[msg],
    )
    
    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]
        
    def bot(history: list):
        user_message = history[-1]["content"]  # Get the latest user message
        response = rag_response(user_message, history)
        
        assistant_response = ""
        history.append({"role": "assistant", "content": assistant_response})
        for chunk in response:  # Stream the response
            assistant_response += chunk
            # Update the assistant's message in place
            history[-1]["content"] = assistant_response
            yield history
        
        time.sleep(0.05)  # Simulate typing
            

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    #clear.click(lambda: None, None, chatbot, queue=False)
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_port = 6006,live=True)
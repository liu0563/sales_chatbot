

# Car Dealership Sales QA Bot

This project builds on a previous RAG-based QA chatbot developed to assist car dealership customers with sales inquiries. It is currently in active development, with future plans to integrate fine-tuning and Text2SQL modules.

## **Tech Stack**
- **Qdrant**: Vector database for efficient similarity search and retrieval
- **Gradio**: User interface framework for deploying the chatbot
- **lmdeploy**: Lightweight model deployment framework
- **Crawl4AI**: A web crawler for scraping LLM-ready data
- **bge embedder and reranker**: Embedding and reranking models
- **LLM Models**: Qwen-0.5B-instruct, Qwen-3B-instruct, and fine-tuned Mistral-7B-Instruct/DeepSeek-V3 for generating responses and query understanding and decomposition

## **Key Features**
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval of relevant documents with generative models to produce accurate and context-aware responses
- **Crawl4AI**: Designed a pipeline to scrape, clean, and extract car reviews from Edmunds.com
- **Gradio UI**: Provides an intuitive and user-friendly interface for interacting with the chatbot
- **Scalable Data Handling**: Utilizes Qdrant vector stores to efficiently store and manage, and accurately retrieve large-scale text data


https://github.com/user-attachments/assets/f7de3384-4abf-4e9a-9bb7-868de942aabb





# Car Dealership Sales QA Bot

This project simulates a RAG-based QA chatbot designed to assist car dealership customers with their sales-related inquiries. This project is still under active development, with plans to incorporate fine-tuning and Text2SQL modules in the future.

## **Tech Stack**
- **Qdrant**: Vector database for efficient similarity search and retrieval
- **Gradio**: User interface framework for deploying the chatbot
- **lmdeploy**: Lightweight model deployment framework
- **Crawl4AI**: A web crawler for scraping LLM-ready data
- **bge embedder and reranker**: Embedding and reranking models

## **Key Features**
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval of relevant documents with generative models to produce accurate and context-aware responses
- **Crawl4AI**: Designed a pipeline to scrape, clean, and extract car reviews from Edmunds.com
- **Gradio UI**: Provides an intuitive and user-friendly interface for interacting with the chatbot
- **Scalable Data Handling**: Utilizes Qdrant vector stores to efficiently store and manage, and accurately retrieve large-scale text data

## **File Structure**
- **`chatbot.py`**: The main script to run the chatbot with a Gradio-based user interface
- **`data`**: Scrapped and preprocessed data to store in vector database:
  - **Car manuals**: Example: `2015-Nissan-Quest-owner-manual.md`
  - **Pickle files**: Serialized car review data, such as `bmw.pkl` and `ford.pkl`
  - **CSV files**: Queries and tickets data, such as `car_queries.csv`
- **`src`**: Core source code for the chatbot:
  - **`LLM_utils.py`**: Utility functions for interfacing with large language models
  - **`retriever.py`**: Logic for retrieving relevant information from the dataset
  - **`prompt.py`**: Functions for generating and managing prompts for the language model
- **`notebooks`**: Development and testing notebooks for experimenting with and refining the chatbot's functionality


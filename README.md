# RAG Demo System

A demonstration of Retrieval-Augmented Generation (RAG) using Pinecone and OpenAI.

## Features

- Semantic search using Pinecone vector database
- Question answering using OpenAI's ChatGPT
- Real-time similarity scoring and context retrieval
- Clean console output with visual progress tracking

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

## Usage

Run the demo:
```bash
python rag_demo.py
```

The system will:
1. Initialize the RAG system
2. Process and index the knowledge base
3. Answer questions using the indexed knowledge
4. Show similarity scores and relevance for each answer

## How it Works

1. **Document Processing**: Text is split into chunks and converted to embeddings
2. **Vector Storage**: Embeddings are stored in Pinecone for efficient retrieval
3. **Query Processing**: User questions are converted to embeddings and matched against the knowledge base
4. **Answer Generation**: Relevant context is retrieved and used by ChatGPT to generate accurate answers

## Notes

- The system uses a similarity threshold of 0.6 to ensure high-quality matches
- The knowledge base can be modified by editing `knowledge_base.txt`
- All operations are logged with detailed progress information

import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import sys

# Set console encoding to UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load environment variables
load_dotenv()

class RAGDemo:
    def __init__(self):
        """Initialize the RAG system with necessary components."""
        print("\n" + "╔" + "═"*48 + "╗")
        print("║          RAG System Initialization             ║")
        print("╚" + "═"*48 + "╝")
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        
        # Connect to existing index
        index_name = "rag-test"
        self.index = self.pc.Index(index_name)
        
        # Initialize models
        self.embeddings = SentenceTransformer('intfloat/multilingual-e5-large')
        self.llm = ChatOpenAI(temperature=0)
        self.index_name = index_name
        self.namespace = "rag_demo"

    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using the E5 model."""
        embeddings = self.embeddings.encode([f"passage: {text}"], normalize_embeddings=True)
        return embeddings[0].tolist()  # Convert numpy array to list

    def wait_for_vectors(self, expected_count: int, max_retries: int = 5) -> bool:
        """Wait for vectors to appear in the index stats."""
        import time
        for i in range(max_retries):
            stats = self.index.describe_index_stats()
            current_count = stats['namespaces'].get(self.namespace, {}).get('vector_count', 0)
            if current_count >= expected_count:
                return True
            time.sleep(2 ** i)  # Exponential backoff
        return False

    def load_and_process_documents(self, file_path: str) -> int:
        """Load and process documents into chunks for embedding."""
        print("\n" + "╔" + "═"*48 + "╗")
        print("║          Knowledge Base Processing             ║")
        print("╚" + "═"*48 + "╝")
        
        # Read and process the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        chunks = text_splitter.split_text(text)
        print(f"[>] Processing {len(chunks)} text chunks")
        
        # Create embeddings and prepare vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.get_embeddings(chunk)
            vector_id = f"doc_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": chunk}
            })
        
        # Upsert to Pinecone
        try:
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
            
            # Wait for vectors to appear
            if self.wait_for_vectors(len(vectors)):
                print("[+] Knowledge base processed and indexed")
            else:
                print("[-] Warning: Some vectors may not be immediately visible")
            
            return len(chunks)
        except Exception as e:
            print(f"[-] Error during upload: {e}")
            raise

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        print("\n" + "╭" + "─"*48 + "╮")
        print("│                  Question                      │")
        print("╰" + "─"*48 + "╯")
        print(f"  {question}                      ")

        
        print("\n" + "╔" + "═"*48 + "╗")
        print("║            RAG Pipeline Steps                  ║")
        print("╚" + "═"*48 + "╝")
        print("[*] Processing query through RAG pipeline...")
        
        # Generate embedding for the question
        query_embedding = self.get_embeddings(question)
        print("[1] Generated question embedding")
        
        # Query Pinecone
        try:
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            # Filter out low-relevance matches
            relevant_matches = [match for match in results['matches'] if match['score'] > 0.6]
            
            if not relevant_matches:
                print("[-] No relevant information found in knowledge base")
                print("    Similarity scores were below threshold (0.6)")
                
                print("\n" + "╭" + "─"*48 + "╮")
                print("│                   Answer                       │")
                print("│" + "─"*48 + "│")
                print("│  I cannot answer this question from the provided")
                print("│  context.")
                print("╰" + "─"*48 + "╯")
                return "I cannot answer this question from the provided context."
            
            # Get context from matches
            context = "\n".join([match['metadata']['text'] for match in relevant_matches])
            print(f"[2] Found {len(relevant_matches)} relevant passages")
            print("    Top match similarity score:", f"{relevant_matches[0]['score']:.2%}")
            
            # Generate answer
            prompt = f"""Answer the question using ONLY the information provided in the context. If the answer cannot be directly found in the context, say "I cannot answer this from the provided context."

Context:
{context}

Question: {question}

Answer (use ONLY information from the context):"""
            
            print("[3] Generating answer from relevant context")
            
            response = self.llm.invoke(prompt)
            print("\n" + "╭" + "─"*48 + "╮")
            print("│                   Answer                       │")
            print("╰" + "─"*48 + "╯")
            print(f"  {response.content}")
            return response.content
            
        except Exception as e:
            print(f"[-] Error during query: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            stats = self.index.describe_index_stats()
            if stats['namespaces'].get(self.namespace, {}).get('vector_count', 0) > 0:
                self.index.delete(deleteAll=True, namespace=self.namespace)
        except Exception as e:
            print(f"[-] Error during cleanup: {e}")

def main():
    # Create sample knowledge base
    sample_text = """
    Artificial Intelligence (AI) is the simulation of human intelligence by machines.
    Machine Learning is a subset of AI that enables systems to learn from data.
    Deep Learning is a type of machine learning based on artificial neural networks.
    Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.
    Computer Vision is the field of AI that enables computers to derive meaningful information from visual inputs.
    """
    
    with open("knowledge_base.txt", "w", encoding='utf-8') as f:
        f.write(sample_text)
    
    rag = None
    try:
        # Initialize RAG system
        rag = RAGDemo()
        
        # Load documents
        rag.load_and_process_documents("knowledge_base.txt")
        
        # Example queries - mix of in-domain and out-of-domain questions
        questions = [
            "What is artificial intelligence?",  # In knowledge base
            "What is quantum computing?",  # Not in knowledge base
            "Can you explain what NLP is?",  # In knowledge base
            "Who invented the internet?",  # Not in knowledge base
        ]
        
        # Run queries
        for question in questions:
            rag.query(question)
    
    finally:
        # Clean up resources
        if rag:
            rag.cleanup()

if __name__ == "__main__":
    main()

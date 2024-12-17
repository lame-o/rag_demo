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
import re
import textwrap

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
        # Add color codes
        self.BLUE = "\033[94m"
        self.GREEN = "\033[92m"
        self.YELLOW = "\033[93m"
        self.RED = "\033[91m"
        self.BOLD = "\033[1m"
        self.RESET = "\033[0m"
        self.CYAN = "\033[96m"
        
        # Initialize text wrapping and column formatting
        self.WIDTH = 45  # Standard column width
        self.TOTAL_WIDTH = (self.WIDTH * 2) + 8  # Total width including borders and padding
        
        # Display initialization header
        print("\n" + "╔" + "═"*(self.TOTAL_WIDTH) + "╗")
        print("║" + " "*((self.TOTAL_WIDTH-50)//2) + "______  ___  _____           ______     _____ " + " "*((self.TOTAL_WIDTH-50)//2) + "║")
        print("║" + " "*((self.TOTAL_WIDTH-50)//2) + "| ___ \\/ _ \\|  __ \\          | ___ \\   /  __ \\" + " "*((self.TOTAL_WIDTH-50)//2) + "║")
        print("║" + " "*((self.TOTAL_WIDTH-50)//2) + "| |_/ / /_\\ \\ |  \\/  ______  | |_/ /__ | /  \\/" + " "*((self.TOTAL_WIDTH-50)//2) + "║")
        print("║" + " "*((self.TOTAL_WIDTH-50)//2) + "|    /|  _  | | __  |______| |  __/ _ \\| |    " + " "*((self.TOTAL_WIDTH-50)//2) + "║")
        print("║" + " "*((self.TOTAL_WIDTH-50)//2) + "| |\\ \\| | | | |_\\ \\          | | | (_) | \\__/\\" + " "*((self.TOTAL_WIDTH-50)//2) + "║")
        print("║" + " "*((self.TOTAL_WIDTH-50)//2) + "\\_| \\_\\_| |_/\\____/          \\_|  \\___/ \\____/" + " "*((self.TOTAL_WIDTH-50)//2) + "║")
        print("║" + "─"*(self.TOTAL_WIDTH) + "║")
        print("║  Welcome to the RAG (Retrieval Augmented Generation) Demo!" + " "*(self.TOTAL_WIDTH-63) + "║")
        print("║" + "─"*(self.TOTAL_WIDTH) + "║")
        print("║  Scenario: You are testing a customer service AI for TechStyle Merch," + " "*(self.TOTAL_WIDTH-72) + "║")
        print("║  a fictional tech-fashion company. This demo shows how RAG improves AI" + " "*(self.TOTAL_WIDTH-73) + "║")
        print("║  responses by grounding them in real company policies." + " "*(self.TOTAL_WIDTH-57) + "║")
        print("║" + "─"*(self.TOTAL_WIDTH) + "║")
        print("║  Compare the outputs:" + " "*(self.TOTAL_WIDTH-24) + "║")
        print("║  Left  → RAG Response (Grounded in TechStyle's actual policies)" + " "*(self.TOTAL_WIDTH-69) + "║")
        print("║  Right → Standard LLM Response (May contain incorrect information)" + " "*(self.TOTAL_WIDTH-72) + "║")
        print("╚" + "═"*(self.TOTAL_WIDTH) + "╝\n")
        
        # Helper function to calculate visible length of text (excluding ANSI codes)
        def visible_length(text):
            ansi_escape = re.compile(r'\033\[[0-9;]*m')
            return len(ansi_escape.sub('', text))
        self.visible_length = visible_length
        
        def format_line(text, width, color=None):
            """Format a single line of text with proper padding"""
            if color:
                colored_text = f"{color}{text}{self.RESET}"
            else:
                colored_text = text
            
            padding = width - visible_length(text)
            return colored_text + " " * padding
        
        def format_columns(left, right, width):
            """Format two columns with proper padding"""
            left_formatted = format_line(left, width, self.CYAN)
            right_formatted = format_line(right, width, self.YELLOW)
            return f"║  {left_formatted}║  {right_formatted}║"
        
        self.format_columns = format_columns
        self.format_line = format_line
        
        def wrap_text(text, width):
            # Wrap text while preserving words
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word)
                if current_length + word_length + 1 <= width:
                    current_line.append(word)
                    current_length += word_length + 1
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length + 1
            
            if current_line:
                lines.append(" ".join(current_line))
            
            return lines
        
        self.wrap_text = wrap_text
        
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
        print("\n" + "╔" + "═"*(self.TOTAL_WIDTH) + "╗")
        print(f"║{' '*((self.TOTAL_WIDTH)//2 - 10)}{self.BOLD}RAG System Setup{self.RESET}{' '*((self.TOTAL_WIDTH)//2 - 10)}    ║")
        print("╠" + "═"*(self.WIDTH + 4) + "╦" + "═"*(self.WIDTH + 3) + "╣")
        print(f"║  {self.BOLD}Augmenting Knowledge Base{self.RESET}" + " "*(self.WIDTH - 21) + f"║  {self.BOLD}LLM Prompt{self.RESET}" + " "*(self.WIDTH - 13) + "║")
        print("╠" + "═"*(self.WIDTH + 4) + "╬" + "═"*(self.WIDTH + 3) + "╣")
        
        # Read and process the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get base prompt
        base_prompt = """You are the customer service AI for TechStyle Merch. Answer the question using the provided context.

Important Instructions:
1. If the exact answer is in the context, provide it completely.
2. If the question conflicts with our policies, explain why it's not possible by referencing our actual policy.
3. If the context has relevant information, use it to provide a complete and helpful response.
4. Only say "I cannot answer this from the provided context" if there is absolutely no relevant information.

For example:
- If someone asks about returns after 90 days, explain that returns must be within 45 days.
- If someone asks about washing instructions, provide all relevant care details from the context.
- If someone asks about membership benefits, list all relevant tier information."""

        # Split and wrap both texts
        kb_lines = self.wrap_text(text.strip(), self.WIDTH)
        prompt_lines = self.wrap_text(base_prompt.strip(), self.WIDTH)
        
        # Display both columns
        max_lines = max(len(kb_lines), len(prompt_lines))
        for i in range(max_lines):
            kb_line = kb_lines[i] if i < len(kb_lines) else ""
            prompt_line = prompt_lines[i] if i < len(prompt_lines) else ""
            print(self.format_columns(kb_line, prompt_line, self.WIDTH))
        
        print("╠" + "═"*(self.WIDTH + 4) + "╩" + "═"*(self.WIDTH + 3) + "╣")
        print(f"║{' '*((self.TOTAL_WIDTH)//2 - 8)}{self.BOLD}Processing Steps{self.RESET}{' '*((self.TOTAL_WIDTH)//2 - 8)}║")
        print("╚" + "═"*(self.TOTAL_WIDTH) + "╝")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        chunks = text_splitter.split_text(text)
        print(f"{self.BLUE}[>] Processing {len(chunks)} text chunks{self.RESET}")
        
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
                print(f"{self.GREEN}[+] Knowledge base processed and indexed{self.RESET}")
            else:
                print(f"{self.RED}[-] Warning: Some vectors may not be immediately visible{self.RESET}")
            
            return len(chunks)
        except Exception as e:
            print(f"{self.RED}[-] Error during upload: {e}{self.RESET}")
            raise

    def query(self, question: str) -> str:
        """Process a question through the RAG pipeline."""
        # Display question and pipeline steps side by side
        print("\n" + "╔" + "═"*(self.TOTAL_WIDTH) + "╗")
        print(f"║{' '*((self.TOTAL_WIDTH)//2 - 10)}{self.BOLD}Query Processing{self.RESET}{' '*((self.TOTAL_WIDTH)//2 - 10)}    ║")
        print("╠" + "═"*(self.WIDTH + 4) + "╦" + "═"*(self.WIDTH + 3) + "╣")
        print(f"║  {self.BOLD}Question{self.RESET}" + " "*(self.WIDTH - 8) + f"║  {self.BOLD}RAG Pipeline Steps{self.RESET}" + " "*(self.WIDTH - 16) + "║")
        print("╠" + "═"*(self.WIDTH + 4) + "╬" + "═"*(self.WIDTH + 3) + "╣")
        
        # Format question and pipeline steps
        question_lines = self.wrap_text(question, self.WIDTH)
        pipeline_steps = [
            f"{self.BLUE}[*] Processing query through RAG pipeline...{self.RESET}",
            f"{self.GREEN}[1] Generated question embedding{self.RESET}",
            f"{self.GREEN}[2] Found {{}} relevant passages{self.RESET}",
            f"{self.GREEN}    Top match similarity score: {{:.2f}}%{self.RESET}",
            f"{self.GREEN}[3] Generating answer from relevant context{self.RESET}"
        ]
        
        # Display both columns
        max_lines = max(len(question_lines), len(pipeline_steps))
        for i in range(max_lines):
            q_line = question_lines[i] if i < len(question_lines) else ""
            p_line = pipeline_steps[i] if i < len(pipeline_steps) else ""
            
            # Format each column with proper padding
            q_formatted = self.format_line(q_line, self.WIDTH, self.CYAN)
            p_formatted = self.format_line(p_line, self.WIDTH, self.GREEN)
            print(f"║  {q_formatted}║  {p_formatted}║")
        
        print("╚" + "═"*(self.TOTAL_WIDTH) + "╝")
        
        # Generate embedding for the question
        query_embedding = self.get_embeddings(question)
        pipeline_width = self.TOTAL_WIDTH  # Match the full width of the box
        print(f"{self.GREEN}[1] Generated question embedding{' ' * (pipeline_width - len('[1] Generated question embedding') - 20)}{self.RESET}")
        
        # Query Pinecone
        try:
            # Query Pinecone
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            # Filter out low-relevance matches
            relevant_docs = [match for match in results['matches'] if match['score'] > 0.7]
            max_similarity = max(match['score'] for match in relevant_docs)
            
            if not relevant_docs:
                default_response = ("I apologize, but I don't have any information about that in our current policies and guidelines. "
                                 "Is there something else I can help you with regarding our documented policies?")
                
                print("\n" + "╭" + "─"*48 + "╮")
                print("│                   Answer                       │")
                print("│" + "─"*48 + "│")
                # Split the response into two lines for better formatting
                lines = textwrap.wrap(default_response, width=46)
                for line in lines:
                    print("│  " + line + " " * (46 - len(line)) + "│")
                print("╰" + "─"*48 + "╯")
                
                print("\n" + "╔" + "═"*48 + "╗")
                print("║             Unconstrained Answer              ║")
                print("║" + "─"*48 + "║")
                
                # Get unconstrained response from OpenAI
                unconstrained_response = self.llm.invoke(question)
                print("║  " + unconstrained_response.content.replace("\n", "\n║  "))
                print("╰" + "─"*48 + "╯")
                
                return default_response
            
            # Get context from matches
            context = "\n".join([match['metadata']['text'] for match in relevant_docs])
            
            # Format pipeline step outputs with consistent width
            pipeline_width = self.TOTAL_WIDTH  # Match the full width of the box
            print(f"{self.GREEN}[2] Found {len(relevant_docs)} relevant passages{' ' * (pipeline_width - len('[2] Found X relevant passages') - 20)}{self.RESET}")
            print(f"{self.GREEN}    Top match similarity score: {max_similarity*100:.2f}%{' ' * (pipeline_width - len('    Top match similarity score: XX.XX%') - 20)}{self.RESET}")
            
            # Generate answer
            prompt = f"""You are the customer service AI for TechStyle Merch. Answer the question using the provided context.

Important Instructions:
1. If the exact answer is in the context, provide it completely.
2. If the question conflicts with our policies, explain why it's not possible by referencing our actual policy.
3. If the context has relevant information, use it to provide a complete and helpful response.
4. Only say "I cannot answer this from the provided context" if there is absolutely no relevant information.

For example:
- If someone asks about returns after 90 days, explain that returns must be within 45 days.
- If someone asks about washing instructions, provide all relevant care details from the context.
- If someone asks about membership benefits, list all relevant tier information.

Context:
{context}

Question: {question}

Answer (be direct and reference our policies):"""
            
            print(f"{self.GREEN}[3] Generating answer from relevant context{' ' * (pipeline_width - len('[3] Generating answer from relevant context') - 20)}{self.RESET}")
            
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Get unconstrained answer for comparison
            unconstrained_prompt = f"You are the customer service AI for TechStyle Merch. Please answer this question: {question}"
            unconstrained_response = self.llm.invoke(unconstrained_prompt)
            
            # Display answers side by side
            print("\n" + "╔" + "═"*(self.TOTAL_WIDTH) + "╗")
            print(f"║{' '*((self.TOTAL_WIDTH)//2 - 10)}{self.BOLD}Response Comparison{self.RESET}{' '*((self.TOTAL_WIDTH)//2 - 10)}    ║")
            print("╠" + "═"*(self.WIDTH + 4) + "╦" + "═"*(self.WIDTH + 3) + "╣")
            print(f"║  {self.BOLD}RAG-Constrained Answer{self.RESET}" + " "*(self.WIDTH - 20) + f"║  {self.BOLD}Unconstrained Answer{self.RESET}" + " "*(self.WIDTH - 19) + "║")
            print("╠" + "═"*(self.WIDTH + 4) + "╬" + "═"*(self.WIDTH + 3) + "╣")
            
            # Wrap both answers
            rag_lines = self.wrap_text(answer, self.WIDTH)
            unconstrained_lines = self.wrap_text(unconstrained_response.content, self.WIDTH)
            
            # Display both columns with colors
            max_lines = max(len(rag_lines), len(unconstrained_lines))
            for i in range(max_lines):
                rag_line = rag_lines[i] if i < len(rag_lines) else ""
                unconstrained_line = unconstrained_lines[i] if i < len(unconstrained_lines) else ""
                
                # Format each column with proper padding
                rag_formatted = self.format_line(rag_line, self.WIDTH, self.GREEN)
                unconstrained_formatted = self.format_line(unconstrained_line, self.WIDTH, self.YELLOW)
                print(f"║  {rag_formatted}║  {unconstrained_formatted}║")
            
            print("╚" + "═"*(self.TOTAL_WIDTH) + "╝")
            
            return answer
            
        except Exception as e:
            print(f"{self.RED}[-] Error during query: {e}{self.RESET}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            stats = self.index.describe_index_stats()
            if stats['namespaces'].get(self.namespace, {}).get('vector_count', 0) > 0:
                self.index.delete(deleteAll=True, namespace=self.namespace)
        except Exception as e:
            print(f"{self.RED}[-] Error during cleanup: {e}{self.RESET}")

def main():
    # Create sample knowledge base for TechStyle Merch
    sample_text = """
    TechStyle Merch Return Policy (Updated December 2024):
    - All returns must be initiated within 45 days of purchase
    - Items must be unworn with original tags attached
    - Limited edition items marked as 'Exclusive Drop' are final sale
    - Returns require the special QR code from your digital receipt
    - Store credit is issued via our TechStyle digital wallet within 24 hours
    - International orders can only be returned at our partner hubs
    
    Shipping Information:
    - Free shipping on orders over $75 using code TECHSHIP
    - Express shipping (2-day) available only in mainland USA
    - International shipping uses TechStyle Global Network (TGN)
    - Hawaii/Alaska orders ship via TechStyle Pacific (5-7 days)
    - Signature required for orders over $200
    
    Membership Tiers:
    - Digital Insider: Free tier, access to regular drops
    - Tech Elite: $99/year, early access and 10% off
    - Style Master: $199/year, free express shipping and 15% off
    - Innovation Circle: Invite-only, custom benefits
    
    Product Care:
    - All graphic tees must be washed inside-out in cold water
    - Smart hoodies with embedded tech must not be machine dried
    - Limited edition metallic prints require special TechCare solution
    - Augmented Reality shirts must be stored flat to maintain QR integrity
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
            "What is your return policy?",  # In knowledge base
            "Can I return items after 90 days?",  # Conflicts with policy
            "How do I wash the smart hoodie?",  # In knowledge base
            "What's your price matching policy?",  # Not in knowledge base
            "Do you offer student discounts?",  # Not in knowledge base
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

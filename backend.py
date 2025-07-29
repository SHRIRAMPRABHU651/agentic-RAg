import os
import tempfile
import json
import hashlib
import time
import re
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import List, Dict, Tuple, Optional, Any

# Document processing imports
import PyPDF2
from docx import Document
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG Assistant API",
    description="Intelligent Document Analysis with Self-Healing Retrieval & Adaptive Learning",
    version="1.0.0"
)

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG backend is running."}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use a proper database)
SYSTEM_STATE = {
    "rag_system": None,
    "documents_loaded": False,
    "sample_questions": [],
    "chat_history": [],
    "processing_status": ""
}

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    source: str
    page: Optional[int] = None
    chunk_id: str = ""
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]

@dataclass
class RetrievalContext:
    """Context for retrieval operations"""
    query: str
    chunks: List[DocumentChunk]
    confidence_score: float
    retrieval_strategy: str
    timestamp: datetime

@dataclass
class AgentMemory:
    """Memory system for the agentic RAG"""
    conversation_history: List[Dict[str, Any]]
    successful_retrievals: List[RetrievalContext]
    failed_queries: List[str]
    document_summaries: Dict[str, str]
    
    def add_interaction(self, query: str, response: str, confidence: float, chunks: List[DocumentChunk]):
        """Add interaction to memory"""
        self.conversation_history.append({
            "query": query,
            "response": response,
            "confidence": confidence,
            "chunks": len(chunks),
            "timestamp": datetime.now().isoformat()
        })

class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt'}
    
    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        text_chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        text_chunks.append((text, page_num))
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise
        return text_chunks
    
    def extract_text_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from DOCX"""
        text_chunks = []
        try:
            doc = Document(file_path)
            current_text = ""
            page_num = 1
            
            for paragraph in doc.paragraphs:
                current_text += paragraph.text + "\n"
                # Simple page break detection (approximate)
                if len(current_text) > 2000:  # Rough page size
                    if current_text.strip():
                        text_chunks.append((current_text, page_num))
                    current_text = ""
                    page_num += 1
            
            if current_text.strip():
                text_chunks.append((current_text, page_num))
                
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise
        return text_chunks
    
    def extract_text_from_txt(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Split into approximate pages
                chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                return [(chunk, idx+1) for idx, chunk in enumerate(chunks) if chunk.strip()]
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            raise
    
    def create_smart_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create intelligent text chunks with semantic boundaries"""
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if chunks and overlap > 0:
                    last_words = current_chunk.split()[-overlap//10:]  # Approximate word overlap
                    current_chunk = " ".join(last_words) + " " + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_documents(self, uploaded_files: List[UploadFile]) -> List[DocumentChunk]:
        """Process uploaded documents and create chunks"""
        all_chunks = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.filename)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.file.read())
                
                # Extract text based on file type
                file_ext = Path(uploaded_file.filename).suffix.lower()
                
                try:
                    if file_ext == '.pdf':
                        text_pages = self.extract_text_from_pdf(temp_path)
                    elif file_ext == '.docx':
                        text_pages = self.extract_text_from_docx(temp_path)
                    elif file_ext == '.txt':
                        text_pages = self.extract_text_from_txt(temp_path)
                    else:
                        logger.error(f"Unsupported file format: {file_ext}")
                        continue
                    
                    # Create chunks for each page
                    for text, page_num in text_pages:
                        if text.strip():
                            chunks = self.create_smart_chunks(text)
                            for chunk_text in chunks:
                                chunk = DocumentChunk(
                                    content=chunk_text,
                                    source=uploaded_file.filename,
                                    page=page_num
                                )
                                all_chunks.append(chunk)
                
                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.filename}: {e}")
                    continue
        
        return all_chunks

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Initialize FAISS index
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.chunks.append(chunk)
    
    def search(self, query: str, k: int = 5, min_similarity: float = 0.3) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.chunks)))
        
        # Filter by minimum similarity and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_similarity:
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def get_document_summary(self) -> Dict[str, int]:
        """Get summary of documents in the vector store"""
        summary = {}
        for chunk in self.chunks:
            if chunk.source in summary:
                summary[chunk.source] += 1
            else:
                summary[chunk.source] = 1
        return summary

class AgenticRAG:
    """Main agentic RAG system with self-healing and reasoning capabilities"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.vector_store = VectorStore()
        self.memory = AgentMemory([], [], [], {})
        self.confidence_threshold = 0.7
        self.max_retry_attempts = 3
        
        # Initialize generation config for temperature control
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=64,
            max_output_tokens=8192,
        )
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build LangGraph workflow for agentic RAG"""
        # Define nodes
        def analyze_query(state: dict) -> dict:
            """Analyze query patterns for agentic behavior"""
            query = state["query"]
            analysis = {
                "query_type": "general",
                "complexity": "medium",
                "domain_specific": False,
                "requires_multiple_sources": False
            }
            
            # Query type classification
            if any(word in query.lower() for word in ['what', 'define', 'explain']):
                analysis["query_type"] = "definition"
            elif any(word in query.lower() for word in ['how', 'process', 'steps']):
                analysis["query_type"] = "procedural"
            elif any(word in query.lower() for word in ['why', 'reason', 'cause']):
                analysis["query_type"] = "causal"
            elif any(word in query.lower() for word in ['compare', 'difference', 'versus']):
                analysis["query_type"] = "comparative"
                analysis["requires_multiple_sources"] = True
            
            # Complexity assessment
            word_count = len(query.split())
            if word_count > 10:
                analysis["complexity"] = "high"
            elif word_count < 5:
                analysis["complexity"] = "low"
            
            state["analysis"] = analysis
            return state
        
        def rewrite_query(state: dict) -> dict:
            """Rewrite query for better retrieval using agentic strategies"""
            original_query = state["query"]
            attempt = state["attempt"]
            
            strategies = [
                f"What specific information about {original_query}?",
                f"Details regarding: {original_query}",
                f"Explain the context of: {original_query}",
                f"Provide examples related to: {original_query}",
                f"Key concepts in: {original_query}",
                f"Summary of: {original_query}"
            ]
            
            if attempt < len(strategies):
                state["query"] = strategies[attempt]
            else:
                # Fallback: extract key terms and reformulate
                key_terms = re.findall(r'\b\w{4,}\b', original_query.lower())
                if key_terms:
                    state["query"] = " ".join(key_terms[:3])
            
            state["strategy"] = f"rewrite_attempt_{attempt}"
            return state
        
        def retrieve_chunks(state: dict) -> dict:
            """Retrieve relevant document chunks"""
            # Dynamic retrieval parameters based on query analysis
            k_value = 7 if state["analysis"].get("requires_multiple_sources", False) else 5
            min_sim = 0.15 if state["attempt"] > 1 else 0.25
            
            # Retrieve relevant chunks
            state["chunks"] = self.vector_store.search(
                state["query"], 
                k=k_value,
                min_similarity=min_sim
            )
            return state
        
        def generate_answer(state: dict) -> dict:
            """Generate answer using Gemini model"""
            retrieved_chunks = [chunk for chunk, _ in state["chunks"]]
            
            # Prepare context
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(
                    f"[Document {i}: {chunk.source}, Page {chunk.page}]\n{chunk.content}\n"
                )
            
            context = "\n" + "="*50 + "\n".join(context_parts)
            
            # Enhanced prompt
            prompt = f"""
            You are an expert document analyst. Provide accurate answers based STRICTLY on the context:
            
            ANALYSIS CONTEXT:
            - Query Type: {state["analysis"].get('query_type', 'unknown')}
            - Complexity: {state["analysis"].get('complexity', 'medium')}
            
            CRITICAL INSTRUCTIONS:
            1. Base your answer ONLY on the provided context
            2. Cite specific documents when making claims
            3. Provide detailed, well-structured responses
            4. Acknowledge uncertainty when appropriate
            
            DOCUMENT CONTEXT:
            {context}
            
            USER QUESTION: {state["query"]}
            
            COMPREHENSIVE ANSWER:
            """
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                state["answer"] = response.text.strip()
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                state["answer"] = f"Error generating answer: {str(e)}"
            
            return state
        
        def evaluate_confidence(state: dict) -> dict:
            """Evaluate confidence in the generated answer"""
            answer = state["answer"]
            retrieved_chunks = state["chunks"]
            
            if not retrieved_chunks or not answer:
                state["confidence"] = 0.0
                return state
            
            # Simple heuristics for confidence scoring
            confidence_factors = []
            
            # Factor 1: Number of supporting chunks
            chunk_factor = min(len(retrieved_chunks) / 3.0, 1.0)
            confidence_factors.append(chunk_factor)
            
            # Factor 2: Answer length and detail
            length_factor = min(len(answer.split()) / 100.0, 1.0)
            confidence_factors.append(length_factor)
            
            # Factor 3: Presence of specific details
            detail_factor = 0.0
            if any(keyword in answer.lower() for keyword in ['specific', 'according to', 'states that', 'mentions']):
                detail_factor = 0.8
            confidence_factors.append(detail_factor)
            
            # Factor 4: No hedging language
            hedging_words = ['might', 'could', 'possibly', 'perhaps', 'unclear', 'ambiguous']
            hedging_factor = 1.0 - (sum(1 for word in hedging_words if word in answer.lower()) / len(hedging_words))
            confidence_factors.append(hedging_factor)
            
            state["confidence"] = sum(confidence_factors) / len(confidence_factors)
            return state
        
        def should_retry(state: dict) -> str:
            """Decision function for whether to retry"""
            if state["confidence"] >= self.confidence_threshold:
                return "end"
            if state["attempt"] < self.max_retry_attempts - 1:
                return "rewrite"
            return "final_attempt"
        
        def final_attempt(state: dict) -> dict:
            """Final attempt with broader search parameters"""
            state["attempt"] = self.max_retry_attempts - 1
            state["strategy"] = "final_attempt"
            
            # Broader search
            state["chunks"] = self.vector_store.search(
                state["query"], k=10, min_similarity=0.1
            )
            return state
        
        # Build the graph
        builder = StateGraph(dict)
        
        # Add nodes
        builder.add_node("analyze", analyze_query)
        builder.add_node("rewrite", rewrite_query)
        builder.add_node("retrieve", retrieve_chunks)
        builder.add_node("generate", generate_answer)
        builder.add_node("evaluate", evaluate_confidence)
        builder.add_node("final", final_attempt)
        
        # Set entry point
        builder.set_entry_point("analyze")
        
        # Add edges
        builder.add_edge("analyze", "retrieve")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "evaluate")
        builder.add_edge("final", "generate")
        
        # Add conditional edges
        builder.add_conditional_edges(
            "evaluate",
            should_retry,
            {
                "end": END,
                "rewrite": "rewrite",
                "final_attempt": "final"
            }
        )
        
        return builder.compile()
    
    def load_documents(self, uploaded_files: List[UploadFile]) -> bool:
        """Load and process documents"""
        try:
            processor = DocumentProcessor()
            chunks = processor.process_documents(uploaded_files)
            
            if not chunks:
                return False
            
            self.vector_store.add_documents(chunks)
            
            # Generate document summaries
            self._generate_document_summaries()
            
            return True
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def _generate_document_summaries(self):
        """Generate summaries for each document"""
        doc_contents = {}
        
        # Group chunks by document
        for chunk in self.vector_store.chunks:
            if chunk.source not in doc_contents:
                doc_contents[chunk.source] = []
            doc_contents[chunk.source].append(chunk.content)
        
        # Generate summaries
        for doc_name, contents in doc_contents.items():
            combined_content = "\n".join(contents[:5])  # Use first 5 chunks for summary
            
            prompt = f"""
            Analyze this document content and provide a concise summary (2-3 sentences) 
            that captures the main topics and themes:
            
            {combined_content[:2000]}
            
            Summary:
            """
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                self.memory.document_summaries[doc_name] = response.text.strip()
            except Exception as e:
                logger.error(f"Error generating summary for {doc_name}: {e}")
                self.memory.document_summaries[doc_name] = "Summary unavailable"
    
    def generate_sample_questions(self, num_questions: int = 5) -> List[str]:
        """Generate sample questions based on loaded documents"""
        if not self.vector_store.chunks:
            return []
        
        # Get diverse content samples
        sample_contents = []
        step = max(1, len(self.vector_store.chunks) // num_questions)
        for i in range(0, len(self.vector_store.chunks), step):
            sample_contents.append(self.vector_store.chunks[i].content[:500])
        
        combined_content = "\n\n---\n\n".join(sample_contents)
        
        prompt = f"""
        Based on the following document content, generate {num_questions} diverse questions:
        
        Content:
        {combined_content[:3000]}
        
        Generate exactly {num_questions} questions, one per line, without numbering:
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            questions = [q.strip() for q in response.text.split('\n') if q.strip()]
            return questions[:num_questions]
        except Exception as e:
            logger.error(f"Error generating sample questions: {e}")
            return [
                "What are the main topics covered in the documents?",
                "Can you summarize the key points?",
                "What are the most important findings or conclusions?"
            ]
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a query using agentic retrieval and generation with LangGraph"""
        start_time = time.time()
        
        # Create initial state
        initial_state = {
            "query": query,
            "attempt": 0,
            "confidence": 0.0,
            "answer": "",
            "chunks": [],
            "strategy": "initial",
            "analysis": {}
        }
        
        # Execute the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Prepare citations
        citations = []
        for i, (chunk, score) in enumerate(final_state["chunks"], 1):
            citations.append({
                "id": i,
                "source": chunk.source,
                "page": chunk.page,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "relevance_score": score
            })
        
        # Add to memory
        self.memory.add_interaction(
            query,
            final_state["answer"],
            final_state["confidence"],
            [chunk for chunk, _ in final_state["chunks"]]
        )
        
        return {
            "answer": final_state["answer"],
            "confidence": final_state["confidence"],
            "citations": citations,
            "processing_time": time.time() - start_time,
            "attempts": final_state["attempt"] + 1,
            "agent_strategy": final_state["strategy"],
            "query_analysis": final_state["analysis"]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status with agentic insights"""
        doc_summary = self.vector_store.get_document_summary()
        
        # Analyze conversation patterns
        successful_patterns = {}
        for retrieval in self.memory.successful_retrievals:
            strategy = retrieval.retrieval_strategy
            if strategy in successful_patterns:
                successful_patterns[strategy] += 1
            else:
                successful_patterns[strategy] = 1
        
        return {
            "documents_loaded": len(doc_summary),
            "total_chunks": len(self.vector_store.chunks),
            "document_summary": doc_summary,
            "conversation_history_length": len(self.memory.conversation_history),
            "memory_summaries": self.memory.document_summaries,
            "successful_retrievals": len(self.memory.successful_retrievals),
            "successful_patterns": successful_patterns,
            "failed_queries": len(self.memory.failed_queries),
            "average_confidence": sum(h["confidence"] for h in self.memory.conversation_history) / len(self.memory.conversation_history) if self.memory.conversation_history else 0.0
        }

# FastAPI Endpoints

@app.post("/initialize")
async def initialize(api_key: str = Form(...)):
    """Initialize the RAG system"""
    try:
        SYSTEM_STATE["rag_system"] = AgenticRAG(api_key)
        SYSTEM_STATE["documents_loaded"] = False
        SYSTEM_STATE["sample_questions"] = []
        SYSTEM_STATE["chat_history"] = []
        SYSTEM_STATE["processing_status"] = "System initialized successfully"
        return JSONResponse(
            content={"status": "success", "message": "Agentic RAG system initialized"},
            status_code=200
        )
    except Exception as e:
        SYSTEM_STATE["processing_status"] = f"Initialization failed: {str(e)}"
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    if not SYSTEM_STATE.get("rag_system"):
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        SYSTEM_STATE["processing_status"] = "Processing documents..."
        success = SYSTEM_STATE["rag_system"].load_documents(files)
        
        if success:
            SYSTEM_STATE["documents_loaded"] = True
            SYSTEM_STATE["processing_status"] = "Documents processed successfully"
            SYSTEM_STATE["sample_questions"] = SYSTEM_STATE["rag_system"].generate_sample_questions()
            return JSONResponse(
                content={"status": "success", "message": "Documents processed successfully"},
                status_code=200
            )
        else:
            SYSTEM_STATE["processing_status"] = "Document processing failed"
            raise HTTPException(status_code=500, detail="Failed to process documents")
    except Exception as e:
        SYSTEM_STATE["processing_status"] = f"Document processing error: {str(e)}"
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status"""
    if not SYSTEM_STATE.get("rag_system"):
        return JSONResponse(
            content={"status": "uninitialized", "message": "System not initialized"},
            status_code=200
        )
    
    status = {
        "documents_loaded": SYSTEM_STATE["documents_loaded"],
        "processing_status": SYSTEM_STATE["processing_status"],
        "sample_questions_count": len(SYSTEM_STATE["sample_questions"]),
        "chat_history_count": len(SYSTEM_STATE["chat_history"])
    }
    
    if SYSTEM_STATE["documents_loaded"]:
        rag_status = SYSTEM_STATE["rag_system"].get_system_status()
        status.update(rag_status)
    
    return JSONResponse(content=status, status_code=200)

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions"""
    if not SYSTEM_STATE["documents_loaded"]:
        raise HTTPException(status_code=400, detail="Documents not loaded")
    
    return JSONResponse(
        content={"sample_questions": SYSTEM_STATE["sample_questions"]},
        status_code=200
    )

@app.post("/query")
async def answer_query(query: str = Form(...)):
    """Answer a query"""
    if not SYSTEM_STATE["documents_loaded"]:
        raise HTTPException(status_code=400, detail="Documents not loaded")
    
    try:
        logger.info(f"Received query: {query}")
        result = SYSTEM_STATE["rag_system"].answer_query(query)
        
        # Add to chat history
        chat_entry = {
            "question": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        SYSTEM_STATE["chat_history"].append(chat_entry)
        
        logger.info(f"Query answered. Confidence: {result.get('confidence')}, Time: {result.get('processing_time')}")
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/chat-history")
async def get_chat_history():
    """Get chat history"""
    return JSONResponse(
        content={"chat_history": SYSTEM_STATE["chat_history"]},
        status_code=200
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
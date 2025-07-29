import streamlit as st
import requests
import time

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.3rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .processing-status {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sample-question {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .sample-question:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# FastAPI base URL
FASTAPI_URL = "http://localhost:8000"

def initialize_system(api_key: str):
    """Initialize the RAG system"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/initialize",
            data={"api_key": api_key}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False

def upload_documents(files):
    """Upload documents to backend"""
    try:
        files = [("files", (file.name, file, file.type)) for file in files]
        response = requests.post(
            f"{FASTAPI_URL}/upload",
            files=files
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

def get_status():
    """Get system status"""
    try:
        response = requests.get(f"{FASTAPI_URL}/status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_sample_questions():
    """Get sample questions"""
    try:
        response = requests.get(f"{FASTAPI_URL}/sample-questions")
        if response.status_code == 200:
            return response.json().get("sample_questions", [])
        return []
    except:
        return []

def answer_query(query: str):
    """Send query to backend"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/query",
            data={"query": query}
        )
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            st.error(f"Backend error: {error_detail}")
            return None
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None

def get_chat_history():
    """Get chat history"""
    try:
        response = requests.get(f"{FASTAPI_URL}/chat-history")
        if response.status_code == 200:
            return response.json().get("chat_history", [])
        return []
    except:
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Document Analysis with Self-Healing Retrieval & Adaptive Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">🔧 Configuration</h2>', unsafe_allow_html=True)
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Enter your API key here..."
        )
        
        if api_key:
            if st.button("Initialize System", use_container_width=True):
                with st.spinner("Initializing..."):
                    if initialize_system(api_key):
                        st.success("System initialized!")
                    else:
                        st.error("Initialization failed")
        
        st.markdown("---")
        
        # Document upload section
        st.markdown('<h3 class="sidebar-header">📄 Document Upload</h3>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose documents to analyze",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )
        
        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
                with st.spinner("Processing..."):
                    if upload_documents(uploaded_files):
                        st.success("Documents processed!")
                    else:
                        st.error("Processing failed")
        
        # System status
        st.markdown("---")
        st.markdown('<h3 class="sidebar-header">📊 System Status</h3>', unsafe_allow_html=True)
        
        if st.button("Check Status", use_container_width=True):
            status = get_status()
            if status:
                if status.get("status") == "uninitialized":
                    st.warning("System not initialized")
                else:
                    st.metric("Documents Loaded", status.get("documents_loaded", False))
                    st.metric("Processing Status", status.get("processing_status", "Unknown"))
                    st.metric("Sample Questions", len(status.get("sample_questions", [])))
                    st.metric("Chat History", len(status.get("chat_history", [])))
            else:
                st.error("Failed to get status")
    
    # Main content
    status = get_status()
    if status and status.get("documents_loaded"):
        st.markdown("### 💡 Sample Questions")
        
        # Display sample questions
        sample_questions = get_sample_questions()
        if sample_questions:
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                with cols[i % 2]:
                    st.markdown(
                        f'<div class="sample-question" onclick="document.getElementById(\'question_input\').value=\'{question}\'">{question}</div>', 
                        unsafe_allow_html=True
                    )
    
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    user_question = st.text_input(
        "Enter your question:",
        key="question_input"
    )
    
    if user_question:
        if st.button("Get Answer", use_container_width=True):
            with st.spinner("Thinking..."):
                result = answer_query(user_question)
                if result:
                    st.session_state.last_result = result
                    st.rerun()
    
    # Display last result if available
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        st.markdown("### 🤖 Response")
        
        # Confidence indicator
        confidence = result['confidence']
        if confidence >= 0.8:
            confidence_class = "confidence-high"
            confidence_icon = "🟢"
        elif confidence >= 0.5:
            confidence_class = "confidence-medium"
            confidence_icon = "🟡"
        else:
            confidence_class = "confidence-low"
            confidence_icon = "🔴"
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(result['answer'])
        with col2:
            st.markdown(f'{confidence_icon} <span class="{confidence_class}">Confidence: {confidence:.1%}</span>', 
                       unsafe_allow_html=True)
            st.markdown(f"⏱️ {result['processing_time']:.1f}s")
            st.markdown(f"🔄 {result['attempts']} attempts")
        
        # Citations
        if result['citations']:
            with st.expander(f"📚 View {len(result['citations'])} Citations"):
                for citation in result['citations']:
                    st.markdown(f"""
                    <div class="citation-box">
                        <strong>📄 {citation['source']}</strong> (Page {citation['page']})
                        <br>
                        <em>Relevance Score: {citation['relevance_score']:.2f}</em>
                        <br><br>
                        {citation['content_preview']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat history
    chat_history = get_chat_history()
    if chat_history:
        st.markdown("### 📝 Conversation History")
        for chat in reversed(chat_history):
            with st.expander(f"Q: {chat['question']}", expanded=False):
                result = chat['result']
                st.markdown(f"**A:** {result['answer']}")
                st.caption(f"Confidence: {result['confidence']:.1%} | Time: {result['processing_time']:.1f}s")

if __name__ == "__main__":
    main()
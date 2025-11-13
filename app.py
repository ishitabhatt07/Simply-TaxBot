import streamlit as st
import os

# Core LangChain imports - Updated for latest version
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Document loading and processing
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Core components
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Agent imports
try:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
except ImportError:
    st.error("Please install: pip install langchain langchain-community")
    st.stop()

# Search
from duckduckgo_search import DDGS

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ITR Help Bot",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stChatInputContainer {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 25px;
        padding: 10px;
    }

    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-weight: 700;
    }

    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
LLM_MODEL = "mistral"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DB_PATH = "./lc_chroma_db"
DATA_PATH = "./data"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    st.markdown("#### 📊 System Status")
    if os.path.exists(DB_PATH):
        st.success("✅ Database: Ready")
    else:
        st.warning("⚠️ Database: Not initialized")

    if os.path.exists(DATA_PATH):
        pdf_count = len([f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')])
        st.info(f"📄 Documents: {pdf_count} PDFs found")
    else:
        st.error("❌ Data folder not found")

    st.markdown("---")
    st.markdown("#### 🤖 Model Info")
    st.text(f"LLM: {LLM_MODEL}")
    st.text(f"Embeddings: {EMBED_MODEL_NAME.split('/')[-1]}")

    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown("""
    - Ask about ITR forms
    - Query tax deductions
    - Check filing deadlines
    - Get live updates
    """)

    if st.button("🔄 Reset Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi! I'm your ITR Filing Assistant. How can I help?"
        }]
        st.rerun()

# --- MAIN TITLE ---
st.markdown("<h1 style='text-align: center;'>🇮🇳 ITR Filing Help Bot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: white; font-size: 18px;'>Your intelligent assistant for Indian Income Tax Returns</p>",
    unsafe_allow_html=True)
st.markdown("---")


# --- INITIALIZE LLM AND EMBEDDINGS ---
@st.cache_resource(show_spinner="🔧 Initializing models...")
def initialize_models():
    try:
        llm = OllamaLLM(model=LLM_MODEL)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        st.stop()


llm, embeddings = initialize_models()


# --- SIMPLE RAG CHAIN (WITHOUT create_retrieval_chain) ---
@st.cache_resource(show_spinner="📚 Setting up document search...")
def setup_rag_tool(_llm, _embeddings):
    try:
        # Check if database exists
        if not os.path.exists(DB_PATH):
            with st.spinner("🔨 Building vector database..."):
                if not os.path.exists(DATA_PATH):
                    st.error(f"❌ Data directory '{DATA_PATH}' not found.")
                    st.stop()

                loader = PyPDFDirectoryLoader(DATA_PATH)
                documents = loader.load()

                if not documents:
                    st.error("❌ No PDF documents found.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)

                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=_embeddings,
                    persist_directory=DB_PATH
                )
                st.success("✅ Vector database created!")

        # Load database
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=_embeddings
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Create a simple RAG function without complex chains
        def rag_query(query: str) -> str:
            """Query the RAG system"""
            try:
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)

                # Format context
                context = "\n\n".join([doc.page_content for doc in docs])

                # Create prompt
                prompt = f"""You are a helpful assistant specializing in Indian Income Tax Returns (ITR).
Use the following context to answer the question accurately and concisely.
If you don't know the answer based on the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

                # Get response from LLM
                response = _llm.invoke(prompt)
                return response
            except Exception as e:
                return f"Error processing query: {str(e)}"

        # Create tool
        rag_tool = Tool(
            name="ITR_Manual_Search",
            func=rag_query,
            description=(
                "Use this tool to answer questions about Indian ITR filing, "
                "tax deductions (80C, 80D, etc.), tax regimes (old vs new), "
                "ITR forms (ITR-1, ITR-2, etc.), and other tax-related queries. "
                "This searches official ITR manuals and documentation."
            )
        )

        return rag_tool
    except Exception as e:
        st.error(f"Error setting up RAG tool: {e}")
        return None


# --- WEB SEARCH TOOL ---
def web_search(query: str) -> str:
    """Perform web search using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                formatted_results = "\n\n".join([
                    f"**{r['title']}**\n{r['body']}\nSource: {r['href']}"
                    for r in results
                ])
                return formatted_results
            return "No results found."
    except Exception as e:
        return f"Web search error: {str(e)}"


web_search_tool = Tool(
    name="Live_Web_Search",
    func=web_search,
    description=(
        "Use this tool for current, live information such as tax deadlines, "
        "recent policy changes, or any time-sensitive queries. "
        "Good for questions like 'What is the ITR filing deadline this year?' "
        "or 'Was the tax deadline extended?'"
    )
)


# --- CREATE AGENT ---
@st.cache_resource(show_spinner="🧠 Creating intelligent agent...")
def create_tax_agent(_llm, _rag_tool, _web_tool):
    try:
        tools = [_rag_tool, _web_tool]

        # Get ReAct prompt
        agent_prompt = hub.pull("hwchase17/react")

        # Create agent
        agent = create_react_agent(
            llm=_llm,
            tools=tools,
            prompt=agent_prompt
        )

        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        return agent_executor
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        return None


# Initialize tools and agent
rag_tool = setup_rag_tool(llm, embeddings)
if rag_tool:
    agent_executor = create_tax_agent(llm, rag_tool, web_search_tool)
else:
    st.error("Failed to initialize the system.")
    st.stop()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! 👋 I'm your ITR Filing Assistant. I can help you with:\n\n✅ Tax deductions and exemptions\n✅ ITR form selection\n✅ Filing procedures\n✅ Current deadlines and updates\n\nWhat would you like to know?"
    }]

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 Ask me anything about ITR filing..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching and analyzing..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                error_msg = f"⚠️ I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: white; font-size: 14px;'>Made with ❤️ for Indian Taxpayers | Powered by LangChain & Ollama</p>",
    unsafe_allow_html=True
)

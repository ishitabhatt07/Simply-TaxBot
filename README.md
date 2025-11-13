# Simply TaxBot - Agentic ITR Filing Help Bot

An intelligent chatbot that helps answer questions about Indian Income Tax Returns (ITR) filing using AI agents, RAG (Retrieval-Augmented Generation), and live web search.

## 🎯 What Does It Do?

This bot can:
- **Search official ITR manuals** stored locally to answer questions about tax deductions (80C, 80D), tax regimes, and ITR forms
- **Search the web** for live, up-to-date information like filing deadlines and recent tax changes
- **Automatically decide** which tool to use based on your question

## 🧠 How It Works

The bot uses an **agentic approach** with two main tools:

1. **ITR Manual Search Tool**: Uses RAG to search through PDF documents stored in the `data/` folder
   - Documents are split into chunks and stored in a vector database (ChromaDB)
   - Retrieves relevant information based on your query

2. **Live Web Search Tool**: Uses DuckDuckGo to fetch current information from the internet
   - Useful for questions about deadlines, recent updates, or general queries

The AI agent (powered by Ollama + Mistral) decides which tool to use based on your question!

## 📁 About the Data Folder

The `data/` folder contains official ITR guides and FAQs from the Income Tax Department of India:
- `80.deductions-or-allowances-allowed-to-salaried-employee.pdf`
- `DeductionsunderChapterVIA.pdf`
- `FAQs on New Tax vs Old Tax Regime - Income Tax Department.pdf`
- `File ITR-1 (Sahaj) Online - FAQs.pdf`
- `File ITR-1 (Sahaj) Online User Manual.pdf`
- `File ITR-2 Online User Manual.pdf`
- `ITR-2 FAQ - Income Tax Department.pdf`

**Note**: These are publicly available government documents. You can add your own ITR-related PDFs to this folder, and the bot will automatically index them on the next run.

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed with the Mistral model

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ishitabhatt07/Simply-TaxBot.git
cd Simply-TaxBot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running with the Mistral model:
```bash
ollama pull mistral
ollama run mistral
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## 💡 Usage Examples

**Question about tax deductions:**
> "What is Section 80C?"

**Question about current deadlines:**
> "What is the ITR filing deadline for this year?"

**Question about forms:**
> "What is the difference between ITR-1 and ITR-2?"

## 🛠️ Tech Stack

- **LangChain**: Framework for building LLM applications
- **Ollama**: Running local Mistral LLM
- **ChromaDB**: Vector database for document storage
- **Streamlit**: Web interface
- **HuggingFace Embeddings**: Text embeddings for RAG
- **DuckDuckGo Search**: Live web search

## 📝 License

This project uses publicly available government documents from the Income Tax Department of India.

## 🤝 Contributing

Feel free to open issues or submit pull requests if you'd like to improve the bot!

---

Made with ❤️ for Indian taxpayers

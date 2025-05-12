# RAG-Powered Multi-Agent Q&A Assistant

A Retrieval-Augmented Generation (RAG) based assistant that intelligently routes user queries to appropriate tools—Calculator, Dictionary, or Document Retrieval—using Google's Gemini model for natural language responses. The system is accessible via a Streamlit web interface.

---

## Architecture Overview
### Data Ingestion & Embedding
- **Documents:** Loads text files (policy.txt, faq.txt, specs.txt) from the data/ directory.
- **Chunking:** Splits documents into semantically coherent chunks.
- **Embedding:** Utilizes Google's text-embedding-004 model to generate vector embeddings.
- **Vector Store:** Stores embeddings in a FAISS index for efficient similarity search.

### Agentic Workflow
- **Query Routing:** Analyzes user queries to determine the appropriate processing path:
  - Calculator Tool: Handles queries containing the keyword "calculate".
  - Dictionary Tool: Handles queries containing the keyword "define".
  - RAG Pipeline: Processes all other queries using the document retrieval and generation pipeline.
- **Decision Logging:** Logs each decision step for transparency.

### Answer Generation
- **Calculator**: Evaluates mathematical expressions using Python's eval function.
- **Dictionary:** Fetches definitions from the Free Dictionary API.
- **RAG Pipeline:** Retrieves relevant document chunks and generates answers using Google's gemini-1.5-flash model.

---

## Key Design Choices
- **Modular Design:** Separates concerns into distinct components—data ingestion, embedding, retrieval, agent routing, and UI—for maintainability and scalability.
- **FAISS for Vector Storage:** Chosen for its efficiency in handling similarity searches over dense vector spaces, suitable for medium-sized document collections.
- **Google's Gemini Model:** Selected for its strong reasoning capabilities and cost-effectiveness compared to other LLMs.
- **Keyword-Based Routing:** Implements a simple yet effective method for directing queries to the appropriate processing tool.

---

## Installation & Setup
### Prerequisites
- Python 3.8 or higher
- Google API Key with access to Gemini models

### Steps to Run Locally
#### Clone the Repository
```bash
git clone <repository-url>
cd QnA_RAG
```
#### Create Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```
#### Set Up Environment Variables (.env file)
```
GOOGLE_API_KEY=<your_google_api_key_here>
```
#### Run the Application
```
streamlit run app.py
```

---

- PFA-
![image](https://github.com/user-attachments/assets/b982ec2e-8379-4a30-8e86-20ed9081723b)


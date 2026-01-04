# Local-First Technical Documentation RAG
A high-performance Retrieval-Augmented Generation (RAG) system for querying technical datasheets locally.

## Project Overview
This application allows users to interact with technical PDF documentation through a conversational AI interface. By utilizing a local LLM and Vector Database, it ensures 100% data privacy and zero API latency or costs.

## Technical Stack
* LLM Engine: Ollama (Llama 3 8B)
* Orchestration: LangChain (LCEL)
* Vector Database: ChromaDB
* Frontend: Streamlit
* Language: Python 3.10+

## Setup Instructions

### 1. Prerequisites (Ollama Setup)
First, ensure you have Ollama installed and the Llama 3 model pulled to your local machine:
1. Download Ollama from https://ollama.com.
2. Open your terminal and run:
   ```bash
   ollama pull llama3

### 2. Environment Setup 
Clone this repository and create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

### 3. Data Folder
Create a data folder in the root directory of the repo

### 4. Install Dependencies
Install the required Python packages using the minimalist requirements file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Ingest Data
Place your technical PDF (e.g., **datasheet.pdf**) inside the `/data` folder. Then, run the ingestion script to build the vector database:
   ```bash
   python ingest.py
   ```
Note: This will create a local folder called chroma_langchain_db.

### Step 2: Launch the Assistant
Start the Streamlit web interface:
   ```bash
   streamlit run app.py
   ```
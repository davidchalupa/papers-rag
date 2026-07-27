# papers-rag

## Overview
The `papers-rag` repository is designed to facilitate the processing and analysis of research papers using
a Retrieval-Augmented Generation (RAG) model. The primary focus is on extracting text chunks from PDFs,
building an index for efficient querying, and generating answers to user queries based on the indexed
data, specifically tailored for research paper analysis.

## Project Structure
```
papers-rag/
├── paper_rag_explorer.py
├── requirements.txt
├── test_paper_rag_explorer.py
├── docs/
├── downloaders/
```

## Key Components

### `paper_rag_explorer.py`
- **Functionality**: 
  - Extracts text chunks from PDFs, builds the index for efficient querying.
  - Provides 
- **Modes**:
  - default: Uses the Ministral-8B-Instruct-2410-Q4_K_M.gguf model.
  - `--fast`: Uses the rocket-3b.Q4_K_M.gguf model.

### `test_paper_rag_explorer.py`
- **Functionality**: Tests the `paper_rag_explorer.py` module.
- **Tests**:
  - `test_algorithm_abbreviation_extraction()`: Tests if the RAG can successfully extract the specific algorithm abbreviation.

### `downloaders/`
- **Functionality**: Scripts for downloading models from Hugging Face.

## Usage
### Downloading Models
Before running the RAG, the required models need to be downloaded by running:
```bash
python downloaders/download_gguf_ministral_8b.py
python downloaders/download_gguf_rocket_3b.py
```

### Running the Papers RAG Explorer
To extract text chunks from PDFs, run:
```bash
python paper_rag_explorer.py
```

### Running Tests
To run the tests, use:
```bash
pytest test_paper_rag_explorer.py
```

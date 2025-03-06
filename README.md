# LangChain Examples

This repository contains simple examples of using LangChain with Python.

## Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Examples

### Basic LangChain Example

This example demonstrates:
- Using LLMChain with a prompt template
- Creating a sequential chain

Run with:
```
python langchain_example.py
```

### Retrieval-Augmented Generation (RAG) Example

This example demonstrates:
- Document loading and text splitting
- Creating embeddings and a vector store
- Building a retrieval QA system

Run with:
```
python langchain_rag_example.py
```

### Agent Example

This example demonstrates:
- Creating an agent with tools
- Using the agent to answer questions requiring external tools
- Handling different types of queries (math, factual knowledge, reasoning)

Run with:
```
python langchain_agent_example.py
```

## Requirements

- Python 3.8+
- OpenAI API key 
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()


def main():
    # Sample text data (in a real scenario, you would load documents)
    with open("sample_data.txt", "w") as f:
        f.write(
            """
        LangChain is a framework for developing applications powered by language models.
        It enables applications that are context-aware, reason, and learn from feedback.
        
        LangChain provides modules for working with language models, prompt templates, 
        memory for storing conversation history, indexes for retrieving relevant context,
        agents that can use tools, and chains for combining multiple components.
        
        The framework is designed to be modular and extensible, allowing developers to 
        use only the components they need. It supports multiple LLM providers including 
        OpenAI, Anthropic, Google, and others.
        
        LangChain applications can be built for various use cases such as chatbots, 
        question answering, summarization, and more complex reasoning tasks.
        """
        )

    # Load documents
    loader = TextLoader("sample_data.txt")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=200, chunk_overlap=20, separator="\n"
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a simple in-memory vector store
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create a retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=True
    )

    # Ask questions
    questions = [
        "What is LangChain?",
        "What modules does LangChain provide?",
        "Which LLM providers does LangChain support?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")


if __name__ == "__main__":
    main()

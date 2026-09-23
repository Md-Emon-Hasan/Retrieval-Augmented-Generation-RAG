import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Google API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# Set the title of the Streamlit application
st.title("RAG Application using Gemini Pro")

# File uploader widget for PDF files
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()

    # Split the loaded PDF data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Display the number of document chunks
    st.write(f"Total number of document chunks: {len(docs)}")

    # Create a vector store from the document chunks using Google Generative AI Embeddings
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )

    # Create a retriever from the vector store for similarity search
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize the language model with specified parameters
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

    # Define the system prompt for the assistant
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # Create a prompt template for the chat
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Create the retrieval-augmented generation (RAG) chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Text input widget for user queries
    query = st.text_input("Ask me anything:")
    if query:
        # Generate a response using the RAG chain
        response = rag_chain.invoke({"input": query})
        # Retrieve the answer from the response
        answer = response.get("answer", "I'm sorry, I don't have an answer for that.")
        # Display the answer in the Streamlit app
        st.write(answer)

    # Clean up the temporary file after processing
    os.remove(temp_file_path)

# Import necessary libraries
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set the title of the Streamlit application
st.title("RAG App Demo")

# List of URLs to load documents from
urls = [
    "https://www.victoriaonmove.com.au/local-removalists.html",
    "https://victoriaonmove.com.au/index.html",
    "https://victoriaonmove.com.au/contact.html",
]

# Load documents from the specified URLs
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Split the loaded documents into smaller chunks of 1000 characters each
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Display the total number of document chunks created
st.write(f"Total number of documents: {len(docs)}")

# Initialize the embedding model using a pre-trained Hugging Face model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store (Chroma) from the document chunks and their embeddings
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

# Set up a retriever to fetch relevant documents based on similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Specify the Hugging Face model to use for text generation
model_name = "distilgpt2"

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline with the loaded model and tokenizer
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

# Wrap the Hugging Face pipeline in a LangChain-compatible interface
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the system prompt template for the assistant
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a prompt template combining the system prompt and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain that combines document retrieval and question answering
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Create an input field in the Streamlit app for user queries
query = st.text_input("Ask me anything:")

# If the user has entered a query, process it
if query:
    # Invoke the RAG chain with the user's query
    response = rag_chain.invoke({"input": query})
    # Display the assistant's answer in the Streamlit app
    st.write(response["answer"])

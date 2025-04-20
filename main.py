import os
import pickle
import time
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import nltk
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

nltk.download('punkt', download_dir='/Users/manas/nltk_data')
nltk.data.path.append('/Users/manas/nltk_data')
# Load environment variables
load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=500)
# Streamlit UI
st.title("New Research Tool")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():  # Validate non-empty input
        urls.append(url)


file_path="faiss_store_embedding.pkl"
# Process URLs
process_url_clicked = st.sidebar.button("Process URLs")
content_loading= st.empty()
if process_url_clicked:
    if urls:  # Ensure there are valid URLs
        try:
            content_loading.text("Data is loading ....")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            content_loading.text("Data splitting started ....")
            text_splitter= RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs= text_splitter.split_documents(data)
            content_loading.text("Creation of Embedding started ....")
            embeddings= OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore_openai= FAISS.from_documents(docs,embeddings)
            time.sleep(2)
            #save it to local pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)



        except Exception as e:
            st.error(f"Error processing URLs: {e}")
    else:
        st.error("No valid URLs provided. Please enter at least one URL.")


query = content_loading.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # RetrievalQA instead of RetrievalQAWithSourcesChain
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
            result_response = chain.run(query)
            st.header("Answer")
            st.subheader(result_response)



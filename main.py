# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain import VectorDBQA
from langchain import OpenAI

import pinecone

pinecone.init(
    api_key="30305c98-ca66-47e4-94db-5601639c7531", environment="us-west4-gcp-free"
)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("PyCharm")
    loader = TextLoader(
        "E:\\Code\\vectordb\\mediumblogs\\mediumblog1.txt", encoding="utf8"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True
    )
    query = "what is a vector DB? Give me a 20 word answer for a beginner"
    result = qa({"query": query})
    print(result)
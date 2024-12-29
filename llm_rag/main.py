import sys
import os
import glob
import requests

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_rag.core.embeddings.constants import SENTENCE_TRANSFORMER_ALL_MPNET
from llm_rag.core.embeddings.embedding_factory import EmbeddingFactory
from utils.file import get_absolute_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
# from docx import Document as DocxDocument
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Document loader 
# Load documents from the text folder

# Load all text files in the text folder
import os
# Function to read all text files from a folder
def read_text_files(folder_path):
    text_files = glob.glob(os.path.join(folder_path, '*.txt'))
    documents = []
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(content)
    return documents

def get_pdf_files_path(folder_path):
    files = glob.glob(os.path.join(folder_path, '*.pdf'))
    return files

def get_files_path(folder_path, extension):
    files = glob.glob(os.path.join(folder_path, f'*.{extension}'))
    return files

def load_html_from_link(link):
    response = requests.get(link)
    response.raise_for_status()  # Raise an error for bad status codes
    html_content = response.text
    return html_content

def create_documents_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return [Document(page_content=text)]

def read_docx_files(folder_path):
    docx_files = glob.glob(os.path.join(folder_path, '*.docx'))
    documents = []
    for file_path in docx_files:
        loader = Docx2txtLoader(file_path)
        docx_docs = loader.load()
        documents.extend(docx_docs)
    return documents

# Example usage
documents = read_text_files(get_absolute_path(current_dir, './data_source/text'))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

split_text_documents = []
# Print the content of each document
for i, doc in enumerate(documents):
    documents = text_splitter.create_documents([doc])    
    split_text_documents.extend(documents)

# Load PDF Files
pdf_documents = []
split_pdf_documents = []
pdf_files = get_pdf_files_path(get_absolute_path(current_dir, './data_source/pdf'))
for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    pdf_documents.extend(pages)

for i, doc in enumerate(pdf_documents):
    split_docs = text_splitter.create_documents([doc.page_content])
    split_pdf_documents.extend(split_docs)

# Webite loader
# read all links from file
website_links = []
website_file_links = get_files_path(get_absolute_path(current_dir, './data_source/website'), 'txt')
for file_path in website_file_links:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            website_links.append(line.strip())

# Load HTML content from links
html_documents = []
split_html_documents = []
for link in website_links:
    html_content = load_html_from_link(link)
    html_docs = create_documents_from_html(html_content)
    html_documents.extend(html_docs)

for doc in html_documents:
    split_docs = text_splitter.create_documents([doc.page_content])
    split_html_documents.extend(split_docs)

# Load Docx files
split_docx_documents = []
docx_documents = read_docx_files(get_absolute_path(current_dir, './data_source/docx'))
# Split the content of each docx document and concatenate the results
for i, doc in enumerate(docx_documents):
    split_docs = text_splitter.create_documents([doc.page_content])
    split_docx_documents.extend(split_docs)


embedding = EmbeddingFactory.create_embedding(SENTENCE_TRANSFORMER_ALL_MPNET)
print(f'split_text_documents: {len(split_text_documents)}\nsplit_pdf_documents: {len(split_pdf_documents)}\nsplit_html_documents: {len(split_html_documents)}\nsplit_docx_documents: {len(split_docx_documents)}')
vectorstore = InMemoryVectorStore.from_documents(
    split_text_documents + split_pdf_documents + split_html_documents + split_docx_documents,
    embedding=embedding,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

llm = ChatOpenAI(model="gpt-4o-mini")
message = """
    Answer this question using the provided context only.
    {question}
    Context:
    {context}
    """
prompt = ChatPromptTemplate.from_messages([("human", message)])
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

def main():
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        response = rag_chain.invoke(question);
        print("Answer:", response.content)
main()

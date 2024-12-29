import sys
import os
import glob

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

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

# Example usage
documents = read_text_files(get_absolute_path(current_dir, './data_source/text'))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

text_documents = []
# Print the content of each document
for i, doc in enumerate(documents):
    documents = text_splitter.create_documents([doc])    
    text_documents.extend(documents)

embedding = EmbeddingFactory.create_embedding(SENTENCE_TRANSFORMER_ALL_MPNET)
vectorstore = InMemoryVectorStore.from_documents(
    text_documents,
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

response = rag_chain.invoke("How much does FPT invest to AI factory?")

print(response.content)
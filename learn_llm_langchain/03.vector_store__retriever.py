import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from llm_rag.core.embeddings.constants import (
    SENTENCE_TRANSFORMER_MINILM,
    SENTENCE_TRANSFORMER_BERT,
    SENTENCE_TRANSFORMER_ALL_MPNET
)
from llm_rag.core.embeddings.embedding_factory import EmbeddingFactory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

embedding = EmbeddingFactory.create_embedding(SENTENCE_TRANSFORMER_ALL_MPNET)
# embedding = EmbeddingFactory.create_embedding(SENTENCE_TRANSFORMER_MINILM)
# embedding = EmbeddingFactory.create_embedding(SENTENCE_TRANSFORMER_BERT)
# embedding = EmbeddingFactory.create_embedding(OPENAI_EMBEDDING_SMALL) 
# Noticed: OpenAIEmbeddings is often got Rate Limit Exceeded error
vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding,
)
# print(embedding.embed_query('Hello World!'))

vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding,
)

# Perform similarity search
# result = vectorstore.similarity_search_with_score("friendliness")
# print(result)

# Retrieve the most similar document to a query
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
response = rag_chain.invoke("tell me about cats")

print(response.content)
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from llm_rag.core.embeddings.constants import (
    OPENAI_EMBEDDING_SMALL,
    OPENAI_EMBEDDING_LARGE,
    SENTENCE_TRANSFORMER_MINILM,
    SENTENCE_TRANSFORMER_BERT,
    SENTENCE_TRANSFORMER_ALL_MPNET
)

class EmbeddingFactory:
    @staticmethod
    def create_embedding(embedding_type):
        if embedding_type in [OPENAI_EMBEDDING_SMALL, OPENAI_EMBEDDING_LARGE]:
            return OpenAIEmbeddings(model=embedding_type)
        elif embedding_type in [SENTENCE_TRANSFORMER_ALL_MPNET]:
            return HuggingFaceEmbeddings(model_name=embedding_type)
        elif embedding_type in [SENTENCE_TRANSFORMER_MINILM, SENTENCE_TRANSFORMER_BERT]:
            return HuggingFaceEmbeddings(model_name=embedding_type)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
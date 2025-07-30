from pydantic_settings import BaseSettings


class ChromaSettings(BaseSettings):
    # Chroma DB settings
    chroma_host: str = "10.32.1.36"
    chroma_port: int = 9941
    allow_reset: bool = False
    
    # Documents collection's settings
    embedding_host: str = "10.32.1.36"
    embedding_port: int = 5002
    embedding_endpoint: str = "/embed"
    
    # Reranker settings
    reranker_host: str = "10.32.1.36"
    reranker_port: int = 5001
    reranker_endpoint: str = "/rerank"

settings = ChromaSettings()
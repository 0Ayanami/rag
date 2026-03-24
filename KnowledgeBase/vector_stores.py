from langchain_chroma import Chroma
import config_data as config
from langchain_ollama import OllamaEmbeddings

class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": config.similarity_threshold},
        )
    
if __name__ == "__main__":
    embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vector_store_service = VectorStoreService(embedding)
    retriever = vector_store_service.get_retriever()
    res = retriever.invoke("演艺+索道双轮驱动的文旅龙头")
    print(res)
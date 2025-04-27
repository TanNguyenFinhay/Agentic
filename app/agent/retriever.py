from app.clients.nvidia_nim_client import ExtractorClient, EmbedderClient, RerankerClient, GeneratorClient
from app.clients.cuvs_client import CuVSClient

class Retriever:
    def __init__(self):
        self.extractor = ExtractorClient()
        self.embedder = EmbedderClient()
        self.vectordb = CuVSClient()
        self.reranker = RerankerClient()
        self.generator = GeneratorClient()

    def retrieve_context(self, query: str) -> str:
        # documents = self.extractor.extract_documents()
        # embeddings = self.embedder.embed(documents)
        # self.vectordb.index(embeddings)

        query_emb = self.embedder.embed([query])
        candidates = self.vectordb.search(query_emb)

        reranked_docs = self.reranker.rerank(query, candidates)

        context = self.generator.generate_context(query, reranked_docs)
        return context

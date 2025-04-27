from openai import OpenAI
import os
# Wrapper for NVIDIA NIM models

class ReasonerClient:
    def __init__(self):
        self.endpoint = "http://your_nim_reasoner_endpoint"

    def generate(self, prompt: str) -> str:
        # call API to Llama 3.3
        return f"[Reasoned output for]: {prompt}"

class ExtractorClient:
    def extract_documents(self):
        return ["Document 1", "Document 2"]

class EmbedderClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("api_key")
        )

    def embed(self, query: str):
        # Gọi API của OpenAI để lấy embedding cho câu hỏi (query)
        response = self.client.embeddings.create(
            input=[query],  # Đưa truy vấn vào dưới dạng list
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        # Trả về embedding của truy vấn
        return response.data[0].embedding

class RerankerClient:
    def rerank(self, query, docs):
        return docs

class GeneratorClient:
    def generate_context(self, query, docs):
        return "Generated context"

# Test phần embedding
if __name__ == "__main__":
    embedder = EmbedderClient()

    # Nhận đầu vào từ người dùng
    query = input("Nhập câu truy vấn: ")

    # Lấy embedding cho câu truy vấn
    embedding = embedder.embed(query)
    
    # In kết quả embedding
    print("Embedding for the query:", embedding)
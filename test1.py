from openai import OpenAI
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import time
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Wrapper for NVIDIA NIM models
class ReasonerClient:
    def __init__(self):
        self.endpoint = "http://your_nim_reasoner_endpoint"

    def generate(self, prompt: str) -> str:
        return f"[Reasoned output for]: {prompt}"

class ExtractorClient:
    def extract_documents(self):
        return ["Document 1", "Document 2"]

class EmbedderClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def embed(self, query: str):
        response = self.client.embeddings.create(
            input=[query],
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            encoding_format="float",
            extra_body={
                "input_type": "query",
                "truncate": "NONE"
            }
        )
        return response.data[0].embedding

class RerankerClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
        self.session = requests.Session()

    def rerank(self, query: str, passages: list[str]):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        payload = {
            "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
        }

        response = self.session.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print("API Response:", result)

        # Sắp xếp kết quả dựa trên thứ tự ranking (index trả về là index trong passages)
        reranked = [(passages[item["index"]], item["logit"]) for item in result["rankings"]]
        return reranked


class GeneratorClient:
    def generate_context(self, query, docs):
        context = "\n".join(docs)  # Kết hợp các đoạn văn thành context
        return context

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    return text

def chunk_text(text: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

if __name__ == "__main__":
    file_path = "./hihi.pdf"
    api_key = ""
    api_key1 = ""

    print(f"Reading file {file_path}...")
    text = extract_text_from_pdf(file_path)
    print(f"Extracted {len(text)} characters.")

    print("Splitting text into chunks...")
    chunks = chunk_text(text)
    print(f"Generated {len(chunks)} chunks.")

    embedder = EmbedderClient(api_key)
    reranker = RerankerClient(api_key1)

    all_chunks = []
    all_embeddings = []

    for idx, chunk in enumerate(chunks):
        print(f"\nEmbedding chunk {idx + 1}/{len(chunks)} (length: {len(chunk)} characters)")
        embedding = embedder.embed(chunk)
        print(f"Embedding vector (first 5 values): {embedding[:5]} ... [len={len(embedding)}]")
        all_chunks.append(chunk)
        all_embeddings.append(embedding)
        time.sleep(0.2)

    print("\n=== Ready for Search ===")
    while True:
        query = input("\n🔍 Nhập truy vấn để tìm kiếm (hoặc gõ 'exit'): ")
        if query.lower() == "exit":
            break

        query_embedding = embedder.embed(query)
        scores = cosine_similarity([query_embedding], all_embeddings)[0]

        top_k = 5
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_k_chunks = [all_chunks[i] for i in top_indices]

        print("\n📌 Top K trước reranking (cosine):")
        for rank, idx in enumerate(top_indices):
            print(f"\n{rank + 1}. 🔹 Score: {scores[idx]:.4f}")
            print(f"📄 Đoạn {idx + 1}:\n{all_chunks[idx][:300]}...")

        print("\n🔁 Đang rerank lại kết quả với NVIDIA Reranker...")
        reranked = reranker.rerank(query, top_k_chunks)

        print("\n🏆 Top kết quả sau reranking:")
        for rank, (chunk, score) in enumerate(reranked):
            print(f"\n{rank + 1}. ⭐ Score: {score:.4f}")
            print(f"{chunk[:300]}...")

        context = GeneratorClient().generate_context(query, [r[0] for r in reranked])

        print("\n💡 Context generated for Reasoner/Generator:")
        print(f"\n{context[:1000]}...")

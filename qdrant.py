from openai import OpenAI
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import time
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from qdrant_client.http import models

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

    def get_embedding_dimension(self):
        """Get the actual dimension of embeddings from the model"""
        test_embedding = self.embed("test")
        return len(test_embedding)

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
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def generate_context(self, query: str, docs: list[str]) -> str:
        # Combine documents into a context string
        context = "\n".join(docs)
        
        # Build messages with instructions and query
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent AI assistant specialized in analyzing documents and answering questions based on their content."
            },
            {
                "role": "user",
                "content": f"{context}\n\nBased on the above information, please answer the following question:\n{query}"
            }
        ]

        # Make the chat completion request
        completion = self.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=messages,
            temperature=0.3,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

        # Collect and return streamed response
        result = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        return result

class QdrantVectorDB:
    def __init__(self, collection_name: str = "documents", embedder: EmbedderClient = None):
        """Khởi tạo Qdrant client."""
        self.client = QdrantClient("http://localhost:6333")
        self.collection_name = collection_name
        
        # Get actual embedding dimension from the model
        if embedder:
            print("Đang kiểm tra kích thước embedding...")
            self.embedding_size = embedder.get_embedding_dimension()
            print(f"Kích thước embedding thực tế: {self.embedding_size}")
        else:
            self.embedding_size = 2048  # Default based on error message
        
        # Tạo collection nếu chưa tồn tại
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Tạo collection nếu chưa tồn tại."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"Tạo collection mới: {self.collection_name} với kích thước vector: {self.embedding_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE)
                )
            else:
                print(f"Collection {self.collection_name} đã tồn tại")
                # Check if dimensions match
                info = self.client.get_collection(self.collection_name)
                existing_size = info.config.params.vectors.size
                if existing_size != self.embedding_size:
                    print(f"⚠️  CẢNH BÁO: Collection hiện tại có kích thước {existing_size}, nhưng model tạo embedding kích thước {self.embedding_size}")
                    print("Đang xóa collection cũ và tạo mới...")
                    self.client.delete_collection(collection_name=self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE)
                    )
                    print("Đã tạo lại collection với kích thước đúng!")
        except Exception as e:
            print(f"Lỗi khi tạo collection: {e}")

    def add_documents(self, documents: List[Dict[str, Any]], embedder: EmbedderClient):
        """Thêm documents vào vector database."""
        print(f"Đang thêm {len(documents)} documents...")
        
        # Trích xuất text để tạo embeddings
        texts = [doc["content"] for doc in documents]
        
        # Tạo embeddings cho tất cả documents
        embeddings = []
        for i, text in enumerate(texts):
            print(f"Tạo embedding cho document {i+1}/{len(texts)}")
            embedding = embedder.embed(text)
            embeddings.append(embedding)
            
            # Kiểm tra kích thước embedding
            if len(embedding) != self.embedding_size:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_size}, got {len(embedding)}")
            
            time.sleep(0.2)  # Tránh rate limiting
        
        # Tạo points cho Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "title": doc.get("title", f"Document {i+1}"),
                    "content": doc["content"],
                    "category": doc.get("category", "general"),
                    "chunk_id": doc.get("chunk_id", i)
                }
            )
            points.append(point)
        
        # Upload points in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            print(f"Uploading batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Đã thêm {len(points)} documents thành công!")

    def search(self, query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.0):
        """Tìm kiếm documents tương tự."""
        # Kiểm tra kích thước embedding
        if len(query_embedding) != self.embedding_size:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_size}, got {len(query_embedding)}")
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        results = []
        for scored_point in search_result:
            results.append({
                "id": scored_point.id,
                "score": scored_point.score,
                "content": scored_point.payload["content"],
                "title": scored_point.payload.get("title", ""),
                "category": scored_point.payload.get("category", ""),
                "chunk_id": scored_point.payload.get("chunk_id", 0)
            })
        
        return results

    def get_collection_info(self):
        """Lấy thông tin collection."""
        info = self.client.get_collection(self.collection_name)
        return info

    def reset_collection(self):
        """Xóa và tạo lại collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Đã xóa collection {self.collection_name}")
        except:
            print(f"Collection {self.collection_name} không tồn tại")
        
        self._create_collection_if_not_exists()

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
    api_key = "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"
    api_key1 = "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"
    api_key2 = "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"
    
    # Cấu hình Qdrant
    qdrant_url = "http://localhost"  # Hoặc URL của Qdrant server
    qdrant_port = 6333
    collection_name = "pdf_documents"

    print(f"Reading file {file_path}...")
    text = extract_text_from_pdf(file_path)
    print(f"Extracted {len(text)} characters.")

    print("Splitting text into chunks...")
    chunks = chunk_text(text)
    print(f"Generated {len(chunks)} chunks.")

    # Khởi tạo các client
    embedder = EmbedderClient(api_key)
    reranker = RerankerClient(api_key1)
    generator = GeneratorClient(api_key2)
    
    # Khởi tạo vector database với embedder để tự động detect dimension
    vector_db = QdrantVectorDB(collection_name, embedder)

    # Chuẩn bị documents cho Qdrant
    documents = []
    for idx, chunk in enumerate(chunks):
        documents.append({
            "title": f"PDF Chunk {idx + 1}",
            "content": chunk,
            "category": "pdf_document",
            "chunk_id": idx
        })

    # Thêm documents vào Qdrant
    print("\n=== Adding documents to Qdrant ===")
    try:
        vector_db.add_documents(documents, embedder)
        # Hiển thị thông tin collection
        info = vector_db.get_collection_info()
    except Exception as e:
        print(f"Lỗi khi thêm documents: {e}")
        print("Thử reset collection...")
        vector_db.reset_collection()
        vector_db.add_documents(documents, embedder)

    print("\n=== Ready for Search ===")
    while True:
        query = input("\n🔍 Nhập truy vấn để tìm kiếm (hoặc gõ 'exit'): ")
        if query.lower() == "exit":
            break

        try:
            # Tạo embedding cho query
            print("Đang tạo embedding cho query...")
            query_embedding = embedder.embed(query)
            
            # Tìm kiếm trong Qdrant
            print("Đang tìm kiếm trong Qdrant...")
            search_results = vector_db.search(query_embedding, top_k=5, score_threshold=0.3)

            print(f"\n📌 Tìm thấy {len(search_results)} kết quả từ Qdrant:")
            for rank, result in enumerate(search_results):
                print(f"\n{rank + 1}. 🔹 Score: {result['score']:.4f}")
                print(f"📄 {result['title']} (Chunk {result['chunk_id']}):")
                print(f"{result['content'][:300]}...")

            if not search_results:
                print("Không tìm thấy kết quả phù hợp!")
                continue

            # Rerank kết quả
            passages = [result["content"] for result in search_results]
            print("\n🔁 Đang rerank lại kết quả với NVIDIA Reranker...")
            reranked = reranker.rerank(query, passages)

            print("\n🏆 Top kết quả sau reranking:")
            for rank, (chunk, score) in enumerate(reranked):
                print(f"\n{rank + 1}. ⭐ Score: {score:.4f}")
                print(f"{chunk[:300]}...")

            # Generate answer
            print("\n💡 Đang tạo câu trả lời...")
            context = generator.generate_context(query, [r[0] for r in reranked])

            print("\n🤖 Câu trả lời:")
            print(f"\n{context}")
            
        except Exception as e:
            print(f"Lỗi trong quá trình tìm kiếm: {e}")
            continue
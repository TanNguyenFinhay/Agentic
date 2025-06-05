import requests

response = requests.post(
    "http://localhost:6333/collections/pdf_documents/points/scroll",
    headers={"Content-Type": "application/json"},
    json={
        "limit": 10,
        "with_payload": True,
        "with_vector": False
    }
)

print(response.json())
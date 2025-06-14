import requests

BASE_URL = "http://localhost:8000"

def chat_with_agent():
    query = input("Ask your travel query (e.g., Suggest a hill stay in Telangana under ₹10K): ")
    response = requests.post(f"{BASE_URL}/chat", json={"query": query})
    print("Chatbot says:", response.json())

if __name__ == "__main__":
    chat_with_agent()


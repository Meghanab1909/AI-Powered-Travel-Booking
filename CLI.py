import requests

print("Hello! I'm \033[1mTRAVIGENIE 🧞\033[0m, your AI companion for exploring Telangana.")
print("I'm ready to assist you with any questions about \033[1mtravel\033[0m, \033[1mtourism\033[0m, or \033[1mhistory\033[0m within Telangana.")

print("\nHow can I help you today❔")
query = input("👉: ")

if query.strip() not in ["", None]:
    print("\n⏳Thinking...\n")
    try:
        response = requests.post("http://localhost:8000/chat", json={"query": query})
        if response.status_code == 200:
            data = response.json()
            answer = data['response']
                    
            if not answer.strip():
                print("No information found for your query.")
            else:
                print("Answer:")
                print(data["response"])
                
                # Show sources if available
                if data.get("sources"):
                    print("Source Documents:")
                    for src in data["sources"]:
                        print(f"- {src}")
                
    except Exception as e:
        print(f"Could not connect to server: {e}")

            



import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Step 1: Read and clean PDF
reader = PdfReader("Database.pdf")
documents = [page.extract_text() for page in reader.pages]
ids = [str(i) for i in range(len(documents))]

# Remove empty or non-string docs
clean_documents = []
clean_ids = []
for doc, id_ in zip(documents, ids):
    if isinstance(doc, str) and doc.strip():
        clean_documents.append(doc)
        clean_ids.append(id_)

# Step 2: Set up embedding function
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Step 3: Connect to ChromaDB
client = chromadb.PersistentClient(path="C:/Users/Srinu/Downloads/db")

# Delete old collection if exists
try:
    client.delete_collection(name="Travigenie")
except:
    pass  # Ignore if it doesn't exist

# Step 4: Create collection
collection = client.get_or_create_collection(
    name="Travigenie",
    embedding_function=embedding_function
)

# Step 5: Define classification rules
def classify_type(doc):
    doc = doc.lower()
    if any(x in doc for x in ["hospital", "clinic", "emergency", "medical"]):
        return "hospital"
    elif any(x in doc for x in ["fort", "temple", "waterfall", "park", "lake", "palace", "monument", "museum"]):
        return "tourist_spot"
    else:
        return "other"

def classify_location(doc):
    doc = doc.lower()
    if any(x in doc for x in ["tirupati", "andhra", "vijayawada", "visakhapatnam", "srisailam"]):
        return "andhra"
    elif any(x in doc for x in ["hyderabad", "warangal", "karimnagar", "nalgonda", "telangana", "nizamabad"]):
        return "telangana"
    else:
        return "unknown"

# Step 6: Generate metadata
metadatas = []
final_docs = []
final_ids = []

for doc, id_ in zip(clean_documents, clean_ids):
    loc = classify_location(doc)
    type_ = classify_type(doc)

    if loc != "telangana":
        type_ = "invalid_location"

    metadata = {
        "type": type_,
        "location": loc
    }

    metadatas.append(metadata)
    final_docs.append(doc)
    final_ids.append(id_)

# Step 7: Add to collection
collection.add(
    documents=final_docs,
    ids=final_ids,
    metadatas=metadatas
)

print("Re-tagging complete with metadata cleanup!")
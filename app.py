import streamlit as st
import requests

st.set_page_config(page_title = "Server") 
st.title("AI Powered Travel Booking") #replace with the project name
st.subheader("🔗 Upload Document")
uploaded_file = st.file_uploader("Choose a file 📂\n\n(Only one file can be uploaded at a time)", accept_multiple_files = False, type = ["pdf", "xlsx"])

with st.sidebar:
    st.header("ℹ️ About the Server")
    st.write("An AI-powered platform for managing specifically Telangana travel-related documents.")
    
    st.header("📄 About the Documents")
    st.write("Upload travel-related files to generate searchable vector embeddings. Only one file to be uploaded at a time\n\nPro Tip: Combine PDF documents or merge Excel sheets to reduce the number of files you need to upload.")
    
if uploaded_file:
    if st.button("🔄 Process"):
        with st.spinner("⏳ Uploading and processing . . ."):
            response = requests.post(
                "http://localhost:8000/upload-docs",
                files = {"file":  (uploaded_file.name, uploaded_file, uploaded_file.type)},
            )
            
        if response.status_code == 200:
            chunks = response.json().get("chunks", 0)
            st.success(f"✅ Document added to vector database")
            st.info(f"🧩 Total chunks created: {chunks}")
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.error(f"❌ Error: {error_msg}")

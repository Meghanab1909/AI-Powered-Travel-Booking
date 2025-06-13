import streamlit as st
import requests

st.set_page_config(page_title="Travel Chat Agent")
st.title("Telangana Travel Chatbot")
st.write("Ask about any travel suggestions, stays, seasons, etc.")

# Chat input
query = st.text_input("Ask a question", placeholder="e.g., Suggest a 3-day trip in Telangana for adventure")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://localhost:8000/chat", json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    st.success("Answer:")
                    st.markdown(data["response"])
                    
                    # Show sources if available
                    if data.get("sources"):
                        st.markdown("#### Source Documents:")
                        for src in data["sources"]:
                            st.markdown(f"- {src}")
                else:
                    st.error("Server error occurred.")
            except Exception as e:
                st.error(f"Could not connect to server: {e}")


st.set_page_config(page_title="Server") 
st.title("AI Powered Travel Booking")
st.subheader("Upload Document")

uploaded_file = st.file_uploader("Choose a file \n\n(Only one file can be uploaded at a time)", 
                                 accept_multiple_files=False, 
                                 type=["pdf", "docx", "xlsx"])

# Sidebar information
with st.sidebar:
    st.header("About the Server")
    st.write("An AI-powered platform for managing specifically Telangana travel-related documents.")
    
    st.header("About the Documents")
    st.write("Upload travel-related files to generate searchable vector embeddings. Only one file to be uploaded at a time.")

# Process button
if uploaded_file:
    if st.button("Process"):
        with st.spinner("Uploading and processing . . ."):
            try:
                response = requests.post(
                    "http://localhost:8000/upload-docs",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                )
                if response.status_code == 200:
                    chunks = response.json().get("chunks", 0)
                    st.success(f"Document added to vector database")
                    st.info(f"Total chunks created: {chunks}")
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_msg}")
            except Exception as e:
                st.error(f"Server error: {str(e)}")

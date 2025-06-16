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
                    answer = data['response']
                    
                    if not answer.strip():
                        st.info("No information found for your query.")
                    else:
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



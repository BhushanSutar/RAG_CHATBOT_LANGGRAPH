import streamlit as st
from graph import rag_app

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")

st.title("Agentic AI RAG Chatbot")
st.caption("Answers strictly from the Agentic AI eBook")

question = st.text_input("Ask a question from the Agentic AI eBook:")

if question:
    with st.spinner("Thinking..."):
        result = rag_app.invoke({"question": question})

    st.markdown("Answer")
    st.write(result["answer"])

    st.markdown("Retrieved Context")
    for i, doc in enumerate(result["docs"]):
        with st.expander(f"Chunk {i+1}"):
            st.write(doc.page_content)

    st.markdown(f"Confidence: {result['confidence']}")

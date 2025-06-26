import streamlit as st
import json
from dotenv import load_dotenv
import os

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Load menu
if not os.path.exists("menu_data.json"):
    st.error("menu_data.json file not found!")
    st.stop()

with open("menu_data.json") as f:
    menu_json = json.load(f)

menu_docs = [
    Document(page_content=f"{item['item']}: {item['description']} - {item['price']}")
    for item in menu_json['menu']
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(menu_docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever()
)

# Load CSS
with open("style.css") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Title
st.title("🍽️ Restaurant Menu Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about the menu?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display bot response
    with st.chat_message("bot"):
        with st.spinner("Thinking..."):
            response = qa_chain.run(prompt)
            st.markdown(response)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": response})

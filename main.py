import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from streamlit_chat import message
from indexing import model, index  

load_dotenv()

st.title("JSS AI HelpDesk")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Welcome to JSS Academy of Technical Education, How can I help you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

retrieval_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def find_match(input_text):
    input_embedding = retrieval_model.encode(input_text).tolist()
    result = index.query(vector=input_embedding, top_k=2, include_metadata=True)
    matches = [match['metadata']['text'] for match in result['matches']]
    unique_matches = list(dict.fromkeys(matches))
    return "\n".join(unique_matches)

def generate_response(query, context):
    return f"Here is what I found based on your query:\n\n{context}\n\nDoes this answer your question?"

response_container = st.container()
text_container = st.container()

with text_container:
    query = st.text_input("Query:", key="input")
    if query:
        with st.spinner("Typing..."):
            context = find_match(query)
            response = generate_response(query, context)
        
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=f"bot_{i}")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=f"user_{i}")

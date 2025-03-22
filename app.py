import re
import string
import regex as re
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# Initializing chromaDB
client = chromadb.PersistentClient(path="searchengine_database") #_test_db
collection = client.get_collection(name="search_engine") #test_collection
# collection_name = client.get_collection(name="search_engine_FileName")
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name, device="cpu")

def clean_data(data):
    data = re.sub(r"\r\n",'',data)
    data = re.sub(r"-->","",data)
    data = re.sub("[<>]","",data)
    data = re.sub('^.*?¶\s*|ï»¿\s*|¶|âª', '', data)
    data = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    data = data.lower()
    data = data.strip()
    
    return data    

def extract_id(id_list):
    new_id_list = []
    for item in id_list:
        match = re.match(r'^(\d+)', item)
        if match:
            extracted_number = match.group(1)
            new_id_list.append(extracted_number)
    return new_id_list

st.title("Enhancing Search Engine Relevance for Video Subtitles") 

with st.form("search_form"):
    search_query = st.text_input("Enter a dialogue to search.", key="search_query")
    submit_button = st.form_submit_button(label="Search")

if submit_button:
    search_query = clean_data(search_query)
    query_embed = model.encode(search_query).tolist()

    search_results = collection.query(query_embeddings=query_embed, n_results=10)
    id_list = search_results['ids'][0]

    id_list = extract_id(id_list)
    
    with st.expander("Relevant Subtitle Files", expanded=True):
        for index, id in enumerate(id_list, start=1):
            file_name = collection.get(ids=f"{id}")["documents"][0]
            st.markdown(f"**{index}** - [{file_name}](https://www.opensubtitles.org/en/subtitles/{id})")

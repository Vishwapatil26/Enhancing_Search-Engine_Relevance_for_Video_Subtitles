import pandas as pd
import sqlite3
import zipfile
import io

conn = sqlite3.connect(r"C:\Users\vishp\Downloads\eng_subtitles_database.db")
cursor = conn.cursor()
print(cursor)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
df = pd.read_sql_query("""SELECT * FROM zipfiles""", conn)
df.head()
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()
df['content']
def decode(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        list = zip_file.namelist()[0]
        extract_zip = zip_file.read(list)
    return extract_zip.decode('latin-1')

df['content'] = df['content'].progress_apply(lambda x: decode(x))
df.info()
df.shape
df.size
df.describe()
df = df.sample(frac=0.3, random_state=42)
import re
df['content']
import nltk
import string
nltk.download('stopwords')
def clean_data(doc):
    doc = re.sub(r"\r\n",'',doc)
    doc = re.sub(r"-->","",doc)
    doc = re.sub("[<>]","",doc)
    doc = re.sub('^.*?¶\s*|ï»¿\s*|¶|âª', '', doc)

    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = doc.lower()
    doc = doc.strip()
    
    return doc

df['content'] = df['content'].progress_apply(clean_data)
 df
chunk_size  = 500 
overlap =100

def chunk_text(text, chunk_size,overlap):
    chunks = []
    start =0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk.lower())
        start+=chunk_size -overlap
    return chunks

df['chunks'] =df['content'].progress_apply(lambda x: chunk_text(x,chunk_size,overlap))
from sentence_transformers import SentenceTransformer, util
from chromadb.utils import embedding_functions
import chromadb
df1 = df.sample(frac=0.5, random_state=42)
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name, device='cpu')
df1['encoding_data'] = df1.chunks(model.encode)
client = chromadb.PersistentClient(path="searchengine_database")
collection = client.get_or_create_collection(name="search_engine", metadata={"hnsw:space": "cosine"})

def store_encoder_db(df):
    for i in range(df.shape[0]):
        embedding_list = df['encoding_data'].iloc[i].tolist()
        collection.add(
            documents=[df['name'].iloc[i]],
            embeddings=[embedding_list],  
            ids=[str(df['num'].iloc[i])]
        )
store_encoder_db(df)

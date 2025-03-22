**🎬 Enhancing Search Engine Relevance for Video Subtitles**

📌 Overview     
This project enhances video subtitle search relevance using Natural Language Processing (NLP) and Machine Learning (ML). It allows users to search video subtitles efficiently using semantic search instead of simple keyword-based matching.   
   
🚀 Features   
1. Keyword-Based & Semantic Search
2. Subtitle Preprocessing & Vectorization
3. Cosine Similarity for Search Ranking
4. ChromaDB for Embedding Storage
5. Audio-Based Query Search

📂 Dataset    
- Subtitle data is stored in a database file.
- Requires preprocessing and chunking for optimal search performance.

🔑 Core Process
Part 1: Document Ingestion     
1. Read & preprocess subtitle data.
2. Convert subtitles into vector embeddings using:  
    TF-IDF (Keyword-Based Search) and 
    SentenceTransformers (Semantic Search)
3. Chunk long documents for better context retention.
4. Store embeddings in ChromaDB.    

Part 2: Document Retrieval
1. Convert user audio query to text.
2. Generate embeddings for the query.
3. Compute cosine similarity between query and subtitle embeddings.
4. Return the most relevant subtitles.

🛠️ Tech Stack
- Python, Streamlit, ChromaDB
- SentenceTransformers, TF-IDF
- OpenAI/Google Speech-to-Text

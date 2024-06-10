from functools import lru_cache
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
#from transformers import AutoTokenizer, AutoModel

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModel.from_pretrained(MODEL_NAME)

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_NAME = "arxiv-semantic-search"

class SemanticModel:
    def __init__(self, api_key, encoder_name=MODEL_NAME, index_name=INDEX_NAME):
        self.load_encoder(encoder_name)
        self.load_index(index_name, api_key)
    
    def load_encoder(self, name):
        self.encoder = SentenceTransformer(name)

    def load_index(self, index_name, api_key):
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)
    
    @lru_cache
    def encode(self, text):
        return self.encoder.encode(text).tolist()
    
    @lru_cache
    def results(self, text, num_results=10):
        return self.index.query(
            vector=self.encode(text), top_k=num_results, include_metadata=False
            )['matches']
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch
import time
import pandas as pd
from tqdm.auto import tqdm

#from transformers import AutoTokenizer, AutoModel

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModel.from_pretrained(MODEL_NAME)

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_DIMENSION = 384
INDEX_NAME = "arxiv-semantic-search"

# File where the abstracts were saved after data processing. Not in the repo since it's too big
CORPUS_FILE = "data/arxiv_all_abstract.parquet"

class SemanticModel:
    def __init__(self, api_key, encoder_name=MODEL_NAME, index_name=INDEX_NAME):
        self.load_encoder(encoder_name)
        self.load_index(index_name, api_key)
    
    def load_encoder(self, name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = SentenceTransformer(name, device=self.device)

    def load_index(self, index_name, api_key):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
    
    @lru_cache
    def encode(self, text):
        return self.encoder.encode(text, device=self.device).tolist()
    
    @lru_cache
    def results(self, text, num_results=10):
        return self.index.query(
            vector=self.encode(text), top_k=num_results, include_metadata=False
            )['matches']
    
    def embed_corpus(self, corpus):
        self.embeddings = self.encoder.encode(
            sentences=corpus,
            normalize_embeddings=False,
            device=self.device,
            convert_to_tensor=True,
            show_progress_bar=True)
    
    def initialize_index(self, index_name, api_key, dimension):
        if self.index_name == index_name:
            return
        cloud = 'aws'
        region = 'us-east-1'
        spec = ServerlessSpec(cloud=cloud, region=region)
        self.pc = Pinecone(api_key=api_key)

        existing_indexes = [
            index_info["name"] for index_info in self.pc.list_indexes()
        ]

        # check if index already exists (it shouldn't if this is first time)
        if index_name not in existing_indexes:
            # if does not exist, create index
            self.pc.create_index(
                index_name,
                dimension=dimension,
                metric='dotproduct',
                spec=spec
            )
            # wait for index to be initialized
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)

        self.index = self.pc.Index(index_name)

    def initial_upsert(self, corpus_filename, batch_size):
        corpus = load_corpus(corpus_filename)
        self.embed_corpus(corpus)

        for i in tqdm(range(0, len(corpus), batch_size)):
            i_end = min(i+batch_size, len(corpus))
            # create IDs batch
            ids = [str(x) for x in range(i, i_end)]
            # create metadata batch
            metadatas = [{'text': text} for text in corpus[i:i_end]]
            # create records list for upsert
            records = zip(ids, self.embeddings[i:i_end], metadatas)
            # upsert to Pinecone
            self.index.upsert(vectors=records)

@lru_cache
def load_corpus(filename):
    return pd.read_parquet(filename)
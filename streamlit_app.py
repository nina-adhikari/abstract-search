import streamlit as st
import pandas as pd
import urllib, urllib.request
import xml.etree.ElementTree as ET
from io import StringIO
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

SHOW_CLASSIC_SEARCH = False
NUM_RESULTS = 10

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_FILE_NAME = 'data/arxiv_all_small.parquet'
INDEX_NAME = "arxiv-semantic-search"

API_KEY = st.secrets['PINECONE_API_KEY']
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModel.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def load_data(file_name):
    return pd.read_parquet(file_name).drop(columns=['index'])

@st.cache_resource
def load_index(index_name, api_key):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def query_arxiv(input, search=True):
    baseUrl = 'https://export.arxiv.org/api/query?'
    if search:
        if input.find(" ") == -1:
            text = ", " + input
        else:
            text = input
        text = text.replace(" ", "%20")
        url = baseUrl + 'sortBy=relevance&start=0&max_results=10&search_query=ab:' + text
    else:
        url = baseUrl + 'id_list=' + ','.join(input)
    data = urllib.request.urlopen(url)

    it = ET.iterparse(StringIO(data.read().decode('utf-8')))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition('}') # strip ns
    root = it.root
    return root

def traverse(tree):
    results = []

    for entry in tree.findall("entry"):
        result = {}
        result['id'] = entry.find("id").text
        result['Title'] = entry.find("title").text
        result['Abstract'] = entry.find("summary").text
        result['Authors'] = ''
        for author in entry.findall("author"):
            result['Authors'] += author.find("name").text + ', '
        result['Authors'] = result['Authors'][:-2]
        results.append(result)
    return results

def prettify(result):
    return f'''**{result['Title']}**  
    <small>[{result['id'].strip("http://arxiv.org/abs/")}]({result['id']})</small>  
    <small>*{result['Authors']}*</small>  
    **Abstract**: {result['Abstract']}'''

def show_results(results):
    for i in range(len(results)):
        result = results[i]
        st.markdown(
            str(i+1) + ". " + prettify(result),
            unsafe_allow_html=True
        )
def semantic_search(text, model, df, index, top_k=10):
    encoded = model.encode(text).tolist()

    results = index.query(vector=encoded, top_k=top_k, include_metadata=False)
    return [df.iloc[int(entry['id'])]['id'] for entry in results['matches']]

def main():
    model = load_model(MODEL_NAME)
    df = load_data(DATA_FILE_NAME)
    index = load_index(INDEX_NAME, API_KEY)
    st.title("Abstract search")

    st.write(
        "This app demonstrates a semantic search functionality by using the \
    entered search term to return the ArXiv abstracts closest to it. The ArXiv \
    archive is updated through June 8, 2024."
    )

    form = st.form("search")

    query = form.text_input("Search ArXiv abstracts: :red[\*]", placeholder="Enter search term")
    #button = form.form_submit_button("Search", on_click=search, args=[query])
    button = form.form_submit_button("Search")
    if button:
        if not query:
            form.error("Search term is empty!")
        else:
            sem = semantic_search(query, model=model, df=df, index=index, top_k=NUM_RESULTS)
            semantic_results = traverse(query_arxiv(sem, search=False))
            with st.container():
                st.subheader("Semantic search results")
                show_results(semantic_results)
            if SHOW_CLASSIC_SEARCH:
                arxiv_results = traverse(query_arxiv(query))
                with st.container():
                    st.subheader("Ordinary search results")
                    show_results(arxiv_results)

if __name__=="__main__":
    main()
import streamlit as st
import pandas as pd
import urllib, urllib.request
import xml.etree.ElementTree as ET
from io import StringIO
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModel.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def load_data(file_name):
    return pd.read_parquet(file_name).drop(columns=['index'])

api_key = st.secrets['PINECONE_API_KEY']

@st.cache_resource
def load_index(index_name):
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
    return f'''[{result['Title']}]({result['id']})  
    <small>*{result['Authors']}*</small>  
    **Abstract**: {result['Abstract'][:200]}...'''

def semantic_search(text, model, df, index):
    encoded = model.encode(text).tolist()

    results = index.query(vector=encoded, top_k=10, include_metadata=False)
    return [df.iloc[int(entry['id'])]['id'] for entry in results['matches']]

def main():
    model = load_model(MODEL_NAME)
    df = load_data('data/arxiv_all_small.parquet')
    index = load_index("arxiv-semantic-search")
    st.title("Abstract search")

    st.write(
        "This app demonstrates a semantic search functionality by using the \
    entered search term to return the ArXiv abstracts closest to it. The ArXiv \
    is updated through June 8, 2024."
    )

    form = st.form("search")

    query = form.text_input("Search ArXiv abstracts:")
    #button = form.form_submit_button("Search", on_click=search, args=[query])
    button = form.form_submit_button("Search")
    if button:
        if not query:
            st.write("Empty")
        else:
            arxiv_results = traverse(query_arxiv(query))
            sem = semantic_search(query, model, df, index)
            semantic_results = traverse(query_arxiv(sem, search=False))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ordinary search")
                for i in range(len(arxiv_results)):
                    result = arxiv_results[i]
                    st.markdown(
                        str(i+1) + ". " + prettify(result),
                        unsafe_allow_html=True
                    )
            with col2:
                st.subheader("Semantic search")
                for i in range(len(semantic_results)):
                    result = semantic_results[i]
                    st.markdown(
                        str(i+1) + ". " + prettify(result),
                        unsafe_allow_html=True
                    )

if __name__=="__main__":
    main()
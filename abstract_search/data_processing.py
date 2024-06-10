#!kaggle datasets download -d Cornell-University/arxiv
#!unzip arxiv.zip

import pandas as pd
import json

# Directory where https://www.kaggle.com/datasets/Cornell-University/arxiv has been downloaded
DIR = 'temp/'
JSON_FILE = DIR + 'arxiv-metadata-oai-snapshot.json'

def parse_text(text):
    return json.loads(text.split('\n')[0])

def load_json(filename=JSON_FILE):
    with open(filename) as json_file:
        data = json_file.readlines()
        #data = list(map(json.loads, data))

    df = pd.DataFrame(data)
    #print("Dataframe Loaded")
    df.rename(columns={0: 'text'}, inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    return df

def write_batches(dataframe, num_batches, dir=DIR):
    length = int(len(dataframe) / num_batches)
    for i in range(num_batches):
        offset = i * length
        last = length if i < num_batches-1 else len(dataframe) - offset
        #print(f'offset: {offset}, last: {offset+last}')
        batch = pd.DataFrame(dataframe.iloc[offset:offset+last].copy())
        #print(batch.head())
        batch['text'] = batch['text'].apply(parse_text)
        #print(batch.head())
        normalized = pd.json_normalize(batch['text'])
        normalized.index = batch.index
        #print(normalized.head())
        batch = batch.join(normalized, how='left', validate='1:1')
        #print(batch.head())
        batch.to_feather(dir + f'arxiv_{i}_cols.feather')

def read_batches(num_batches, dir=DIR, small=True):
    df = [0]*num_batches
    for i in range(num_batches):
        df[i] = pd.read_feather(dir + f'arxiv_{i}_cols.feather')
        if small:
            df[i]['index'] = df[i].index
            df[i] = df[i][['index', 'id']]
    
    print("Loaded partials")
    df = pd.concat(df)
    #print(len(df))
    #print(df.head())
    return df

def to_parquet(df, dir=DIR, small=True):
    filename = dir + 'arxiv_all.parquet'
    if small:
        filename = dir + 'arxiv_all_small.parquet'
    df.to_parquet(filename)

def main():
    df = load_json(JSON_FILE)

    write_batches(df, 10, DIR)

    df = read_batches(10, DIR)

    to_parquet(df, DIR)

if __name__=="__main__":
    main()

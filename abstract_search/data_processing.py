#!kaggle datasets download -d Cornell-University/arxiv
#!unzip arxiv.zip

import pandas as pd
import json

# Directory where https://www.kaggle.com/datasets/Cornell-University/arxiv has been downloaded
DIR = 'temp/'
JSON_FILE = DIR + 'arxiv-metadata-oai-snapshot.json'

def parse_text(text):
    return json.loads(text.split('\n')[0])

def main():
    with open(JSON_FILE) as json_file:
        data = json_file.readlines()
        #data = list(map(json.loads, data))

    df = pd.DataFrame(data)
    print("Dataframe Loaded")
    df.rename(columns={0: 'text'}, inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    length = int(len(data) / 10)
    print(type(df))
    print(df.head())

    for i in range(10):
        offset = i * length
        last = length if i < 9 else len(data) - offset
        print(f'offset: {offset}, last: {offset+last}')
        batch = pd.DataFrame(df.iloc[offset:offset+last].copy())
        #print(batch.head())
        batch['text'] = batch['text'].apply(parse_text)
        #print(batch.head())
        normalized = pd.json_normalize(batch['text'])
        normalized.index = batch.index
        #print(normalized.head())
        batch = batch.join(normalized, how='left', validate='1:1')
        #print(batch.head())
        batch.to_feather(DIR + f'arxiv_{i}_cols.feather')
        del batch
    
    del df

    #df = [0]*10
    
    #for i in range(10):
    #    df[i]['text'] = df[i]['text'].apply(parse_text)
    #    df[i] = 0

    # df['text'] = df['text'].apply(parse_text)
    # df = df.join(pd.json_normalize(df['text']), how='left', validate='1:1').drop(columns=['text'])

    df = [0]*10
    for i in range(10):
        df[i] = pd.read_feather(DIR + f'arxiv_{i}_cols.feather')
    
    print("Loaded partials")
    df = pd.concat(df)
    print(len(df))
    print(df.head())
    df.to_parquet(DIR + 'arxiv_all.parquet')
    df['index'] = df.index
    df = df[['index', 'id']]
    df.to_parquet(DIR + 'arxiv_all_small.parquet')

    df = pd.read_parquet(DIR + 'arxiv_all_small.parquet')
    print(df.head())

if __name__=="__main__":
    main()

import pandas as pd
import requests
import datetime as dt
import os
from os import path
import time
import regex as re
import ssl
import certifi
from elasticsearch import helpers, Elasticsearch
from elasticsearch.exceptions import ConnectionError
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile

# function for querying Reddit's Pushshift API
def query_pushshift(subreddit, # subreddit name
                    days, # number of days to iterate through
                    kind=['submission', 'comment'], # can be 'submission' or 'comment'
                    skip = 1, # number of days in each time period
                    subfield = ['title', 'selftext', 'subreddit', 'created_utc', 
                                'author', 'num_comments', 'score', 'is_self', 'full_link'], 
                    # subfields for just submissions
                    comfields = ['body', 'score', 'created_utc']): # fields for comments

    stem = "https://api.pushshift.io/reddit/search/{}/?subreddit={}&size=1000".format(kind, subreddit)
    # creating base url
    mylist = [] # instantiating empty list
    
    if kind == 'submission': 
        iter_ = 24 # for submissions, iterate every 24 hours
    elif kind == 'comment':
        iter_ = 4 # for comments, iterate every 4 hours
        
    times = days * 24
    after = list(range(1, times + (1 * iter_), iter_)) 
    before = [0] + after
    
    for x, i in zip(before, after): # iterating through times

        try:
            URL = "{}&before={}h&after={}h".format(stem, skip * x, skip * i) # new url for each time period
            print(URL)
            response = requests.get(URL) # setting up scraper
            timer = 0
            while response.status_code != 200:
                time.sleep(5)
                timer += 5
                if response.status_code == 200:
                    break# prints url as it's scraping it
                elif timer >= 20:
                    break
                    print("Couldn't connect")
            mine = response.json()['data'] # content we want from scrape
            df = pd.DataFrame.from_dict(mine) # setting up dataframe from dictionaries of scraped content
            mylist.append(df) # adding to mylist
            time.sleep(0.25)
        except:
            pass

    full = pd.concat(mylist, sort=False) # concatenating all dfs into one
    
    if kind == "submission": # for submissions, dropping dups and not including comfields
        
        full = full[subfield]
        full = full.drop_duplicates(subset = 'full_link')
        
    elif kind == "comment":
        
        full = full.drop_duplicates(subset = 'permalink')
    
    _timestamp = full["created_utc"].apply(lambda x: dt.date.fromtimestamp(x)) # changing created_utc to date
    full['timestamp'] = _timestamp # setting new timestamp as field in df

    print("Pulled {} {}s".format(full.shape[0], kind)) #prints shape of final df at end of scrape
    
    return full 

# organizing posts/comments
def comments_posts(subreddit, days):
    
    posts = query_pushshift(subreddit, days, kind = 'submission')
    posts['common_link'] = [i.replace('https://www.reddit.com', '') for i in posts['full_link']]
    posts['type'] = 'post'
    posts.fillna("", inplace = True)
    posts['alltext'] = posts['title'] + " " + posts['selftext']
    
    comments = query_pushshift(subreddit, days, kind = 'comment')
    comments['common_link'] = [i[:-8] for i in comments['permalink']]
    comments['comment_id'] = [i[-8:] for i in comments['permalink']]
    comments['type'] = 'comment'
    comments.rename(columns = {'body': 'alltext'}, inplace = True)
    
    both = pd.concat([posts, comments], axis = 0, sort=True)
    both = both.sort_values(by = ['timestamp', 'common_link'], ascending = False)
    both = both.reset_index()
    both = both[['author', 'title', 'type', 'timestamp', 'alltext', 'common_link']]
    
    timecol = list(both['timestamp'])
    end = str(timecol[0].strftime("%Y-%m-%d"))
    start = str(timecol[-1].strftime("%Y-%m-%d"))
    
    print("Done concatenating posts and comments for {}".format(subreddit))
    print("Earliest date pulled: {}".format(start))
    print("Final date pulled: {}".format(end))
    
    return both

def clean(string):
    
    try:
        return re.sub(r"[^a-zA-Z0-9 -]", "", string)
    except:
        pass

# function fo create connection
def reddit_to_elastic(df):

    connected = False

    while not connected:
        try:
            es = Elasticsearch(['elasticsearch:9200']) # here we are connecting to docker container called 'elasticsearch' which is listening on port 9200
            es.info()
            connected = True
        except ConnectionError: # backup in case DB hasn't started yet
            print("Elasticsearch not available yet, trying again in 2s...")
            time.sleep(2)

    es.indices.delete(index = 'namenerds', ignore = [400, 404]) # delete indices if exist
    es.indices.create(
        index='namenerds',
        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "author": {
                        "type": "text"
                    },
                    "title": {
                        "type": "text"
                    },
                    "type": {
                        "type": "text"
                    },
                    "timestamp": {
                        "type": "date"
                    },
                    "alltext": {
                        "type": "text"
                    },
                    "common_link": {
                        "type": "text"
                    }
                }
            }
        })

    df_ingest = df.to_dict("index")
    docs = []
    for i in df_ingest.keys():
        header = {
            "_index": "namenerds",
            "_source": df_ingest[i]
        }
        docs.append(header)

    try:
        response = helpers.bulk(es, docs, chunk_size = 100)
        print("\mRESPONSE:", response)
    except Exception as e:
        print("\nERROR:", e)

## set word2vec args
min_count = 2 # ignores all words with total frequency lower than this.
size = 50 # dimensionality of the word vectors
workers = 3 # more workers = faster   
window = 3 # maximum distance between the current and predicted word within a sentence
sg = 1 # training algorithm: 1 for skip-gram; otherwise CBOW

text_col = 'alltext' # name of column in the dataframe with text for word2vec

def clean_string(i):
    
    try:
        clean = re.sub('[^a-zA-Z\s]', '', i)
        clean = re.sub(r'\s+', ' ', clean)
        clean = clean.lower()
        
        return clean
    
    except:
        pass

def make_word2vec_corpus(df, text_col):
    
    df = pd.read_csv('./data/reddit.csv') # path to csv with reddit data
    df[text_col].fillna("", inplace = True) 
    cleanstrings = df[text_col].apply(lambda x: clean_string(x))
    corpus = [word_tokenize(i) for i in cleanstrings]
    
    return corpus

def make_model(df, text_col = text_col, min_count = min_count, size = size, workers = workers, window = window, sg = sg):
    
    print("Making corpus...")
    corpus = make_word2vec_corpus(df, text_col)
    
    print("Fitting word2vec model...")
    model = Word2Vec(corpus, min_count = min_count, size = size, workers = workers, window = window, sg = sg)
    
    print("Saving keyed vectors...") ## saving the word embedding vectors from word2vec
    keyed_vectors = model.wv

    return keyed_vectors.save('./data/vectors.kv')

def wrapper(subreddit, days):

    if path.exists("./data/reddit.csv"): # if reddit data is already in the folder, won't pull new data
        print("Data has already been pulled.")
        pass
    else:
        df = comments_posts(subreddit, days)
        df = df.applymap(lambda x: clean(x))
        df.to_csv('./data/reddit.csv') # will save locally

    if path.exists("./data/vectors.kv"): # if word embeddings are already in the folder, won't re-train
        print("Word embeddings have already been made.")
        pass

    else:
        make_model(df)

    return reddit_to_elastic(df) # send to elastic

if __name__ == "__main__": # the lazy way to run the wrapper function with arguments pre-set
    wrapper('namenerds', 120) # will pull posts/comments from the subreddit 'namenerds' for the last 3 months
# Name Nerds

## Purpose

Need inspiration for a name for a person or animal? The subreddit '/r/namenerds' has over 100k members who share conversations about names. This app pulls 6 months of posts/comments from the subreddit through pushshift's API, trains a gensim word2vec embedding on the posts, and allows the user to search for names that appear in common through a Flask app that queries the embeddings.  

## Steps to build

1. After cloning the repo, navigate into the 'name-nerds' main directory
2. Build the pushshift image by executing ```docker build -t pushshift ./pushshift```
3. Spin up Elasticsearch, Kibana, and the pushshift image by executing ```docker-compose up```. This takes some time to run (can be sped up by changing the number of days to scrape reddit at the end of /pushshift/data_load.py), and should complete when there is a reddit.csv and vectors.kv file in the /data folder (or when the ingestion to elasticsearch has been completed).
4. After the word2vec model has finished training, build the Flask app image: ```docker build -t myapp .```
5. When this image is built, start the app with ```docker run -p 5000:5000 --name name-nerds myapp:latest```

## Using the app

When the Flask app is up and running, navigate to http://localhost:5000/ to test it by entering a name at the end of the url (ex. http://localhost:moonchild).

## Future

There's a bug with the Kibana/Elastic connection, but once that is issue is resolved the user will also be able to navigate to localhost:5601 to search through their reddit posts. Also, some reconfiguration is needed to scrape/train the model more than the initial time. 

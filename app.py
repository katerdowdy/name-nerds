from flask import Flask
from flask.json import jsonify
from gensim.models import KeyedVectors

app = Flask(__name__)
app.debug = True

@app.route('/') 
def index():

	return jsonify("Let's look up names!")

@app.route('/<string:word>', methods = ['GET']) # this endpoint takes a name and returns back similar names
def word2vec(word):
    vectors = KeyedVectors.load("./data/vectors.kv") # loads pre-trained embeddings
    result = vectors.most_similar(word, topn=100) # pulls most similar from top 100
    return jsonify(result) # need to jsonify the result

if __name__ == '__main__':
    app.run(host='0.0.0.0')
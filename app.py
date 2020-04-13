import config
import pickle
import nltk
from flask import Flask, request, jsonify 
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

app = Flask(__name__)
model = pickle.load(open("categorized_model.sav", 'rb'))

@app.route("/", methods=['GET', 'POST'])
def predict():
    req = request.get_json()
    print(req)
    sentence = req['sentence']
    prediction = int(model.predict([sentence])[0])
    return jsonify({'sentence':sentence, 'class':prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
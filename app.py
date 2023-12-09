from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Home Route 
@app.route('/', methods=['GET'])
def index():
    return jsonify({'Message': "Welcome to our sentiment_model"})

# Create a route that manages user requests and performs sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Analyze sentiment
    sentiment = analyzer.polarity_scores(text)

    # Interpret the compound score
    compound_score = sentiment['compound']

    if compound_score >= 0.5:
        sentiment_label = 'This sentence is positive'
    elif 0.0 <= compound_score < 0.5:
        sentiment_label = 'This sentence is  neutral'
    else:
        sentiment_label = 'This sentence is negative'

    # Return simplified sentiment label
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)

# # Load the saved model
# sentiment_review_model = joblib.load('sentiment_model.pkl')

# # Index Route 
# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({'Message': "Welcome to our sentiment analysis Model"})

# # create a route that manages user request and does sentiment prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     text = data['text']
#     vectorizer = CountVectorizer(max_features=1000)
#     vectorized_text = vectorizer.transform([text])
#     prediction = sentiment_review_model.predict(vectorized_text)[0]
#     return jsonify({'sentiment': prediction})

# if __name__ == '__main__':
#     app.run()
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from cluster import DBSCANPipeline, Vectorizer
def load(filename):
    """Load a clustering pipeline"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

stored_classifier = load('./clustering/stars(0.017-500)-sentiment.pkl')
stored_vectorizer = load('./clustering/six_eye.pkl')
top_categories = load('./clustering/top_categories.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json  # Retrieve the JSON data sent by the form
    
    # process input to work with our existing model
    # --- Example Usage: Define a New Business ---
    new_business = {
        'text': "good",
        'categories': "Cafes, Coffee, Breakfast",
        'latitude': 37.02,
        'longitude': -88.8104,
        'city': "San Francisco",
        'hours': {
            "Monday": "07:00-18:00",
            "Tuesday": "07:00-18:00",
            "Wednesday": "07:00-18:00",
            "Thursday": "07:00-18:00",
            "Friday": "07:00-18:00",
            "Saturday": "08:00-16:00",
            "Sunday": "08:00-16:00"
        },
        "stars": None,
    }
    
        
    classification = stored_classifier['classify_by_closest_cluster'](new_business, stored_classifier['avg_features'], stored_classifier['classification_features'], stored_vectorizer)
    # res = stored_classifier['insights'][stored_classifier['insights']['cluster'] == classification]
    
    return str(classification)
    # Log or process the data (for now, just return it back)
    # return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
    





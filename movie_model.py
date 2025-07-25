import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class MovieRecommendationModel:
    def __init__(self, model_path='omdb_movie_model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from pickle file"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found. Please run generate_model.py first.")
            self.model_data = None
    
    def predict(self, age, gender, num_recommendations=6):
        """Get movie recommendations for user"""
        if not self.model_data:
            return []
        
        try:
            # Encode input
            gender_encoded = self.model_data['gender_encoder'].transform([gender])[0]
            input_features = np.array([[age, gender_encoded]])
            
            # Get predictions
            probabilities = self.model_data['model'].predict_proba(input_features)[0]
            top_indices = np.argsort(probabilities)[-num_recommendations:][::-1]
            
            # Get recommended movies
            recommended_movie_names = self.model_data['movie_encoder'].inverse_transform(top_indices)
            
            recommendations = []
            for i, movie_name in enumerate(recommended_movie_names):
                movie_data = next((m for m in self.model_data['movies_data'] 
                                 if m['title'] == movie_name), None)
                if movie_data:
                    recommendations.append({
                        'title': movie_data['title'],
                        'genre': movie_data['primary_genre'],
                        'rating': movie_data['rating'],
                        'confidence': round(float(probabilities[top_indices[i]]) * 100, 1),
                        'release_date': movie_data['release_date'],
                        'overview': movie_data['overview'][:150] + "..." if len(movie_data['overview']) > 150 else movie_data['overview'],
                        'poster_url': movie_data['poster_path'] if movie_data['poster_path'] != 'N/A' else None
                    })
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
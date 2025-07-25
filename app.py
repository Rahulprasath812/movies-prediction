# app.py - Complete Flask Movie Recommendation System
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import requests
import os
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# OMDB API Configuration
OMDB_API_KEY = ""  # Replace with your actual API key
OMDB_BASE_URL = "http://www.omdbapi.com/"

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_cache = None
        self.cache_timestamp = None
        self.cache_duration = 1800  # 30 minutes
        self.model_data = None
        self.load_or_create_model()
    
    def get_sample_movies(self):
        """Sample movie database for demonstration"""
        return [
            {
                'id': 1, 'title': 'Spider-Man: No Way Home', 'primary_genre': 'Action',
                'genres': ['Action', 'Adventure', 'Sci-Fi'], 'rating': 8.4,
                'release_date': '17 Dec 2021', 'popularity': 95.5,
                'overview': 'With Spider-Man\'s identity now revealed, Peter asks Doctor Strange for help. When a spell goes wrong, dangerous foes from other worlds start to appear, forcing Peter to discover what it truly means to be Spider-Man.',
                'poster_path': 'https://via.placeholder.com/300x450/1a1a1a/ffffff?text=Spider-Man%3A+No+Way+Home',
                'year': '2021', 'director': 'Jon Watts', 'runtime': '148 min', 'imdb_id': 'tt10872600'
            },
            {
                'id': 2, 'title': 'Top Gun: Maverick', 'primary_genre': 'Action',
                'genres': ['Action', 'Drama'], 'rating': 8.3,
                'release_date': '27 May 2022', 'popularity': 89.2,
                'overview': 'After thirty years, Maverick is still pushing the envelope as a top naval aviator, but must confront ghosts of his past when he leads TOP GUN\'s elite graduates on a mission that demands the ultimate sacrifice.',
                'poster_path': 'https://via.placeholder.com/300x450/2a2a2a/ffffff?text=Top+Gun%3A+Maverick',
                'year': '2022', 'director': 'Joseph Kosinski', 'runtime': '131 min', 'imdb_id': 'tt1745960'
            },
            {
                'id': 3, 'title': 'The Batman', 'primary_genre': 'Action',
                'genres': ['Action', 'Crime', 'Drama'], 'rating': 7.8,
                'release_date': '04 Mar 2022', 'popularity': 87.3,
                'overview': 'When a sadistic serial killer begins murdering key political figures in Gotham, Batman is forced to investigate the city\'s hidden corruption and question his family\'s involvement.',
                'poster_path': 'https://via.placeholder.com/300x450/3a3a3a/ffffff?text=The+Batman',
                'year': '2022', 'director': 'Matt Reeves', 'runtime': '176 min', 'imdb_id': 'tt1877830'
            },
            {
                'id': 4, 'title': 'Black Panther: Wakanda Forever', 'primary_genre': 'Action',
                'genres': ['Action', 'Adventure', 'Drama'], 'rating': 6.7,
                'release_date': '11 Nov 2022', 'popularity': 84.1,
                'overview': 'Queen Ramonda, Shuri, M\'Baku, Okoye and the Dora Milaje fight to protect their nation from intervening world powers in the wake of King T\'Challa\'s death.',
                'poster_path': 'https://via.placeholder.com/300x450/4a4a4a/ffffff?text=Black+Panther%3A+Wakanda+Forever',
                'year': '2022', 'director': 'Ryan Coogler', 'runtime': '161 min', 'imdb_id': 'tt9114286'
            },
            {
                'id': 5, 'title': 'Avatar: The Way of Water', 'primary_genre': 'Sci-Fi',
                'genres': ['Action', 'Adventure', 'Family', 'Sci-Fi'], 'rating': 7.6,
                'release_date': '16 Dec 2022', 'popularity': 92.8,
                'overview': 'Jake Sully lives with his newfound family formed on the extrasolar moon Pandora. Once a familiar threat returns to finish what was previously started, Jake must work with Neytiri and the army of the Na\'vi race to protect their planet.',
                'poster_path': 'https://via.placeholder.com/300x450/5a5a5a/ffffff?text=Avatar%3A+The+Way+of+Water',
                'year': '2022', 'director': 'James Cameron', 'runtime': '192 min', 'imdb_id': 'tt1630029'
            },
            {
                'id': 6, 'title': 'Everything Everywhere All at Once', 'primary_genre': 'Comedy',
                'genres': ['Comedy', 'Adventure', 'Sci-Fi'], 'rating': 7.8,
                'release_date': '08 Apr 2022', 'popularity': 85.4,
                'overview': 'An aging Chinese immigrant is swept up in an insane adventure, where she alone can save what\'s important to her by connecting with the lives she could have led in other universes.',
                'poster_path': 'https://via.placeholder.com/300x450/6a6a6a/ffffff?text=Everything+Everywhere+All+at+Once',
                'year': '2022', 'director': 'Daniel Kwan, Daniel Scheinert', 'runtime': '139 min', 'imdb_id': 'tt6710474'
            },
            {
                'id': 7, 'title': 'Nope', 'primary_genre': 'Horror',
                'genres': ['Horror', 'Mystery', 'Sci-Fi'], 'rating': 6.8,
                'release_date': '22 Jul 2022', 'popularity': 76.2,
                'overview': 'The residents of a lonely gulch deal with a mysterious threat that reproduces itself by means of any reflective surface.',
                'poster_path': 'https://via.placeholder.com/300x450/7a7a7a/ffffff?text=Nope',
                'year': '2022', 'director': 'Jordan Peele', 'runtime': '130 min', 'imdb_id': 'tt10954984'
            },
            {
                'id': 8, 'title': 'Minions: The Rise of Gru', 'primary_genre': 'Animation',
                'genres': ['Animation', 'Comedy', 'Family'], 'rating': 6.5,
                'release_date': '01 Jul 2022', 'popularity': 81.9,
                'overview': 'The untold story of one twelve-year-old\'s dream to become the world\'s greatest supervillain.',
                'poster_path': 'https://via.placeholder.com/300x450/8a8a8a/ffffff?text=Minions%3A+The+Rise+of+Gru',
                'year': '2022', 'director': 'Kyle Balda', 'runtime': '87 min', 'imdb_id': 'tt5113044'
            },
            {
                'id': 9, 'title': 'Thor: Love and Thunder', 'primary_genre': 'Action',
                'genres': ['Action', 'Adventure', 'Comedy'], 'rating': 6.2,
                'release_date': '08 Jul 2022', 'popularity': 78.5,
                'overview': 'Thor enlists the help of Valkyrie, Korg and ex-girlfriend Jane Foster to fight Gorr the God Butcher, who intends to make the gods extinct.',
                'poster_path': 'https://via.placeholder.com/300x450/9a9a9a/ffffff?text=Thor%3A+Love+and+Thunder',
                'year': '2022', 'director': 'Taika Waititi', 'runtime': '119 min', 'imdb_id': 'tt10648342'
            },
            {
                'id': 10, 'title': 'Encanto', 'primary_genre': 'Animation',
                'genres': ['Animation', 'Comedy', 'Family'], 'rating': 7.2,
                'release_date': '24 Nov 2021', 'popularity': 82.7,
                'overview': 'A Colombian teenage girl faces the frustration of being the only member of her family without magical powers.',
                'poster_path': 'https://via.placeholder.com/300x450/aaaaaa/ffffff?text=Encanto',
                'year': '2021', 'director': 'Jared Bush, Byron Howard', 'runtime': '102 min', 'imdb_id': 'tt2953050'
            },
            {
                'id': 11, 'title': 'Dune', 'primary_genre': 'Sci-Fi',
                'genres': ['Action', 'Adventure', 'Drama', 'Sci-Fi'], 'rating': 8.0,
                'release_date': '22 Oct 2021', 'popularity': 88.9,
                'overview': 'Paul Atreides, a brilliant and gifted young man born into a great destiny beyond his understanding, must travel to the most dangerous planet in the universe.',
                'poster_path': 'https://via.placeholder.com/300x450/bababa/ffffff?text=Dune',
                'year': '2021', 'director': 'Denis Villeneuve', 'runtime': '155 min', 'imdb_id': 'tt1160419'
            },
            {
                'id': 12, 'title': 'Scream', 'primary_genre': 'Horror',
                'genres': ['Horror', 'Mystery', 'Thriller'], 'rating': 6.3,
                'release_date': '14 Jan 2022', 'popularity': 75.8,
                'overview': 'Twenty-five years after the original series of murders in Woodsboro, a new Ghostface emerges, and Sidney Prescott must return to uncover the truth.',
                'poster_path': 'https://via.placeholder.com/300x450/cacaca/ffffff?text=Scream',
                'year': '2022', 'director': 'Matt Bettinelli-Olpin, Tyler Gillett', 'runtime': '114 min', 'imdb_id': 'tt11245972'
            }
        ]
    
    def create_training_data(self, movies):
        """Create training data based on demographic patterns"""
        np.random.seed(42)
        data = []
        
        # Define genre preferences by demographics
        male_young_genres = ['Action', 'Adventure', 'Sci-Fi', 'Thriller', 'Comedy', 'Horror']
        male_middle_genres = ['Action', 'Thriller', 'Drama', 'Crime', 'Adventure', 'Biography']
        male_older_genres = ['Drama', 'Thriller', 'Crime', 'Biography', 'History', 'War']
        
        female_young_genres = ['Romance', 'Comedy', 'Animation', 'Family', 'Fantasy', 'Musical']
        female_middle_genres = ['Romance', 'Drama', 'Comedy', 'Thriller', 'Family', 'Mystery']
        female_older_genres = ['Drama', 'Romance', 'Biography', 'Comedy', 'History']
        
        # Generate training data
        for _ in range(3000):
            age = np.random.randint(13, 75)
            gender = np.random.choice(['Male', 'Female'])
            
            # Select preferred genres based on demographics
            if gender == 'Male':
                if age < 25:
                    preferred_genres = male_young_genres
                elif age < 45:
                    preferred_genres = male_middle_genres
                else:
                    preferred_genres = male_older_genres
            else:
                if age < 25:
                    preferred_genres = female_young_genres
                elif age < 45:
                    preferred_genres = female_middle_genres
                else:
                    preferred_genres = female_older_genres
            
            # Select movies that match preferred genres
            matching_movies = [movie for movie in movies 
                             if any(genre in preferred_genres for genre in movie['genres'])]
            
            if matching_movies:
                num_movies = min(np.random.randint(1, 5), len(matching_movies))
                selected_movies = np.random.choice(matching_movies, 
                                                 size=num_movies, 
                                                 replace=False)
                
                for movie in selected_movies:
                    genre_match_score = len([g for g in movie['genres'] if g in preferred_genres])
                    preference_score = (movie['rating'] / 10.0) * (1 + genre_match_score * 0.2)
                    
                    data.append({
                        'age': age,
                        'gender': gender,
                        'movie_id': movie['id'],
                        'movie_title': movie['title'],
                        'primary_genre': movie['primary_genre'],
                        'rating': movie['rating'],
                        'preference_score': preference_score + np.random.normal(0, 0.1)
                    })
        
        return pd.DataFrame(data)
    
    def train_model(self):
        """Train the recommendation model"""
        print("Training model with sample movie data...")
        
        movies = self.get_sample_movies()
        df = self.create_training_data(movies)
        
        # Prepare encoders
        le_gender = LabelEncoder()
        le_movie = LabelEncoder()
        
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['movie_encoded'] = le_movie.fit_transform(df['movie_title'])
        
        # Features and target
        X = df[['age', 'gender_encoded']].values
        y = df['movie_encoded'].values
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X, y)
        
        # Store model data
        self.model_data = {
            'model': model,
            'gender_encoder': le_gender,
            'movie_encoder': le_movie,
            'training_df': df,
            'movies_data': movies,
            'last_updated': datetime.now()
        }
        
        print(f"Model trained with {len(df)} interactions and {len(movies)} movies!")
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            with open('omdb_movie_model.pkl', 'rb') as f:
                self.model_data = pickle.load(f)
                print("Loaded existing model")
        except FileNotFoundError:
            print("Creating new model...")
            self.train_model()
            # Save the model
            with open('omdb_movie_model.pkl', 'wb') as f:
                pickle.dump(self.model_data, f)
            print("Model saved to omdb_movie_model.pkl")
    
    def get_recommendations(self, age, gender, num_recommendations=6):
        """Get personalized movie recommendations"""
        try:
            if not self.model_data:
                return []
            
            # Encode input
            gender_encoded = self.model_data['gender_encoder'].transform([gender])[0]
            input_features = np.array([[age, gender_encoded]])
            
            # Get predictions
            probabilities = self.model_data['model'].predict_proba(input_features)[0]
            top_indices = np.argsort(probabilities)[-num_recommendations:][::-1]
            
            # Get recommended movies
            recommended_movie_names = self.model_data['movie_encoder'].inverse_transform(top_indices)
            
            # Match with movie data
            recommendations = []
            for i, movie_name in enumerate(recommended_movie_names):
                movie_data = next((m for m in self.model_data['movies_data'] if m['title'] == movie_name), None)
                if movie_data:
                    recommendations.append({
                        'title': movie_data['title'],
                        'genre': movie_data['primary_genre'],
                        'genres': movie_data['genres'],
                        'rating': movie_data['rating'],
                        'confidence': round(float(probabilities[top_indices[i]]) * 100, 1),
                        'release_date': movie_data['release_date'],
                        'year': movie_data['year'],
                        'director': movie_data.get('director', 'Unknown'),
                        'runtime': movie_data.get('runtime', 'Unknown'),
                        'overview': movie_data['overview'][:200] + "..." if len(movie_data['overview']) > 200 else movie_data['overview'],
                        'poster_url': movie_data['poster_path'],
                        'popularity': round(movie_data['popularity'], 1),
                        'imdb_id': movie_data.get('imdb_id', '')
                    })
            
            return recommendations
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

# Initialize the recommendation system
recommender = MovieRecommendationSystem()

@app.route('/')
def home():
    """Serve the main 3D interface"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations API endpoint"""
    try:
        data = request.get_json()
        age = int(data['age'])
        gender = data['gender']
        
        # Validate input
        if age < 13 or age > 100:
            return jsonify({'error': 'Age must be between 13 and 100'}), 400
        
        if gender not in ['Male', 'Female']:
            return jsonify({'error': 'Gender must be Male or Female'}), 400
        
        # Get recommendations
        recommendations = recommender.get_recommendations(age, gender)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'user_info': {'age': age, 'gender': gender},
            'timestamp': datetime.now().isoformat(),
            'total': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_movies():
    """Search for movies by title"""
    try:
        data = request.get_json()
        search_term = data.get('search_term', '').strip().lower()
        
        if not search_term:
            return jsonify({'error': 'Search term is required'}), 400
        
        # Search in our movie database
        all_movies = recommender.model_data['movies_data'] if recommender.model_data else []
        results = [movie for movie in all_movies 
                  if search_term in movie['title'].lower()]
        
        return jsonify({
            'success': True,
            'results': results[:10],  # Limit to 10 results
            'search_term': search_term,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refresh-movies', methods=['POST'])
def refresh_movies():
    """Force refresh of movie data"""
    try:
        recommender.train_model()
        
        # Save updated model
        with open('omdb_movie_model.pkl', 'wb') as f:
            pickle.dump(recommender.model_data, f)
        
        movie_count = len(recommender.model_data['movies_data']) if recommender.model_data else 0
        
        return jsonify({
            'success': True,
            'message': f'Refreshed with {movie_count} movies',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/movie-details/<movie_title>')
def get_movie_details(movie_title):
    """Get detailed information about a specific movie"""
    try:
        all_movies = recommender.model_data['movies_data'] if recommender.model_data else []
        movie_data = next((m for m in all_movies if m['title'].lower() == movie_title.lower()), None)
        
        if movie_data:
            return jsonify({
                'success': True,
                'movie': movie_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Movie not found'
            }), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    model_info = {
        'model_loaded': recommender.model_data is not None,
        'total_movies': len(recommender.model_data['movies_data']) if recommender.model_data else 0,
        'last_updated': recommender.model_data['last_updated'].isoformat() if recommender.model_data else None
    }
    
    return jsonify({
        'status': 'online',
        'version': '1.0',
        'model_info': model_info,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎬 Starting Movie Recommendation System...")
    print("=" * 50)
    print(f"🚀 Server running at: http://localhost:5000")
    print(f"📱 3D Interface: http://localhost:5000")
    print(f"🔗 API Status: http://localhost:5000/api/status")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  /                    - 3D Web Interface")
    print("  POST /recommend           - Get recommendations")
    print("  POST /search              - Search movies")
    print("  POST /refresh-movies      - Refresh database")
    print("  GET  /movie-details/<id>  - Movie details")
    print("  GET  /api/status          - API status")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

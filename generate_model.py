#!/usr/bin/env python3
"""
generate_model.py - Creates omdb_movie_model.pkl for Flask app
Run this script to generate the machine learning model pickle file
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_movie_database():
    """Complete movie database with current popular movies"""
    return [
        {
            'id': 1, 'title': 'Spider-Man: No Way Home', 'primary_genre': 'Action',
            'genres': ['Action', 'Adventure', 'Sci-Fi'], 'rating': 8.4,
            'release_date': '17 Dec 2021', 'popularity': 95.5,
            'overview': 'With Spider-Man\'s identity now revealed, Peter asks Doctor Strange for help. When a spell goes wrong, dangerous foes from other worlds start to appear.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BZWMyYzFjYTYtNTRjYi00OGExLWE2YzgtOGRmYjAxZTU3NzBiXkEyXkFqcGdeQXVyMzQ0MzA0NTM@._V1_SX300.jpg',
            'year': '2021', 'director': 'Jon Watts', 'runtime': '148 min', 'imdb_id': 'tt10872600'
        },
        {
            'id': 2, 'title': 'Top Gun: Maverick', 'primary_genre': 'Action',
            'genres': ['Action', 'Drama'], 'rating': 8.3,
            'release_date': '27 May 2022', 'popularity': 89.2,
            'overview': 'After thirty years, Maverick is still pushing the envelope as a top naval aviator, but must confront ghosts of his past.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BZWYzOGEwNTgtNWU3NS00ZTQ0LWJkODUtMmVhMjIwMjA1ZmQwXkEyXkFqcGdeQXVyMjkwOTAyMDU@._V1_SX300.jpg',
            'year': '2022', 'director': 'Joseph Kosinski', 'runtime': '131 min', 'imdb_id': 'tt1745960'
        },
        {
            'id': 3, 'title': 'The Batman', 'primary_genre': 'Action',
            'genres': ['Action', 'Crime', 'Drama'], 'rating': 7.8,
            'release_date': '04 Mar 2022', 'popularity': 87.3,
            'overview': 'When a sadistic serial killer begins murdering key political figures in Gotham, Batman is forced to investigate the city\'s hidden corruption.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BOGE2NWUwMDItMjA4Yi00N2Y3LWJjMzEtMDJjZTMzZTdlZGE5XkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_SX300.jpg',
            'year': '2022', 'director': 'Matt Reeves', 'runtime': '176 min', 'imdb_id': 'tt1877830'
        },
        {
            'id': 4, 'title': 'Black Panther: Wakanda Forever', 'primary_genre': 'Action',
            'genres': ['Action', 'Adventure', 'Drama'], 'rating': 6.7,
            'release_date': '11 Nov 2022', 'popularity': 84.1,
            'overview': 'Queen Ramonda, Shuri, M\'Baku, Okoye and the Dora Milaje fight to protect their nation from intervening world powers.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BNTM4NjIxNmEtYWE5NS00NDczLTkyNWQtYThhNmQyZGQzMjM0XkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_SX300.jpg',
            'year': '2022', 'director': 'Ryan Coogler', 'runtime': '161 min', 'imdb_id': 'tt9114286'
        },
        {
            'id': 5, 'title': 'Avatar: The Way of Water', 'primary_genre': 'Sci-Fi',
            'genres': ['Action', 'Adventure', 'Family', 'Sci-Fi'], 'rating': 7.6,
            'release_date': '16 Dec 2022', 'popularity': 92.8,
            'overview': 'Jake Sully lives with his newfound family formed on the extrasolar moon Pandora. Once a familiar threat returns to finish what was previously started.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BYjhiNjBlODctY2ZiOC00YjVlLWFlNzAtNTVhNzM1YjI1NzMxXkEyXkFqcGdeQXVyMjQxNTE1MDA@._V1_SX300.jpg',
            'year': '2022', 'director': 'James Cameron', 'runtime': '192 min', 'imdb_id': 'tt1630029'
        },
        {
            'id': 6, 'title': 'Everything Everywhere All at Once', 'primary_genre': 'Comedy',
            'genres': ['Comedy', 'Adventure', 'Sci-Fi'], 'rating': 7.8,
            'release_date': '08 Apr 2022', 'popularity': 85.4,
            'overview': 'An aging Chinese immigrant is swept up in an insane adventure, where she alone can save what\'s important to her.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BYTdiOTIyZTQtNmQ1OS00NjZlLWIyMTgtYzk5Y2M3ZDVmMDk1XkEyXkFqcGdeQXVyMTAzMDM4MjM0._V1_SX300.jpg',
            'year': '2022', 'director': 'Daniel Kwan, Daniel Scheinert', 'runtime': '139 min', 'imdb_id': 'tt6710474'
        },
        {
            'id': 7, 'title': 'Nope', 'primary_genre': 'Horror',
            'genres': ['Horror', 'Mystery', 'Sci-Fi'], 'rating': 6.8,
            'release_date': '22 Jul 2022', 'popularity': 76.2,
            'overview': 'The residents of a lonely gulch deal with a mysterious threat that reproduces itself by means of any reflective surface.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BMzZhZDg0M2YtZDBiYi00ZTJmLTdhZDctYzFiNTExZGZiZWUxXkEyXkFqcGdeQXVyMTM1MTE1NDMx._V1_SX300.jpg',
            'year': '2022', 'director': 'Jordan Peele', 'runtime': '130 min', 'imdb_id': 'tt10954984'
        },
        {
            'id': 8, 'title': 'Minions: The Rise of Gru', 'primary_genre': 'Animation',
            'genres': ['Animation', 'Comedy', 'Family'], 'rating': 6.5,
            'release_date': '01 Jul 2022', 'popularity': 81.9,
            'overview': 'The untold story of one twelve-year-old\'s dream to become the world\'s greatest supervillain.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BNjUwMTVkMWItNjUxMi00OTU4LTk4ZGQtMzE4NDJhNDI5YWEyXkEyXkFqcGdeQXVyODc0OTEyNDU@._V1_SX300.jpg',
            'year': '2022', 'director': 'Kyle Balda', 'runtime': '87 min', 'imdb_id': 'tt5113044'
        },
        {
            'id': 9, 'title': 'Thor: Love and Thunder', 'primary_genre': 'Action',
            'genres': ['Action', 'Adventure', 'Comedy'], 'rating': 6.2,
            'release_date': '08 Jul 2022', 'popularity': 73.8,
            'overview': 'Thor enlists the help of Valkyrie, Korg and ex-girlfriend Jane Foster to fight Gorr the God Butcher, who intends to make the gods extinct.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BYmMxZWRiMTgtZjM0Ny00NDQxLWIxYWQtZDI2NWE1OWZhNjhmXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_SX300.jpg',
            'year': '2022', 'director': 'Taika Waititi', 'runtime': '119 min', 'imdb_id': 'tt10648342'
        },
        {
            'id': 10, 'title': 'Doctor Strange in the Multiverse of Madness', 'primary_genre': 'Action',
            'genres': ['Action', 'Adventure', 'Horror'], 'rating': 6.9,
            'release_date': '06 May 2022', 'popularity': 82.7,
            'overview': 'Doctor Strange teams up with a mysterious teenage girl from his dreams who can travel across multiverses.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BNWM0ZGJlMzMtZmYwMi00NzI3LTgzMzMtZWIwMSZkM2Y2ODllXkEyXkFqcGdeQXVyMTM1MTE1NDMx._V1_SX300.jpg',
            'year': '2022', 'director': 'Sam Raimi', 'runtime': '126 min', 'imdb_id': 'tt9419884'
        },
        {
            'id': 11, 'title': 'Jurassic World Dominion', 'primary_genre': 'Action',
            'genres': ['Action', 'Adventure', 'Sci-Fi'], 'rating': 5.6,
            'release_date': '10 Jun 2022', 'popularity': 79.4,
            'overview': 'Four years after the destruction of Isla Nublar, dinosaurs now live and hunt alongside humans all over the world.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BZGEwYmMwZmMtMTQ1MS00YzQ2LWI2MjEtYmZhZjIzNjdkNjBmXkEyXkFqcGdeQXVyMTA3MDk2NDg2._V1_SX300.jpg',
            'year': '2022', 'director': 'Colin Trevorrow', 'runtime': '147 min', 'imdb_id': 'tt8041270'
        },
        {
            'id': 12, 'title': 'Lightyear', 'primary_genre': 'Animation',
            'genres': ['Animation', 'Adventure', 'Family'], 'rating': 6.1,
            'release_date': '17 Jun 2022', 'popularity': 71.3,
            'overview': 'The story of Buzz Lightyear and his adventures to infinity and beyond.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BMTkxNzE2NDY5OV5BMl5BanBnXkFtZTcwNDM1NDE5Nw@@._V1_SX300.jpg',
            'year': '2022', 'director': 'Angus MacLane', 'runtime': '105 min', 'imdb_id': 'tt10298810'
        },
        {
            'id': 13, 'title': 'The Northman', 'primary_genre': 'Action',
            'genres': ['Action', 'Adventure', 'Drama'], 'rating': 7.0,
            'release_date': '22 Apr 2022', 'popularity': 68.9,
            'overview': 'A young Viking prince on a quest to avenge his father\'s murder.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BMzVlMmY2NTctODgwOC00NDMzLWE3YWYtY2ZmZThiOWU1NTcyXkEyXkFqcGdeQXVyNTAzNzgwNTg@._V1_SX300.jpg',
            'year': '2022', 'director': 'Robert Eggers', 'runtime': '137 min', 'imdb_id': 'tt11138512'
        },
        {
            'id': 14, 'title': 'Turning Red', 'primary_genre': 'Animation',
            'genres': ['Animation', 'Comedy', 'Drama'], 'rating': 7.0,
            'release_date': '11 Mar 2022', 'popularity': 75.6,
            'overview': 'A thirteen-year-old girl named Mei Lee turns into a giant red panda whenever she gets too excited.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BNzc5Mjk1NzgtNjE4MS00NGMzLTk4M2QtOWUzNjkzOTJhODE0XkEyXkFqcGdeQXVyMTA3MDk2NDg2._V1_SX300.jpg',
            'year': '2022', 'director': 'Domee Shi', 'runtime': '100 min', 'imdb_id': 'tt8097030'
        },
        {
            'id': 15, 'title': 'Sonic the Hedgehog 2', 'primary_genre': 'Adventure',
            'genres': ['Adventure', 'Comedy', 'Family'], 'rating': 6.5,
            'release_date': '08 Apr 2022', 'popularity': 77.2,
            'overview': 'When the manic Dr Robotnik returns with a new partner, Knuckles, Sonic and his new friend Tails face a dangerous adventure.',
            'poster_path': 'https://m.media-amazon.com/images/M/MV5BM2Q2NzVkZjMtNWRlNy00MTEzLWFjM2QtYjlkMGZkZGNmMjA5XkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_SX300.jpg',
            'year': '2022', 'director': 'Jeff Fowler', 'runtime': '122 min', 'imdb_id': 'tt12412888'
        }
    ]

def prepare_features(movies_data):
    """Prepare features for machine learning model"""
    df = pd.DataFrame(movies_data)
    
    # Extract runtime as numeric value
    df['runtime_minutes'] = df['runtime'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) and x != 'N/A' else 120)
    
    # Extract year as numeric
    df['year_numeric'] = df['year'].apply(lambda x: int(x) if isinstance(x, str) and x != 'N/A' else 2022)
    
    # Count genres
    df['genre_count'] = df['genres'].apply(len)
    
    # Create binary features for popular genres
    all_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Animation', 'Family', 'Crime', 'Mystery']
    for genre in all_genres:
        df[f'has_{genre.lower()}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    
    # Encode primary genre
    genre_encoder = LabelEncoder()
    df['primary_genre_encoded'] = genre_encoder.fit_transform(df['primary_genre'])
    
    # Create popularity categories (target variable)
    def categorize_popularity(rating, popularity):
        if rating >= 8.0 and popularity >= 90:
            return 'Blockbuster'
        elif rating >= 7.5 and popularity >= 80:
            return 'Hit'
        elif rating >= 6.5 and popularity >= 70:
            return 'Moderate Success'
        else:
            return 'Average'
    
    df['success_category'] = df.apply(lambda row: categorize_popularity(row['rating'], row['popularity']), axis=1)
    
    # Prepare feature matrix
    feature_columns = ['runtime_minutes', 'year_numeric', 'genre_count', 'primary_genre_encoded'] + \
                     [f'has_{genre.lower()}' for genre in all_genres]
    
    X = df[feature_columns]
    y = df['success_category']
    
    return X, y, df, genre_encoder

def create_recommendation_features(movies_data):
    """Create features for movie recommendation"""
    df = pd.DataFrame(movies_data)
    
    # Create similarity features based on genres, ratings, and popularity
    recommendation_data = {
        'movies': df.to_dict('records'),
        'genre_similarity_matrix': {},
        'rating_ranges': {
            'excellent': (8.0, 10.0),
            'good': (7.0, 7.9),
            'average': (6.0, 6.9),
            'below_average': (0, 5.9)
        }
    }
    
    # Calculate genre similarity matrix
    unique_genres = set()
    for movie in movies_data:
        unique_genres.update(movie['genres'])
    
    unique_genres = list(unique_genres)
    
    for i, movie1 in enumerate(df.to_dict('records')):
        for j, movie2 in enumerate(df.to_dict('records')):
            if i != j:
                # Calculate Jaccard similarity for genres
                set1 = set(movie1['genres'])
                set2 = set(movie2['genres'])
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                
                key = f"{movie1['imdb_id']}_{movie2['imdb_id']}"
                recommendation_data['genre_similarity_matrix'][key] = similarity
    
    return recommendation_data

def generate_model():
    """Generate and save the machine learning model"""
    print("Loading movie database...")
    movies_data = get_movie_database()
    
    print("Preparing features...")
    X, y, df, genre_encoder = prepare_features(movies_data)
    
    print("Training Random Forest model...")
    # Train a Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    rf_model.fit(X, y)
    
    print("Creating recommendation data...")
    recommendation_data = create_recommendation_features(movies_data)
    
    # Create the complete model package
    model_package = {
        'classifier': rf_model,
        'genre_encoder': genre_encoder,
        'feature_columns': X.columns.tolist(),
        'movies_database': movies_data,
        'recommendation_data': recommendation_data,
        'success_categories': list(rf_model.classes_),
        'model_info': {
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'RandomForestClassifier',
            'n_movies': len(movies_data),
            'features': len(X.columns)
        }
    }
    
    print("Saving model to omdb_movie_model.pkl...")
    with open('omdb_movie_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved successfully!")
    print(f"- Total movies: {len(movies_data)}")
    print(f"- Features: {len(X.columns)}")
    print(f"- Success categories: {list(rf_model.classes_)}")
    print(f"- Model accuracy on training data: {rf_model.score(X, y):.3f}")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))

def test_model():
    """Test the saved model"""
    try:
        print("\nTesting saved model...")
        with open('omdb_movie_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        # Test prediction with first movie
        test_movie = model_package['movies_database'][0]
        print(f"Test movie: {test_movie['title']}")
        print(f"Actual rating: {test_movie['rating']}")
        print(f"Actual popularity: {test_movie['popularity']}")
        
        # Create feature vector for test movie
        feature_vector = []
        runtime_minutes = int(test_movie['runtime'].split()[0])
        year_numeric = int(test_movie['year'])
        genre_count = len(test_movie['genres'])
        primary_genre_encoded = model_package['genre_encoder'].transform([test_movie['primary_genre']])[0]
        
        feature_vector.extend([runtime_minutes, year_numeric, genre_count, primary_genre_encoded])
        
        # Add genre binary features
        all_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Animation', 'Family', 'Crime', 'Mystery']
        for genre in all_genres:
            feature_vector.append(1 if genre in test_movie['genres'] else 0)
        
        # Make prediction
        prediction = model_package['classifier'].predict([feature_vector])[0]
        print(f"Predicted success category: {prediction}")
        
        print("Model test completed successfully!")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    generate_model()
    test_model()
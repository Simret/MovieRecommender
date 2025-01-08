import sys
from pathlib import Path
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


# Connect to neo4j instance on GCP
DB_URI = "neo4j+s://8364f4e7.databases.neo4j.io"
DB_USER = "neo4j"
DB_PASSWORD = "SwHqTGYLfNlZvLcfgXTPMqCOwq_E48z1A5qZlrWChBg"
AUTH = (DB_USER, DB_PASSWORD)

with GraphDatabase.driver(DB_URI, auth=AUTH) as driver:
    driver.verify_connectivity()

# Load the dataset
df = pd.read_csv('ratings_small.csv')
movies_df = pd.read_csv('movies_metadata.csv', low_memory=False)

df.head()

# Filter out rows with NaN titles
movies_df = movies_df.dropna(subset=['title'])
movies_df['movieId'] = pd.to_numeric(movies_df['movieId'], errors='coerce')
movies_df = movies_df.dropna(subset=['movieId'])
movies_df['movieId'] = movies_df['movieId'].astype(int)
# Create User and Movie graph in Neo4j
def create_user_movie_graph():
    with driver.session() as session:
        # Create Movie nodes
        for _, row in movies_df.iterrows():
            if pd.notna(row['title']):  # Ensuring title is not NaN
                session.run("""
                    MERGE (m:Movie {id: $movieId, title: $title, genres: $genres})
                """, movieId=row['movieId'], title=row['title'], genres=row['genres'])

        # Create User nodes and relationships
        for _, row in df.iterrows():
            session.run("""
                MERGE (u:User {id: $userId})
                MERGE (m:Movie {id: $movieId})
                CREATE (u)-[:RATED {rating: $rating}]->(m)
            """, userId=row['userId'], movieId=row['movieId'], rating=row['rating'])

create_user_movie_graph()


# Prepare data for collaborative filtering using Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Build the SVD model
model = SVD()
model.fit(trainset)

# Predict ratings for testset
predictions = model.test(testset)

# Generate top-N recommendations for a user
def collaborative_filtering_recommendation(user_id, n=10):
    # Get all unique movie IDs in the ratings data
    movie_ids = df['movieId'].unique()
    
    # Make predictions for each movie ID
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Take the top-N movie IDs
    recommended_movie_ids = [pred.iid for pred in predictions[:n]]
    
    # Ensure the movie IDs are in movies_df before mapping to titles
    movie_id_to_title = dict(zip(movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['movieId'], 
                                 movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title']))
    
    # Map recommended movie IDs to their titles
    recommended_movie_titles = [movie_id_to_title.get(movie_id, "Unknown Movie") for movie_id in recommended_movie_ids]

    return recommended_movie_titles

def context_based_recommendation(user_id, genre_preference):
  
    query = f"""
    MATCH (u:User)-[r:RATED]->(m:Movie)
    WHERE u.id = {user_id} AND m.genres CONTAINS '{genre_preference}'
    WITH m, AVG(r.rating) AS avg_rating
    RETURN m.title, avg_rating
    ORDER BY avg_rating DESC
    LIMIT 10
    """
    with driver.session() as session:
        result = session.run(query)
        recommendations = [record['m.title'] for record in result]

    # If content-based recommendations are empty, fall back to collaborative filtering
    if not recommendations:
        print("No content-based recommendations found. Using collaborative filtering as fallback.")
        recommendations = collaborative_filtering_recommendation(user_id)

    return recommendations

# Test
user_id = 1
genre_preference = "Action"  # Example genre
recommendations = context_based_recommendation(user_id, genre_preference)
print(f"Recommendations for user {user_id} with genre preference '{genre_preference}': {recommendations}")

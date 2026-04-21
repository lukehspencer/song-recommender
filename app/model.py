import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_song_vector(df, track_name, artists):
    if isinstance(artists, list):
        artists = artists[0]  # Take the first artist if it's a list
    song_row = df[(df['track_name'] == track_name) & (df['artists'] == artists)]
    if not song_row.empty:
        return song_row.drop(columns=["track_name", "artists"]).iloc[0].values.astype(float)
    else:
        return None
    
def rating_to_weight(rating):
    return rating / 5.0

def build_user_profile(df, songs, artists, ratings):
    vectors = []
    weights = []
    
    for song, artist, rating in zip(songs, artists, ratings):
        v = get_song_vector(df, song, artist)
        if v is None:
            continue  # ← skip songs not found in dataset
        w = rating_to_weight(rating)
        vectors.append(v)
        weights.append(w)
    
    if not vectors:
        return None
    
    vectors = np.vstack(vectors)
    weights = np.array(weights)
    
    return np.dot(weights, vectors) / np.sum(np.abs(weights))
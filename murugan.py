import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components  # ✅ Add this


# Load Dataset
df = pd.read_csv("C:/my project filess/Music.csv")

st.title("🎵 TuneMatch")

# Load your dataset
df = pd.read_csv("Music.csv")

# Clean and fill missing data
df.fillna('', inplace=True)

# 🎯 Create a list of song titles
song_titles = df['name'].tolist()

# 🔍 Add search bar
selected_song = st.selectbox("Search for a song", song_titles)

# 🔎 Get index of selected song
song_index = df[df['name'] == selected_song].index[0]

# 👉 Select features for similarity (you can customize)
features = ['danceability', 'energy', 'valence']  # adjust based on your dataset

# 🎯 Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# 🔁 Find similar songs
similarity = cosine_similarity([df_scaled[song_index]], df_scaled)
similar_indices = similarity[0].argsort()[::-1][1:6]  # Top 5 similar songs

# ✅ Display recommendations
st.subheader("🎶 Recommended Songs")

for i in similar_indices:
    song = df.iloc[i]
    st.markdown(f"### 🎵 {song['name']} - *{song['artist']}*")

    # 🖼️ Show image if valid
    if song['img'].startswith("http"):
        st.image(song['img'], width=200)

    # ▶️ Play preview if available
    if song['preview'].startswith("http"):
        st.audio(song['preview'], format="audio/mp3")

    # ▶️ Embed Spotify player
    if 'spotify_id' in song and song['spotify_id']:
        spotify_url = f"https://open.spotify.com/embed/track/{song['spotify_id']}"
        components.iframe(spotify_url, height=80)

    st.markdown("---")

import numpy as np
df['popularity'] = np.random.randint(50, 100, size=len(df))
st.subheader("🔥 Trending Songs")
top_songs = df.sort_values(by='popularity', ascending=False).head(5)

for i, row in top_songs.iterrows():
    st.markdown(f"**🎧 {row['name']}** - {row['artist']}  | Popularity: {row['popularity']}")

mood = st.radio("Pick your mood 🎭", ["Happy", "Chill", "Energetic"])

if mood == "Happy":
    mood_df = df[df['valence'] > 0.7]
elif mood == "Chill":
    mood_df = df[(df['valence'] < 0.5) & (df['energy'] < 0.5)]
else:
    mood_df = df[(df['energy'] > 0.7) & (df['danceability'] > 0.6)]

import streamlit as st
import base64

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            color: white;  /* 🌟 This sets all text to white */
        }}
        h1, h2, h3, h4, h5, h6, p, div, span, label {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call with your image path
set_background("C:/my project filess/song-7058726_640.jpg")

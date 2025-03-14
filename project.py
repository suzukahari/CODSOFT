import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import difflib
import base64

# Function to convert image to base64
def image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        st.error(f"Error: The file '{image_path}' was not found.")
        return None

# Load dataset
data = pd.read_csv("Book2.csv")

# Extract 'Year' from 'release_date'
if 'release_date' in data.columns:
    data['Year'] = pd.to_datetime(data['release_date'], errors='coerce').dt.year
    data.dropna(subset=['Year'], inplace=True)
    data['Year'] = data['Year'].astype(int)

# Convert 'original_language' to string format
if 'original_language' in data.columns:
    data['original_language'] = data['original_language'].astype(str)

# Function for searching movies based on user input
def search_movies(user_language, user_year, user_genre, user_min_rating):
    available_genres = set(', '.join(data['genres'].dropna()).split(', '))
    closest_match = difflib.get_close_matches(user_genre, available_genres, n=1, cutoff=0.6)
    if not closest_match:
        return None
    user_genre = closest_match[0]

    filtered_data = data[(
        data['original_language'].str.lower() == user_language.lower()) &
        (data['Year'] == user_year) &
        (data['genres'].str.contains(user_genre, case=False, na=False)) &
        (data['vote_average'] >= user_min_rating)
    ].sort_values(by='vote_average', ascending=False)
    return filtered_data

# Function for visualizing the filtered movie data
def visualize_movies(filtered_data):
    st.write("### Top Movies by Rating")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        x=filtered_data['vote_average'], 
        y=filtered_data['title'], 
        palette="coolwarm", 
        order=filtered_data.sort_values('vote_average', ascending=False)['title'],
        ax=ax
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Movie Title")
    ax.set_title("Top Movies by Rating")
    st.pyplot(fig)

    st.write("### Popularity vs Rating")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(x=filtered_data['popularity'], y=filtered_data['vote_average'], hue=filtered_data['title'], s=100, ax=ax)
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Rating")
    ax.set_title("Popularity vs. Rating")
    st.pyplot(fig)

    st.write("### Movie Ratings Distribution by Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Year', y='vote_average', data=filtered_data, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Movie Ratings Distribution by Year")
    st.pyplot(fig)

    st.write("### Top 10 Movie Genres")
    fig, ax = plt.subplots(figsize=(8, 8))
    genre_counts = filtered_data['genres'].value_counts().head(10)
    plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
    ax.set_title("Top 10 Movie Genres")
    st.pyplot(fig)

# Function to build and evaluate the Random Forest model
def build_model():
    if 'genres' in data.columns:
        genres = data['genres'].str.get_dummies(', ')
        data = pd.concat([data, genres], axis=1)
        data.drop('genres', axis=1, inplace=True)

    columns_to_drop = ['title', 'overview', 'original_language', 'release_date']
    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, errors='ignore')
    y = data['vote_average']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"### Best Model Parameters: {random_search.best_params_}")
    st.write(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Function to display the correlation heatmap
def display_heatmap():
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

# Streamlit UI
def app():
    image_path = r'C:\tamil\vinoth\back.jpg'  # Update to your correct image path
    img_base64 = image_to_base64(image_path)
    
    if img_base64:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    
        st.markdown("""
<h1 style="text-align: center; color: #ffffff ; border: 5px solid #ffffff ; padding: 25px; border-radius: 20px; font-family: 'Arial', sans-serif; font-size: 40px;">
WELCOME TO THE MOVIE FILTERING 📽
</h1>
""", unsafe_allow_html=True)


    # Dropdown menu for user inputs
    languages = sorted(data['original_language'].unique())  # Get unique languages
    user_language = st.selectbox("Select the language:", languages)

    years = sorted(data['Year'].unique())  # Get unique years
    user_year = st.selectbox("Select the year:", years)

    genres = sorted(set(', '.join(data['genres'].dropna()).split(', ')))  # Get unique genres
    user_genre = st.selectbox("Select the genre:", genres)

    user_min_rating = st.slider("Select the minimum rating (0-10):", 0.0, 10.0, 5.0, 0.1)

    if st.button("Search Movies"):
        filtered_data = search_movies(user_language, user_year, user_genre, user_min_rating)
        
        if filtered_data is not None and not filtered_data.empty:
            st.write(f"### Movies matching your criteria:")
            st.dataframe(filtered_data[['title', 'Year', 'genres', 'vote_average', 'popularity']])

            visualize_movies(filtered_data)
        else:
            st.write(f"No movies found matching your criteria.")

# Run the Streamlit app
if __name__ == "__main__":
    app()

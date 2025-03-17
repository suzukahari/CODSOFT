import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="📚 Book Recommendation System", layout="wide")

st.title("📚 Book Recommendation System")
st.write("Get personalized book recommendations with book cover images!")

# Load and process data
@st.cache_data
def load_data():
    books = pd.read_csv("D:/Book reviews/BX_Books.csv", sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    ratings = pd.read_csv("D:/Book reviews/BX-Book-Ratings.csv", sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    users = pd.read_csv("D:/Book reviews/BX-Users.csv", sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    return books, ratings, users

books, ratings, users = load_data()
books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
users.columns = ['User-ID', 'Location', 'Age']

# Popular books
popular_books = ratings.groupby('ISBN').count()['Book-Rating'].reset_index()
popular_books.rename(columns={'Book-Rating': 'Rating-Count'}, inplace=True)
popular_books = popular_books.merge(books, on='ISBN')
popular_books = popular_books[popular_books['Rating-Count'] >= 200].sort_values('Rating-Count', ascending=False).head(10)

# 📸 Show top books with images
st.subheader("📊 Top 10 Popular Books")

cols = st.columns(5)
for i, row in popular_books.iterrows():
    with cols[i % 5]:
        st.image(row['Image-URL-M'], width=120)
        st.caption(f"**{row['Book-Title']}**\n\n📖 {row['Book-Author']}")

# Pivot table
@st.cache_data
def create_pivot():
    pt = ratings.merge(books, on='ISBN')
    final_ratings = pt.groupby('User-ID').filter(lambda x: len(x) >= 50)
    book_ratings = final_ratings.groupby('Book-Title').filter(lambda x: len(x) >= 50)
    pt = book_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
    return pt, book_ratings

pt, book_ratings = create_pivot()

# Similarity matrix
@st.cache_data
def get_similarity():
    similarity = cosine_similarity(pt)
    return similarity

similarity_score = get_similarity()

# Recommend books
def recommend(book_name):
    book_name = book_name.strip().lower()
    book_names = [b.lower() for b in pt.index]
    if book_name not in book_names:
        return []

    index = book_names.index(book_name)
    distances = similarity_score[index]
    book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_books = []
    for i in book_list:
        title = pt.index[i[0]]
        info = books[books['Book-Title'] == title].drop_duplicates('Book-Title')
        if not info.empty:
            recommended_books.append({
                'title': title,
                'author': info.iloc[0]['Book-Author'],
                'image': info.iloc[0]['Image-URL-M']
            })
    return recommended_books

# UI for recommendation
st.subheader("🔍 Recommend Books")
book_input = st.text_input("Enter a Book Title (case-insensitive)", "")

if st.button("Recommend"):
    if book_input:
        recommendations = recommend(book_input)
        if recommendations:
            st.success(f"Top Recommendations for '{book_input.title()}':")
            cols = st.columns(5)
            for idx, book in enumerate(recommendations):
                with cols[idx % 5]:
                    st.image(book['image'], width=120)
                    st.caption(f"**{book['title']}**\n\n📖 {book['author']}")
        else:
            st.error("Sorry! Book not found or not enough data for recommendations.")
    else:
        st.warning("Please enter a book title.")

st.markdown("---")
st.caption("Created by Hari 💡 | Powered by Streamlit, Pandas & 💻 Machine Learning")

import streamlit as st
import pandas as pd
import requests

# Load the data
books = pd.read_csv("D:/Book reviews/BX_Books.csv", sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)

books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

# Streamlit UI
st.title("📚 Book Explorer by Genre/Theme")

# User Input
query = st.text_input("Search for books by theme (e.g., comedy, love, adventure, fight):")

if query:
    query_lower = query.lower()

    # Search in title and author fields
    matching_books = books[
        books['Book-Title'].str.lower().str.contains(query_lower, na=False) |
        books['Book-Author'].str.lower().str.contains(query_lower, na=False)
    ].drop_duplicates(subset='Book-Title').head(20)

    if not matching_books.empty:
        st.subheader(f"🔍 Showing books related to: '{query}'")

        for index, row in matching_books.iterrows():
            st.image(row['Image-URL-M'], width=100)
            st.markdown(f"**{row['Book-Title']}**")
            st.markdown(f"*Author:* {row['Book-Author']}")
            st.markdown(f"*Published:* {row['Year-Of-Publication']}")
            st.markdown("---")
    else:
        st.warning("No books found for that theme. Try different keywords.")

else:
    st.info("Please enter a theme or keyword to search for books.")

st.subheader("📚 Rate & Review a Book")

book_to_review = st.selectbox("Select a book to review:", books['Book-Title'].unique())

user_rating = st.slider("Your Rating (0 to 10):", 0, 10, 5)

user_review = st.text_area("Write your review here:")

if st.button("Submit Review"):
    with open("reviews.txt", "a", encoding='utf-8') as f:
        f.write(f"{book_to_review} | Rating: {user_rating} | Review: {user_review}\n")
    st.success("✅ Review submitted!")

if st.checkbox("Show All Reviews"):
    st.subheader("📝 All Reviews")
    try:
        with open("reviews.txt", "r", encoding='utf-8') as f:
            reviews = f.readlines()
        for review in reviews:
            st.markdown(review)
    except FileNotFoundError:
        st.info("No reviews yet.")

import speech_recognition as sr

st.subheader("🎤 Voice Search for Books")

if st.button("Start Voice Search"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            results = books[books['Book-Title'].str.contains(query, case=False, na=False)]
            if not results.empty:
                st.write(f"🔍 Books matching '{query}':")
                st.dataframe(results[['Book-Title', 'Book-Author']])
            else:
                st.warning("No matching books found.")
        except sr.UnknownValueError:
            st.error("Sorry, couldn't understand your voice.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")

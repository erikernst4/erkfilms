import os
from time import time

from typing import List
import cohere
import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils import seed_everything, streamlit_header_and_footer_setup

torchfy = lambda x: torch.as_tensor(x, dtype=torch.float32)

def get_similarity_combined(target: List[float], candidates_1: List[float], candidates_2: List[float], top_k: int):
    target = torchfy(target)
    candidates_1 = torchfy(candidates_1).transpose(0, 1)
    cos_scores_1 = torch.mm(target, candidates_1)
    
    candidates_2 = torchfy(candidates_2).transpose(0, 1)
    cos_scores_2 = torch.mm(target, candidates_2)

    cos_scores = cos_scores_1 + cos_scores_2
    scores, indices = torch.topk(cos_scores, k=top_k)
    similarity_hits = [{'id': idx, 'score': score} for idx, score in zip(indices[0].tolist(), scores[0].tolist())]

    return similarity_hits

def get_similarity(target: List[float], candidates: List[float], top_k: int):
    candidates = torchfy(candidates).transpose(0, 1)
    target = torchfy(target)
    cos_scores = torch.mm(target, candidates)

    scores, indices = torch.topk(cos_scores, k=top_k)
    similarity_hits = [{'id': idx, 'score': score} for idx, score in zip(indices[0].tolist(), scores[0].tolist())]

    return similarity_hits


import imdb

ia = imdb.IMDb()

seed_everything(3777)

st.set_page_config(layout="wide")
streamlit_header_and_footer_setup()
st.markdown("## Movies Search and Recommendation 🔍 🎬 🍿 ")

# model_name: str = 'multilingual-2210-alpha'
model_name: str = 'multilingual-22-12'
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)


@st.cache()
def setup():
    PODCAST_FIELDS = [
        "movieId", "id", "imdb_id", "original_title", "title", "overview", "genres", "release_date", "language_code",
        "lang2idx", "language_name", "embeddings", "keywords_embedding", "vote_average"
    ]
    movies_df = pd.read_json("./data/movies_with_keywords_embeddings_31_03_2023.json", orient="index")
    movies_df = movies_df.dropna(subset=['imdb_id', "keywords_embedding"])
    movies_df = movies_df.fillna("")
    movies_df['movieId'] = movies_df.index
    movies_df = movies_df[PODCAST_FIELDS]
    df_watchlist = pd.read_csv("./data/letterboxd/watchlist.csv")
    df_watched = pd.read_csv("./data/letterboxd/watched.csv")
    return movies_df, df_watchlist, df_watched


movies_df, df_watchlist, df_watched = setup()

movies_available_languages = sorted(movies_df.language_name.unique().tolist())
images_cache = {}

query_text = st.text_input("Let's retrieve similar text 🔍", "")
search_expander = st.expander(label='Search Fields, Expand me!')
with search_expander:
    'Hello there!'
    limit = st.slider("limit", min_value=1, max_value=100, value=5, step=1)
    min_imdb_rating = st.slider("min imdb rating", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    selected_languages = st.multiselect(
        label=f"Desired languages | Number of Unique languages: {len(movies_available_languages)}",
        options=movies_available_languages,
    )
    only_not_watched = st.checkbox('Only show non watched movies')
    only_in_watchlist = st.checkbox('Only show movies from watchlist')
    output_fields: List[str] = [
        "movieId", "id", "imdb_id", "original_title", "title", "overview", "genres", "release_date", "language_code",
        "lang2idx", "language_name"
    ]

filters = [selected_languages, only_not_watched, only_in_watchlist]
retrieve_button = st.button("retrieve! 🧐")
if query_text or retrieve_button:
    sub_movies_df = movies_df.copy()
    rating_filter = movies_df["vote_average"] > min_imdb_rating
    sub_movies_df = sub_movies_df[rating_filter]
    if selected_languages:
        selected_languages_str = "|".join(language for language in selected_languages)
        sub_movies_df = sub_movies_df[sub_movies_df["language_name"].str.contains(selected_languages_str, regex=True)]
        
    if only_in_watchlist:
        watchlist_names = df_watchlist["Name"].tolist()
        in_watchlist_filter = sub_movies_df["title"].isin(watchlist_names) | sub_movies_df["original_title"].isin(watchlist_names)
        sub_movies_df = sub_movies_df[in_watchlist_filter]
    if only_not_watched:
        watched_names = df_watchlist["Name"].tolist()
        watched_filter = sub_movies_df["title"].isin(watched_names) | sub_movies_df["original_title"].isin(watched_names)
        sub_movies_df = sub_movies_df[watched_filter]
    
    candidates_content = np.array(sub_movies_df.embeddings.values.tolist(), dtype=np.float32)
    candidates_keywords = np.array(sub_movies_df.keywords_embedding.values.tolist(), dtype=np.float32)
    
    print(f"Query: {query_text}")
    vectors_to_search = np.array(
        co.embed(model=model_name, texts=[query_text], truncate="RIGHT").embeddings,
        dtype=np.float32,
    )

    start_time = time()
    result = get_similarity_combined(vectors_to_search, candidates_content, candidates_keywords, top_k=limit)
    print(result)
    end_time = time()

    similar_results = {}
    for index, hit in enumerate(result):
        print(hit)
        similar_example = sub_movies_df.iloc[hit['id']]
        similar_results[index] = {podcast_field: similar_example[podcast_field] for podcast_field in output_fields}
        # similar_results[index].update({"distance": hit.distance})

    print("Similar Results:")
    print(similar_results)
    for index in range(0, len(similar_results), 5):
        cols = st.columns(5)
        for i in range(5):
            try:
                genres = [genre['name'] for genre in eval(similar_results[index + i]['genres'])]
                cols[i].markdown(f"**movieId**: {similar_results[index + i]['movieId']}")
                cols[i].markdown(f"**URL**: https://www.imdb.com/title/{similar_results[index + i]['imdb_id']}/")
                try:
                    imdb_id = similar_results[index + i]['imdb_id'].replace("tt", "")
                    image = images_cache[imdb_id] = images_cache.get(imdb_id, ia.get_movie(imdb_id).data['cover url'])
                    # cols[i].image(image, use_column_width=True)
                    cols[i].markdown(
                        f'<img src="{image}" style="width:100%;height:75%;border-radius: 5%;">',
                        unsafe_allow_html=True,
                    )
                except:
                    pass
                cols[i].markdown(f"**Original Title**: {similar_results[index + i]['original_title']}")
                cols[i].markdown(f"**English Title**: {similar_results[index + i]['title']}")
                cols[i].markdown(f"**Overview**: {similar_results[index + i]['overview']}")
                cols[i].markdown(f"**Genres**: {genres}")
                cols[i].markdown(f"**Release Data**: {similar_results[index + i]['release_date']}")
                cols[i].markdown(f"**Language**: {similar_results[index + i]['language_name']}")
                cols[i].markdown(f"**Distance**: {similar_results[index + i]['distance']}")
            except:
                continue

    st.markdown(f"search latency = {end_time - start_time:.4f}s")

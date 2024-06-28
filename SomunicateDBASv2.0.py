# Created by Max Rodriguez
# Project for the Somunicate Team at TU Berlin
# Date: 28.06.2024
# Description: This project involves developing a web-based 
# application using Streamlit that allows users to find and 
# play audio files based on their selected ratings across 
# various audio dimensions. The application leverages both 
# Euclidean and Mahalanobis distance calculations to find the 
# closest matching sound based on user input. 

import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import base64
import matplotlib.pyplot as plt

# Function Header: loads the csv file data into two lists,
# one including all of the data from the file, and one including only
# the rating data found in the last 19 columns of the data file
# (19 audio dimensions)
def load_median_data(file_path):
    data = pd.read_csv(file_path)
   
    rating_columns = data.columns[4:23]  
   
    # convert rating columns to numeric types, coercing any errors to NaN
    data[rating_columns] = data[rating_columns].apply(pd.to_numeric, errors='coerce')
   
    return data, rating_columns

def load_correlation_data(file_path):
    data = pd.read_csv(file_path, delimiter=';')
    
    # Convert to numeric types, coercing any errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    return data

# Function header: finds the closest sound using Euclidean distance
def find_closest_sound_euclidean(user_ratings, dimensions, data):
    # filter data to only include the relevant dimensions
    specified_dimensions = data[dimensions]
   
    # calculate the Euclidean distance between user ratings and each sound's ratings
    distances = np.linalg.norm(specified_dimensions.values - user_ratings, axis=1)
   
    # find the index of the minimum distance
    closest_index = np.argmin(distances)
   
    # return the sound corresponding to the closest index
    return data.iloc[closest_index]

# Function header: finds the closest sound using Mahalanobis distance
def find_closest_sound_mahalanobis(user_ratings, dimensions, data, inv_cov_matrix):
    # filter data to only include the relevant dimensions
    specified_dimensions = data[dimensions]
   
    # Ensure the inverse covariance matrix is of the correct size
    selected_inv_cov_matrix = inv_cov_matrix[np.ix_(
        [rating_columns.get_loc(dim) for dim in dimensions],
        [rating_columns.get_loc(dim) for dim in dimensions]
    )]
   
    # manually calculate the Mahalanobis distance between user ratings and each sound's ratings
    distances = []
    user_ratings = np.array(user_ratings)
    for i in range(len(specified_dimensions)):
        row = specified_dimensions.iloc[i].values
        diff = row - user_ratings
        distance = np.sqrt(diff.T @ selected_inv_cov_matrix @ diff)
        distances.append(distance)
   
    # find the index of the minimum distance
    closest_index = np.argmin(distances)
   
    # return the sound corresponding to the closest index
    return data.iloc[closest_index]

# relative path to median rating data
median_path = "240620_median_rating_data.csv"

# relative path to the audio files directory
audio_files_dir = "HiDrive-audio_data"

# relative path to correlation file
corr_path = "correlations.csv"

# load and prepare data
data, rating_columns = load_median_data(median_path)
corr_matrix = load_correlation_data(corr_path)

# Calculate the standard deviation for each of the 19 dimensions
std_devs = data[rating_columns].std()

# Create a diagonal matrix with the standard deviations
std_diag_matrix = np.diag(std_devs)

# Convert correlation matrix to NumPy array
corr_matrix = load_correlation_data(corr_path).iloc[:, 1:].to_numpy()
# Scale based on -1 to 1 rating range
corr_matrix = (corr_matrix * 2) - 1

# Ensure standard deviation matrix is a NumPy array of floats
std_diag_matrix = np.diag(data[rating_columns].std().astype(float))

# Calculate the inverse of the covariance matrix for Mahalanobis distance
inv_cov_matrix = np.linalg.inv(std_diag_matrix @ corr_matrix @ std_diag_matrix)

# Display the matrices for verification
# st.write("Standard Deviation Diagonal Matrix:")
# st.write(pd.DataFrame(std_diag_matrix, columns=rating_columns, index=rating_columns))

# st.write("Inverse Covariance Matrix:")
# st.write(pd.DataFrame(inv_cov_matrix, columns=rating_columns, index=rating_columns))

# German translations for each English dimension
dimension_translations = {
    'being_ready': 'bereit sein',
    'having_news': 'Nachrichten haben',
    'being_empty': 'leer sein',
    'shutting_down': 'Herunterfahren',
    'negative_warnings': 'negative Warnungen',
    'urgency_reminder': 'Dringlichkeitserinnerung',
    'encouraging_confirmations': 'ermutigende Bestätigungen',
    'starting_prompts': 'Startaufforderungen',
    'waiting_prompts': 'Warteaufforderungen',
    'sophistication': 'Raffinesse',
    'positivity': 'Positivität',
    'progressiveness': 'Fortschrittlichkeit',
    'dominance': 'Dominanz',
    'solidity': 'Solidität',
    'process_ongoing': 'Prozess läuft',
    'having_a_problem': 'ein Problem haben',
    'having_finished_successfully': 'erfolgreich abgeschlossen',
    'purity': 'Reinheit',
    'playfulness': 'Verspieltheit'
}

# Function header: get the dimension in German
def get_bilingual_dimension(dimension):
    return f"{dimension} ({dimension_translations[dimension]})"

st.write("Select from the dimensions shown in the following list:")
st.write("Wählen Sie aus den in der folgenden Liste angezeigten Dimensionen:")

# allow user to multiselect dimensions
selected_dimensions = st.multiselect(
    "Select dimensions (Dimensionen auswählen)",
    options=[get_bilingual_dimension(dim) for dim in rating_columns]
)

if len(selected_dimensions) == 0:
    st.warning("Please select at least 1 dimension. (Bitte wählen Sie mindestens 1 Dimension aus.)")
else:
    st.success(f"You have selected: {selected_dimensions}")
    st.success(f"Sie haben ausgewählt: {selected_dimensions}")

    # user input for the selected dimensions using sliders
    user_ratings = []
    for dimension in selected_dimensions:
        # Extract the original dimension name
        original_dimension = dimension.split(' ')[0]
        rating = st.slider(f"Rate {dimension} (Bewerten Sie {dimension_translations[original_dimension]})", min_value=-1.0, max_value=1.0, step=0.01)
        user_ratings.append(rating)

    if len(user_ratings) > 0:
        # find the closest sound using Euclidean distance
        closest_sound_euclidean = find_closest_sound_euclidean(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], data)

        st.write("Closest sound based on Euclidean distance:")
        st.write("Nächster Klang basierend auf Ihren Bewertungen (Euclidean):")
        st.write(closest_sound_euclidean)

        sound_file_name_euclidean = closest_sound_euclidean['sound'].strip().lstrip('/')
        sound_file_path_euclidean = os.path.join(audio_files_dir, sound_file_name_euclidean)

        if os.path.exists(sound_file_path_euclidean):
            audio_file_euclidean = open(sound_file_path_euclidean, 'rb').read()
            audio_base64_euclidean = base64.b64encode(audio_file_euclidean).decode()

            html_code_euclidean = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Play Audio</title>
            </head>
            <body>
                <audio id="audioPlayerEuclidean" src="data:audio/mp3;base64,{audio_base64_euclidean}" type="audio/mpeg" controls></audio>
                <button onclick="document.getElementById('audioPlayerEuclidean').play()" style="background-color: green; color: white;">Play Euclidean Audio</button>
            </body>
            </html>
            """
            components.html(html_code_euclidean, height=200)
        else:
            st.error("Sound file not found. (Sounddatei nicht gefunden.)")

        # find the closest sound using Mahalanobis distance
        closest_sound_mahalanobis = find_closest_sound_mahalanobis(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], data, inv_cov_matrix)

        st.write("Closest sound based on Mahalanobis distance:")
        st.write("Nächster Klang basierend auf Ihren Bewertungen (Mahalanobis):")
        st.write(closest_sound_mahalanobis)

        sound_file_name_mahalanobis = closest_sound_mahalanobis['sound'].strip().lstrip('/')
        sound_file_path_mahalanobis = os.path.join(audio_files_dir, sound_file_name_mahalanobis)

        if os.path.exists(sound_file_path_mahalanobis):
            audio_file_mahalanobis = open(sound_file_path_mahalanobis, 'rb').read()
            audio_base64_mahalanobis = base64.b64encode(audio_file_mahalanobis).decode()

            html_code_mahalanobis = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Play Audio</title>
            </head>
            <body>
                <audio id="audioPlayerMahalanobis" src="data:audio/mp3;base64,{audio_base64_mahalanobis}" type="audio/mpeg" controls></audio>
                <button onclick="document.getElementById('audioPlayerMahalanobis').play()" style="background-color: blue; color: white;">Play Mahalanobis Audio</button>
            </body>
            </html>
            """
            components.html(html_code_mahalanobis, height=200)
        else:
            st.error("Sound file not found. (Sounddatei nicht gefunden.)")
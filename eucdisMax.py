import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import base64

# Function Header: loads the csv data
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # rating columns from the 5th column onward up to a specified upper limit
    rating_columns = data.columns[4:23]  
    
    # convert rating columns to numeric types, coercing any errors to NaN
    data[rating_columns] = data[rating_columns].apply(pd.to_numeric, errors='coerce')
    
    return data, rating_columns

# Function header: finds the closest sound in the audio data file given the input dimensions
def find_closest_sound(user_ratings, dimensions, data):
    # filter data to only include the relevant dimensions
    filtered_data = data[dimensions]
    
    # calculate the Euclidean distance between user ratings and each sound's ratings
    distances = np.linalg.norm(filtered_data.values - user_ratings, axis=1)
    
    # find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    # return the sound corresponding to the closest index
    return data.iloc[closest_index]

# relative path to the CSV file
file_path = "240620_median_rating_data.csv"

# relative path to the audio files directory
audio_files_dir = "HiDrive-audio_data"

# load and prepare data
data, rating_columns = load_data(file_path)

# English and German translations for each dimension
dimension_translations = {
    'being_ready': 'bereit sein',
    'having_news': 'Nachrichten haben',
    'being_empty': 'leer sein',
    'shutting_down': 'Herunterfahren',
    'negative_warnigns': 'negative Warnungen',
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

# Function to get the bilingual dimension
def get_bilingual_dimension(dimension):
    return f"{dimension} ({dimension_translations[dimension]})"

# Streamlit code to allow user to select up to 19 dimensions
st.write("Select from the dimensions shown in the following list:")
st.write("Wählen Sie aus den in der folgenden Liste angezeigten Dimensionen:")

# Allow user to select dimensions
selected_dimensions = st.multiselect(
    "Select dimensions (Dimensionen auswählen)",
    options=[get_bilingual_dimension(dim) for dim in rating_columns]
)

if len(selected_dimensions) == 0:
    st.warning("Please select at least 1 dimension. (Bitte wählen Sie mindestens 1 Dimension aus.)")
else:
    st.success(f"You have selected: {selected_dimensions}")
    st.success(f"Sie haben ausgewählt: {selected_dimensions}")

    # User input for the selected dimensions using sliders
    user_ratings = []
    for dimension in selected_dimensions:
        # Extract the original dimension name
        original_dimension = dimension.split(' ')[0]
        rating = st.slider(f"Rate {dimension} (Bewerten Sie {dimension_translations[original_dimension]})", min_value=-1.0, max_value=1.0, step=0.01)
        user_ratings.append(rating)

    if len(user_ratings) > 0:
        # Find the closest sound and display it
        closest_sound = find_closest_sound(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], data)

        st.write("Closest sound based on your input ratings:")
        st.write("Nächster Klang basierend auf Ihren Bewertungen:")
        st.write(closest_sound)

        # Use the correct column name for sound files
        sound_file_name = closest_sound['sound']

        # Ensure there are no leading slashes or whitespace issues
        sound_file_name = sound_file_name.strip().lstrip('/')

        # Construct the full path of the sound file
        sound_file_path = os.path.join(audio_files_dir, sound_file_name)

        if os.path.exists(sound_file_path):
            # Read the audio file as binary
            audio_file = open(sound_file_path, 'rb').read()

            # Encode the audio file to base64
            audio_base64 = base64.b64encode(audio_file).decode()

            # Define the HTML for the audio player without the download button
            html_code = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Play Audio</title>
            </head>
            <body>
                <audio id="audioPlayer" src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg" controls></audio>
                <button onclick="document.getElementById('audioPlayer').play()">Play Audio</button>
            </body>
            </html>
            """

            # Display the audio player using Streamlit components
            components.html(html_code, height=200)
        else:
            st.error("Sound file not found. (Sounddatei nicht gefunden.)")

# Created by Max Rodriguez
# Supervisor: Dr. Steffan Lepa
# Project for the Somunicate Team at TU Berlin
# Date: 11.07.2024
# Description: This project involves developing a web-based
# application using Streamlit that allows users to find and
# play audio files based on their optional demographic input and selected ratings across
# various audio dimensions. The application leverages both
# Euclidean and Mahalanobis distance calculations to find the top 5
# closest matching sounds based on user input.

import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import base64

NUM_DEMOGRAPHIC_GROUPS = 12

class SomunicateApp:
    def __init__(self):
        self.data = None
        self.rating_columns = None
        self.inv_cov_matrix = None
        self.audio_files_dir = "HiDrive-audio_data"  # INSERT YOUR RELATIVE PATH TO THE FOLDER
        self.user_group_ids = []
        self.final_combined_data = None
        self.dimension_translations = {
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

    def set_style(self):
        # Set the background color to black and other styles
        st.markdown(
            """
            <style>
            .reportview-container {
                background: black;
                color: white;
            }
            .markdown-text-container {
                color: white;
            }
            h1 {
                color: #A9A9A9;
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
                font-size: 2.5em;
            }
            h2 {
                color: #2196F3;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
                font-size: 2em;
            }
            p {
                color: white;
                text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.1);
                font-size: 1.2em;
            }
            .custom-bullet {
                color: white;
                font-size: 1.2em;
                list-style-type: none;
                padding: 10px 0;
            }
            .custom-bullet li::before {
                content: "• ";
                color: #FF6347;
            }
            .custom-bullet span {
                color: yellow;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
            }
            .custom-container {
                margin: 0 auto;
                max-width: 800px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    def show_header(self):
        # Centered title with 3D effect
        st.markdown("<div class='custom-container'><center><h1>Welcome to Somunicate Audio Search</h1></center></div>", unsafe_allow_html=True)
        
        # Description section with 3D effect
        st.markdown("<div class='custom-container'><center><p>An audio search algorithm designed to help you find a sound tailored to your preferences.</p></center></div>", unsafe_allow_html=True)

        with st.expander("How it Works"):
            # How It Works section with 3D effect
            st.markdown("""
                <div class='custom-container'>
                <ol class='custom-bullet'>
                    <li><strong>Select Your Demographic Group:</strong> Choose demographic parameters such as gender, migration background, and age group. This selection helps refine the search to find sounds that are more likely to be preferred by people with similar demographics.</li>
                    <li><strong>Select Your Dimensions:</strong> Choose from 19 distinct sound dimensions, each capturing a unique aspect of audio quality.</li>
                    <li><strong>Rate Each Dimension:</strong> Provide your ratings for each chosen dimension on a scale from -1 to 1:</li>
                    <ul>
                        <li><strong>-1:</strong> Completely opposite of the selected dimension</li>
                        <li><strong>0:</strong> Neutral or indifferent</li>
                        <li><strong>1:</strong> The epitome of the selected dimension</li>
                    </ul>
                    <li><strong>Receive Your Match:</strong> This algorithm identifies the top 1-10 closest sounds to your ratings using two different mathematical techniques:</li>
                    <ul>
                        <li><strong>Euclidean Distance:</strong> A simple yet effective method that calculates the closest matches based on direct distance measurements.</li>
                        <li><strong>Mahalanobis Distance:</strong> A more sophisticated method that considers correlations between dimensions for more accurate matches.</li>
                        <ul>
                            <li><span>If you have selected specific demographic groups, the algorithm will further refine the results to find sounds with a high liking score (greater than 50) from the selected demographic groups. This ensures the recommended sounds are not only close to your ratings but also preferred by similar demographic groups.</span></li>
                        </ul>
                    </ul>
                </ol>
                </div>
            """, unsafe_allow_html=True)

    # Function to load median rating data
    @staticmethod
    def load_median_data(file_path):
        data = pd.read_csv(file_path, dtype={'sound': str})
        rating_columns = data.columns[4:23]

        # Convert rating columns to numeric types, coercing any errors to NaN
        data[rating_columns] = data[rating_columns].apply(pd.to_numeric, errors='coerce')
        return data, rating_columns

    # Function to load liking data
    @staticmethod
    def load_liking_data(file_path):
        liking_data = pd.read_csv(file_path, dtype={'sound': str, 'group_id': int, 'liking': float})
        return liking_data

    # Function to load group ID data
    @staticmethod
    def load_groupID(file_path):
        groupID_data = pd.read_csv(file_path)
        return groupID_data

    @staticmethod
    def load_fin_combined_data(file_path):
        final_combined_data = pd.read_csv(file_path)
        return final_combined_data

    # Function to load correlation data
    @staticmethod
    def load_correlation_data(file_path):
        data = pd.read_csv(file_path, delimiter=';')

        # Convert to numeric types, coercing any errors to NaN
        data = data.apply(pd.to_numeric, errors='coerce')
        return data

    # Function to find the closest sound using Euclidean distance
    @staticmethod
    def find_closest_sounds_euclidean(user_ratings, dimensions, data, user_group_ids, top_n):
        # Filter data to only include the relevant dimensions
        specified_dimensions = data[dimensions]

        # Calculate the Euclidean distance between user ratings and each sound's ratings
        distances = np.linalg.norm(specified_dimensions.values - user_ratings, axis=1)

        # Find the indices of the top N smallest distances
        top_indices = np.argsort(distances)

        # Meaning that the user has checked a group demographic feature box other than "Non-Binary"
        if len(user_group_ids) < NUM_DEMOGRAPHIC_GROUPS:
            filtered_indices = []
            for index in top_indices:
                if any(data.iloc[index][str(group_id)] > 50 for group_id in user_group_ids):
                    filtered_indices.append(index)
                if len(filtered_indices) >= top_n:
                    break
            top_indices = filtered_indices
        else:
            top_indices = top_indices[:top_n]

        # Return the sounds corresponding to the top N indices
        return data.iloc[top_indices]

    # Function to find the closest sound using Mahalanobis distance
    @staticmethod
    def find_closest_sounds_mahalanobis(user_ratings, dimensions, data, inv_cov_matrix, user_group_ids, top_n):
        # Filter data to only include the relevant dimensions
        specified_dimensions = data[dimensions]

        # Ensure the inverse covariance matrix is of the correct size
        selected_inv_cov_matrix = inv_cov_matrix[np.ix_(
            [dimensions.index(dim) for dim in dimensions],
            [dimensions.index(dim) for dim in dimensions]
        )]

        # Manually calculate the Mahalanobis distance between user ratings and each sound's ratings
        distances = []
        user_ratings = np.array(user_ratings)

        for i in range(len(specified_dimensions)):
            row = specified_dimensions.iloc[i].values
            diff = row - user_ratings
            distance = np.sqrt(diff.T @ selected_inv_cov_matrix @ diff)
            distances.append(distance)

        # Find the indices of the top N smallest distances
        top_indices = np.argsort(distances)

        # Meaning that the user has checked a group demographic feature box other than "Non-Binary"
        if len(user_group_ids) < NUM_DEMOGRAPHIC_GROUPS:
            filtered_indices = []
            for index in top_indices:
                if any(data.iloc[index][str(group_id)] > 50 for group_id in user_group_ids):
                    filtered_indices.append(index)
                if len(filtered_indices) >= top_n:
                    break
            top_indices = filtered_indices
        else:
            top_indices = top_indices[:top_n]

        # Return the sounds corresponding to the top N indices
        return data.iloc[top_indices]

    def load_and_prepare_data(self):
        """
        Loads and prepares all necessary data for the application.
        """
        median_path = "240620_median_rating_data.csv"
        corr_path = "correlations.csv"
        final_combined_data_path = "final_combined_data.csv"
        
        self.data, self.rating_columns = self.load_median_data(median_path)
        std_diag_matrix = self.std_matrix_setup(self.rating_columns)
        corr_matrix = self.corr_matrix_setup(corr_path)
        self.inv_cov_matrix = np.linalg.inv(std_diag_matrix @ corr_matrix @ std_diag_matrix)
        self.final_combined_data = self.load_fin_combined_data(final_combined_data_path)

    def std_matrix_setup(self, rate_cols):
        # Ensure standard deviation matrix is a NumPy array of floats
        return np.diag(self.data[rate_cols].std().astype(float))

    def corr_matrix_setup(self, corr_path):
        # Convert correlation matrix to NumPy array
        corr_matrix = self.load_correlation_data(corr_path).iloc[:, 1:].to_numpy()

        # Scale based on -1 to 1 rating range
        return (corr_matrix * 2) - 1

    def get_user_group_ids(self):
        """
        Gets user group id number based on demographic specifics.
        """
        st.markdown("<center><h3 style='color: tan;'>OPTIONAL: Select from the demographic options shown in the following list:</h3></center>", unsafe_allow_html=True)
        params = {
            "Gender": {"Female": False, "Male": False},
            "Migration Background": {"Migration Background": False, "No Migration Background": False},
            "Age": {"Age Group 1": False, "Age Group 2": False, "Age Group 3": False}
        }
        groupID_params = list(params.keys())

        with st.expander("Demographic Options"):
            # display demographic parameter selection based on sex, migration background, and age
            selected_params = {}
            for param in groupID_params:
                st.markdown(f"<center><h3 style='color: tan;'>{param}</h3></center>", unsafe_allow_html=True)
                selected_options = []
                for option in params[param]:
                    params[param][option] = st.checkbox(option, value=params[param][option])
                    if params[param][option]:
                        selected_options.append(option)
                    if option == "Male":
                        st.checkbox("Non-Binary")
                selected_params[param] = selected_options

        # Initialize the filtered DataFrame to include all data initially
        filtered_data = self.groupID_data

        # Apply the filters iteratively for each parameter
        for param in groupID_params:
            selected_options = selected_params[param]
            if selected_options:
                filtered_data = filtered_data[filtered_data[param].isin(selected_options)]

        # Collect the IDs from the filtered DataFrame
        self.user_group_ids = filtered_data['Group ID'].tolist()

        # ########### DEBUG #############
        # with st.expander("DEBUG: User Group IDs: "):
        #     # Display the selected group IDs
        #     st.write("User Group IDs:", self.user_group_ids)

    def get_target_dimensions_ratings(self):
        """
        Collects user ratings for selected dimensions.
        """
        # Display checkboxes for each dimension
        st.markdown("<center><h3 style='color: yellow;'>Select from the dimensions shown in the following list:</h3></center>", unsafe_allow_html=True)
        selected_dimensions = []

        with st.expander("See Dimension Options"):
            for dimension in self.dimension_translations:
                bilingual_dimension = self.get_bilingual_dimension(dimension)
                if st.checkbox(bilingual_dimension):
                    selected_dimensions.append(dimension)

        # Collect user ratings for selected dimensions
        user_ratings = []
        for dimension in selected_dimensions:
            original_dimension = dimension.split(' ')[0]
            rating = st.slider(f"Rate {dimension} (Bewerten Sie {self.dimension_translations[original_dimension]})", min_value=-1.0, max_value=1.0, step=0.01)
            user_ratings.append(rating)

        # Ensure user ratings and selected dimensions are not empty
        if not selected_dimensions:
            st.error("Please select at least one dimension.")
        elif not user_ratings:
            st.error("Please rate the selected dimensions.")

        return user_ratings, selected_dimensions

    def display_results(self, user_ratings, selected_dimensions, top_n):
        """
        Displays the closest sounds based on user ratings.
        """
        # Columns to display closest sounds based on user ratings
        col1, spacer, col2 = st.columns([2, 1, 2])  # Adjust the ratio as needed

        audio_closeness_display = {
            1 : "(first closest)",
            2 : "(second closest)",
            3 : "(third closest)",
            4 : "(fourth closest)",
            5 : "(fifth closest)",
            6 : "(sixth closest)",
            7 : "(seventh closest)",
            8 : "(eighth closest)",
            9 : "(ninth closest)",
            10 : "(tenth closest)"
        }

        if len(user_ratings) > 0:
            # Find the 5 closest sounds using Euclidean distance
            closest_sounds_euclidean = self.find_closest_sounds_euclidean(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], self.final_combined_data, self.user_group_ids, top_n)
            
            if not closest_sounds_euclidean.empty:
                with col1:
                    with st.expander("Expand to see advanced (Euclidean) audio information [Erweitern, um erweiterte (Euclidean) Audioinformationen zu sehen]:"):
                        st.write("Top 5 closest sounds based on Euclidean distance:")
                        st.write("Die 5 nächstgelegenen Klänge basierend auf Ihren Bewertungen (Euclidean):")
                        st.write(closest_sounds_euclidean)
            
            # Find the 5 closest sounds using Mahalanobis distance
            closest_sounds_mahalanobis = self.find_closest_sounds_mahalanobis(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], self.final_combined_data, self.inv_cov_matrix, self.user_group_ids, top_n)
            
            if not closest_sounds_mahalanobis.empty:
                with col2:
                    with st.expander("Expand to see advanced (Mahalanobis) audio information [Erweitern, um erweiterte (Mahalanobis) Audioinformationen zu sehen]:"):
                        st.write("Top 5 closest sounds based on Mahalanobis distance:")
                        st.write("Die 5 nächstgelegenen Klänge basierend auf Ihren Bewertungen (Mahalanobis):")
                        st.write(closest_sounds_mahalanobis)

            euc_audio_num = 0
            # Display audio player for each closest sound (Euclidean)
            for i, row in closest_sounds_euclidean.iterrows():
                euc_audio_num += 1
                sound_file_name = row['sound'].strip().lstrip('/')
                sound_file_path = os.path.join(self.audio_files_dir, sound_file_name)
                if os.path.exists(sound_file_path):
                    audio_file = open(sound_file_path, 'rb').read()
                    audio_base64_euclidean = base64.b64encode(audio_file).decode()
                    html_code = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>Play Audio</title>
                            <style>
                                .hover-button {{
                                    background-color: #5D3FD3;
                                    color: white;
                                    font-size: 16px;
                                    padding: 10px 20px;
                                    width: 200px;
                                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                                    border-radius: 12px;
                                    transition: background-color 0.3s ease;
                                }}
                                .hover-button:hover {{
                                    background-color: #9370DB;
                                }}
                            </style>
                        </head>
                        <body>
                            <audio id="audioPlayer{i}" src="data:audio/mp3;base64,{audio_base64_euclidean}" type="audio/mpeg" controls></audio>   
                            <button class="hover-button" onclick="document.getElementById('audioPlayer{i}').play()">Play Euclidean Audio {audio_closeness_display[euc_audio_num]}</button>
                        </body>
                    </html>
                    """
                    with col1:
                        components.html(html_code, height=150, width=400)
                else:
                    st.error(f"Sound file {sound_file_name} not found. (Sounddatei nicht gefunden)")

            mah_audio_num = 0
            # Display audio player for each closest sound (Mahalanobis)
            for i, row in closest_sounds_mahalanobis.iterrows():
                mah_audio_num += 1

                sound_file_name = row['sound'].strip().lstrip('/')
                sound_file_path = os.path.join(self.audio_files_dir, sound_file_name)
                if os.path.exists(sound_file_path):
                    audio_file = open(sound_file_path, 'rb').read()
                    audio_base64_mahalanobis = base64.b64encode(audio_file).decode()
                    html_code_mahalanobis = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>Play Audio</title>
                            <style>
                                .hover-button {{
                                    background-color: CornflowerBlue;
                                    color: white;
                                    font-size: 16px;
                                    padding: 10px 20px;
                                    width: 200px;
                                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                                    border-radius: 12px;
                                    transition: background-color 0.3s ease;
                                }}
                                .hover-button:hover {{
                                    background-color: LightSkyBlue;
                                }}
                            </style>
                        </head>
                        <body>
                            <audio id="audioPlayerMahalanobis" src="data:audio/mp3;base64,{audio_base64_mahalanobis}" type="audio/mpeg" controls></audio>   
                            <button class="hover-button" onclick="document.getElementById('audioPlayerMahalanobis').play()">Play Mahalanobis Audio {audio_closeness_display[mah_audio_num]}</button>
                        </body>
                    </html>
                    """

                    with col2:
                        components.html(html_code_mahalanobis, height=150, width=400)
                else:
                    st.error("Sound file not found. (Sounddatei nicht gefunden)")

    def get_bilingual_dimension(self, dimension):
        """
        Function to get the dimension in German.
        """
        return f"{dimension} ({self.dimension_translations[dimension]})"

    def run(self):
        """
        Runs the SomunicateApp application.
        """
        self.set_style()
        self.show_header()
        self.load_and_prepare_data()
        self.groupID_data = self.load_groupID("groupid.csv")
        self.get_user_group_ids()
        user_ratings, selected_dimensions = self.get_target_dimensions_ratings()
        
        # Check if dimensions are selected and rated
        if selected_dimensions and user_ratings:
            st.warning("Please select the number of sounds to be displayed.")
            top_n = st.slider(f"Display Sound(s)", min_value=1, max_value=10, step=1)
            self.display_results(user_ratings, selected_dimensions, top_n)
        
# Run the application
app = SomunicateApp()
app.run()

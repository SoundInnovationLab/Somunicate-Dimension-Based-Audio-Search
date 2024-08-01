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

import pandas as pd

class AudioScoreCalculator:
    def __init__(self, user_group_ids):
        self.user_group_ids = user_group_ids

    def calculate_scores(self, row):
        total_liking = 0
        total_familiarity = 0
        count_liking = 0
        count_familiarity = 0

        for id in self.user_group_ids:
            # Get the liking and familiarity scores, replacing NaN with 0
            current_liking = row.get(str(id), 0)
            current_familiarity = row.get(str(id + 12), 0)
            
            # Handle cases where the value might be NaN
            current_liking = 0 if pd.isna(current_liking) else int(current_liking)
            current_familiarity = 0 if pd.isna(current_familiarity) else int(current_familiarity)

            total_liking += current_liking
            count_liking += 1

            total_familiarity += current_familiarity
            count_familiarity += 1

        # Calculate averages and ensure they are integers
        average_liking = total_liking // count_liking if count_liking > 0 else 0
        average_familiarity = total_familiarity // count_familiarity if count_familiarity > 0 else 0

        return average_liking, average_familiarity

class SomunicateApp:
    def __init__(self):
        self.data = None
        self.rating_columns = None
        self.inv_cov_matrix = None
        self.audio_files_dir = "HiDrive-audio_data"  # INSERT YOUR RELATIVE PATH TO THE FOLDER
        self.user_group_ids = []
        self.final_combined_data = None
        self.dimension_display = {
            'being_ready': 'Being Ready',
            'having_news': 'Having News',
            'being_empty': 'Being Empty',
            'shutting_down': 'Shutting Down',
            'negative_warnings': 'Negative Warnings',
            'urgency_reminder': 'Urgency Reminder',
            'encouraging_confirmations': 'Encouraging Confirmations',
            'starting_prompts': 'Starting Prompts',
            'waiting_prompts': 'Waiting Prompts',
            'sophistication': 'Sophistication',
            'positivity': 'Positivity',
            'progressiveness': 'Progressiveness',
            'dominance': 'Dominance',
            'solidity': 'Solidity',
            'process_ongoing': 'Process Ongoing',
            'having_a_problem': 'Having A Problem',
            'having_finished_successfully': 'Having Finished Successfully',
            'purity': 'Purity',
            'playfulness': 'Playfulness'
        }

        # Initialize session state for sliders if not already set
        if 'MIN_LIKING' not in st.session_state:
            st.session_state.MIN_LIKING = 0
        if 'MIN_FAMILIARITY' not in st.session_state:
            st.session_state.MIN_FAMILIARITY = 0

    def set_style(self):
        # Set the background color to black and other styles
        st.markdown(
            """
            <style>
            .reportview-container {
                background: black;
                color: #8A2BE2;
            }
            .markdown-text-container {
                color: #8A2BE2;
            }
            h1 {
                color: #8A2BE2; /* Violet color */
                font-size: 2.5em;
                text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3); /* 3D shadow effect */
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            h3 {
                color: #8A2BE2;
                font-size: 1.5em;
                text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3); /* 3D shadow effect */
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            h4 {
                color: #8A2BE2;
                font-size: 1.0em;
                text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3); /* 3D shadow effect */
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            p {
                color: #9370DB; /* Medium Purple */
                font-style: oblique; /* Italicize text */
                font-weight: bold; /* Bold text */
                text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3); /* 3D shadow effect */
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            .custom-bullet {
                color: #9370DB;
                font-size: 1.2em;
                list-style-type: none;
                padding: 10px 0;
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            .custom-bullet li::before {
                content: "• ";
                color: #8A2BE2;
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            .custom-bullet2 li::before {
                content: "   •";
                color: #8A2BE2;
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            .custom-bullet span {
                color: #CBC3E3;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            .custom-container {
                margin: 0 auto;
                max-width: 800px;
                text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    import streamlit as st

    def show_header(self):
        # Centered title with 3D effect
        st.markdown("""
            <div class='custom-container' style='color: #8A2BE2; text-align: center;'>
                <h1>Welcome to Somunicate Audio Search</h1>
                </div>
            """, unsafe_allow_html=True
        )

        st.markdown("""
            <div class='custom-container' style='color: #8A2BE2; text-align: center;'>
                <h4><i>An audio search algorithm designed to help you find a sound tailored to your preferences.</i></h4>
            </div>
        """, unsafe_allow_html=True)

        # First expander for "How it Works"
        with st.expander("How it Works"):
            st.markdown("""
                <style>
                    /* Custom styles for the list */
                    .custom-container {
                        font-family: Arial, sans-serif;
                    }
                    .custom-bullet {
                        list-style-type: none; /* Remove default bullet points */
                        padding-left: 0;
                    }
                    .custom-bullet > li {
                        margin-bottom: 10px; /* Space between items */
                        padding-left: 20px; /* Indentation for top-level list items */
                    }
                    .custom-bullet > li::before {
                        content: '•'; /* Custom bullet point */
                        color: #8A2BE2; /* Bullet color */
                        font-weight: bold; /* Bullet point style */
                        display: inline-block; 
                        width: 1em; /* Adjust space between bullet and text */
                        margin-left: -1em; /* Align bullet point properly */
                    }
                    .custom-bullet ul {
                        list-style-type: none; /* Remove default bullets for nested lists */
                        padding-left: 0; /* Remove padding */
                        margin-left: 20px; /* Indentation for nested lists */
                    }
                    .custom-bullet ul li {
                        margin-bottom: 5px; /* Space between nested items */
                        padding-left: 20px; /* Indentation for nested items */
                    }
                    .custom-bullet ul li::before {
                        content: '◦'; /* Custom bullet point for nested items */
                        color: #8A2BE2; /* Bullet color */
                        font-weight: bold; /* Bullet point style */
                        display: inline-block; 
                        width: 1em; /* Adjust space between bullet and text */
                        margin-left: -1em; /* Align bullet point properly */
                    }
                    .custom-bullet ul ul {
                        padding-left: 0; /* Remove additional padding */
                        margin-left: 20px; /* Further indentation for sub-nested lists */
                    }
                    .custom-bullet ul ul li::before {
                        content: '■'; /* Custom bullet point for sub-nested lists */
                        color: #8A2BE2; /* Bullet color */
                        font-weight: bold; /* Bullet point style */
                        display: inline-block; 
                        width: 1em; /* Adjust space between bullet and text */
                        margin-left: -1em; /* Align bullet point properly */
                    }
                    /* Additional styles for non-strong text */
                    .custom-container p, .custom-container li {
                        color: #D8BFD8; /* Light purple color for non-strong text */
                        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3); /* 3D shadow effect */
                        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.2);
                    }
                </style>
                <div class='custom-container'>
                    <ol class='custom-bullet'>
                        <li><strong>Select Your Demographic Group:</strong> Choose parameters such as gender, migration background, and age group. When liking and familiarity weightings are factored in, this selection helps refine the search to find sounds that are more likely to be preferred by those with similar demographics.</li>
                        <li><strong>Select Your Dimensions:</strong> Choose from 19 distinct sound dimensions, each capturing a unique aspect of audio quality.</li>
                        <li><strong>Rate Each Dimension:</strong> Provide your ratings for each chosen dimension on a scale from -1 to 1:
                            <ul>
                                <li><strong>-1:</strong> Completely opposite of the selected dimension</li>
                                <li><strong>0:</strong> Neutral or indifferent</li>
                                <li><strong>1:</strong> The epitome of the selected dimension</li>
                            </ul>
                        </li>
                        <li><strong>Weighting Liking & Familiarity:</strong> This advanced feature allows for precise control over how sounds are displayed.
                            <!-- Nested expander -->
                            <details>
                                <summary>Read More</summary>
                                <ul>
                                    <li><strong>Liking:</strong> This factor represents the average preference of a demographic group for a specific sound. Increasing the weight for 'Liking' will prioritize sounds that are highly favored by the demographic group, making these sounds more prominently featured. Conversely, reducing the weight will decrease the emphasis on the group's preference, allowing for a broader range of sounds to be displayed.</li>
                                    <li><strong>Familiarity:</strong> This factor reflects the average level of exposure of a demographic group to a particular sound. Elevating the weight for 'Familiarity' will highlight sounds that the demographic group is more frequently exposed to, ensuring that these familiar sounds are more prominently featured. Lowering the weight will reduce the focus on exposure, allowing less familiar sounds to be considered in the results.</li>
                                </ul>
                            </details>
                        </li>
                        <li><strong>Receive Your Match:</strong> This algorithm identifies the top 1-10 closest sounds to your ratings using two different mathematical techniques:
                            <ul>
                                <li><strong>Euclidean Distance:</strong> A simple yet effective method that calculates the closest matches based on direct distance measurements.</li>
                                <li><strong>Mahalanobis Distance:</strong> A more sophisticated method that considers correlations between dimensions for more accurate matches.</li>
                            </ul>
                        </li>
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
    def find_closest_sounds_euclidean(user_ratings, dimensions, final_combined_data, user_group_ids, top_n):
        # Filter data to only include the relevant dimensions
        specified_dimensions = final_combined_data[dimensions]

        # Calculate the Euclidean distance between user ratings and each sound's ratings
        distances = np.linalg.norm(specified_dimensions.values - user_ratings, axis=1)

        # Find the indices of the top N smallest distances
        top_indices = np.argsort(distances)


        filtered_indices = []
        for index in top_indices:
            # First condition: Average liking data check
            average_liking = np.mean([final_combined_data.iloc[index][str(group_id)] for group_id in user_group_ids])
            condition1 = average_liking >= st.session_state.MIN_LIKING

            # Second condition: Average familiarity data check
            average_familiarity = np.mean([final_combined_data.iloc[index][str(group_id + 12)] for group_id in user_group_ids])
            condition2 = average_familiarity >= st.session_state.MIN_FAMILIARITY

            # Both conditions combined
            if condition1 and condition2:
                filtered_indices.append(index) 
            if len(filtered_indices) >= top_n or index == '':
                break

        top_indices = filtered_indices

        # Return the sounds corresponding to the top N indices
        return final_combined_data.iloc[top_indices]

    # Function to find the closest sound using Mahalanobis distance
    @staticmethod
    def find_closest_sounds_mahalanobis(user_ratings, dimensions, final_combined_data, inv_cov_matrix, user_group_ids, top_n):
        # Filter data to only include the relevant dimensions
        specified_dimensions = final_combined_data[dimensions]

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

        
        filtered_indices = []
        for index in top_indices:
            # First condition: Average liking data check
            average_liking = np.mean([final_combined_data.iloc[index][str(group_id)] for group_id in user_group_ids])
            condition1 = average_liking >= st.session_state.MIN_LIKING

            # Second condition: Average familiarity data check
            average_familiarity = np.mean([final_combined_data.iloc[index][str(group_id + 12)] for group_id in user_group_ids])
            condition2 = average_familiarity >= st.session_state.MIN_FAMILIARITY

            # Both conditions combined
            if condition1 and condition2:
                filtered_indices.append(index) 
            if len(filtered_indices) >= top_n or index == '':
                break

        top_indices = filtered_indices

        # Return the sounds corresponding to the top N indices
        return final_combined_data.iloc[top_indices]

    def load_and_prepare_data(self):
        """
        Loads and prepares all necessary data for the application.
        """
        median_path = "240620_median_rating_data.csv"
        corr_path = "correlations.csv"
        final_combined_data_path = "all_combined_data.csv"
        
        self.data, self.rating_columns = self.load_median_data(median_path)
        std_diag_matrix = self.std_matrix_setup(self.rating_columns)
        corr_matrix = self.corr_matrix_setup(corr_path)
        self.inv_cov_matrix = np.linalg.inv(std_diag_matrix @ corr_matrix @ std_diag_matrix)
        self.final_combined_data = self.load_fin_combined_data(final_combined_data_path)

        return self.final_combined_data

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
        st.markdown("<div class='custom-container'><center><h3><b><i>(OPTIONAL)</i> Select Demographic Options:<b></h3></center></div>", unsafe_allow_html=True)
        params = {
            "Gender": {"Female": False, "Male": False},
            "Migration Background": {"Migration Background": False, "No Migration Background": False},
            "Age": {"Age Group 1": False, "Age Group 2": False, "Age Group 3": False}
        }
        groupID_params = list(params.keys())

        with st.expander("Demographic Options"):
            # display demographic parameter selection based on sex, migration background, and age
            selected_params = {}
            age_range_key = {
                "Age Group 1": "Ages 18-35",
                "Age Group 2": "Ages 36-54",
                "Age Group 3": "Ages 55+"
            }
            for param in groupID_params:
                st.markdown(f"<center><h3 style='color: #8A2BE2;'>{param}</h3></center>", unsafe_allow_html=True)
                selected_options = []
                for option in params[param]:
                    if param == "Age":
                        params[param][option] = st.checkbox(age_range_key[option], value=params[param][option])
                        if params[param][option]:
                            selected_options.append(option)
                    else:
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

        return self.user_group_ids

    def get_target_dimensions_ratings(self):
        """
        Collects user ratings for selected dimensions.
        """
        # Display checkboxes for each dimension
        st.markdown("<div class='custom-container'><center><h3><b>Select Dimension Options:<b></h3></center></div>", unsafe_allow_html=True)
        selected_dimensions = []

        communication_levels = {
            "having_finished_successfully": "Status",
            "having_a_problem": "Status",
            "process_ongoing": "Status",
            "being_ready": "Status",
            "having_news": "Status",
            "being_empty": "Status",
            "shutting_down": "Status",
            "negative_warnings": "Appeal",
            "urgency_reminder": "Appeal",
            "encouraging_confirmations": "Appeal",
            "starting_prompts": "Appeal",
            "waiting_prompts": "Appeal",
            "sophistication": "Brand Identity",
            "positivity": "Brand Identity",
            "progressiveness": "Brand Identity",
            "dominance": "Brand Identity",
            "solidity": "Brand Identity",
            "purity": "Brand Identity",
            "playfulness": "Brand Identity"
        }

        status_dimensions = []
        appeal_dimensions = []
        brand_identity_dimensions = []

        for dimension, level in communication_levels.items():
            display_dimension = self.dimension_display[dimension]
            if level == "Status":
                status_dimensions.append((dimension, display_dimension))
            elif level == "Appeal":
                appeal_dimensions.append((dimension, display_dimension))
            elif level == "Brand Identity":
                brand_identity_dimensions.append((dimension, display_dimension))

        col1, col2, col3 = st.columns([4.0, 4.0, 4.0])

        with col1:
            with st.expander("Status"):
                for dimension, display_dimension in status_dimensions:
                    if st.checkbox(display_dimension):
                        selected_dimensions.append(dimension)

        with col2:
            with st.expander("Appeal"):
                for dimension, display_dimension in appeal_dimensions:
                    if st.checkbox(display_dimension):
                        selected_dimensions.append(dimension)

        with col3:
            with st.expander("Brand Identity"):
                for dimension, display_dimension in brand_identity_dimensions:
                    if st.checkbox(display_dimension):
                        selected_dimensions.append(dimension)

        # Collect user ratings for selected dimensions
        user_ratings = []
        for dimension in selected_dimensions:
            original_dimension = dimension.split(' ')[0]
            rating = st.slider(f"Please Weight {self.dimension_display[dimension]}", min_value=-1.0, max_value=1.0, value = 0.0, step=0.01)
            user_ratings.append(rating)

        # Ensure user ratings and selected dimensions are not empty
        if not selected_dimensions:
            st.info("Please Select At Least One Dimension")
        elif not user_ratings:
            st.info("Please Weight the Selected Dimensions")

        return user_ratings, selected_dimensions

    def display_results(self, user_ratings, selected_dimensions, top_n, data, group_id):
        """
        Displays the closest sounds based on user ratings.
        """
        # Columns to display closest sounds based on user ratings
        col1, spacer, col2 = st.columns([2, 1, 2])  # Adjust the ratio as needed

        audio_closeness_display = {
            1 : "(1st Closest)",
            2 : "(2nd Closest)",
            3 : "(3rd Closest)",
            4 : "(4th Closest)",
            5 : "(5th Closest)",
            6 : "(6th Closest)",
            7 : "(7th Closest)",
            8 : "(8th Closest)",
            9 : "(9th Closest)",
            10 : "(10th Closest)"
        }

        # Initialize Euclidean_Display_Check
        Euclidean_Display_Check = False

        if len(user_ratings) > 0:
            # Find the 1-10 closest sounds using Euclidean distance
            closest_sounds_euclidean = self.find_closest_sounds_euclidean(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], self.final_combined_data, self.user_group_ids, top_n)
            
            #### EUCLIDEAN ####


            with col2:
                Euclidean_Display_Check = st.checkbox("Display Euclidean Sounds?")
                if Euclidean_Display_Check:
                    with st.expander("See Advanced Euclidean"):
                        st.write("Top Audio Matches Based on Euclidean Distance:")
                        st.write(closest_sounds_euclidean)
                    if len(closest_sounds_euclidean) < top_n: 
                        st.warning(f"Cannot Display {top_n - len(closest_sounds_euclidean)} Additional Sound(s) With Current Weights")
            
            # Find the 1-10 closest sounds using Mahalanobis distance
            closest_sounds_mahalanobis = self.find_closest_sounds_mahalanobis(user_ratings, [dim.split(' ')[0] for dim in selected_dimensions], self.final_combined_data, self.inv_cov_matrix, self.user_group_ids, top_n)
            
            #### MAHALANOBIS ####

            with col1:
                with st.expander("See Advanced Mahalanobis"):
                    st.write("Top Audio Matches Based on Mahalanobis Distance:")
                    st.write(closest_sounds_mahalanobis)
                if len(closest_sounds_mahalanobis) < top_n: 
                    st.warning(f"Cannot Display {top_n - len(closest_sounds_mahalanobis)} Additional Sound(s) With Current Weights")

            #### EUCLIDEAN ####

            if Euclidean_Display_Check:
                euc_audio_num = 0
                audio_calc = AudioScoreCalculator(self.user_group_ids)  # Instantiate the calculator
                # Display audio player for each closest sound (Euclidean)
                for i, row in closest_sounds_euclidean.iterrows():
                    av_audio_liking, av_audio_familiarity = audio_calc.calculate_scores(row)

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
                                    audio::-webkit-media-controls-panel {{
                                        background-color: #8A2BE2;
                                    }}
                                    audio::-webkit-media-controls-play-button,
                                    audio::-webkit-media-controls-volume-slider,
                                    audio::-webkit-media-controls-timeline,
                                    audio::-webkit-media-controls-current-time-display,
                                    audio::-webkit-media-controls-time-remaining-display {{
                                        color: #FFF;
                                    }}
                                </style>
                            </head>
                            <body>
                                <div class="audio-container">
                                    <audio id="audioPlayer{i}" src="data:audio/mp3;base64,{audio_base64_euclidean}" type="audio/mpeg" controls></audio>   
                                </div>
                            </body>
                        </html>
                        """
                        with col2:
                            components.html(html_code, height=70, width=320)

                            # Format the information display using Markdown for better styling
                            st.markdown(f"""
                            <div style="
                                color: #A083F1; 
                                font-size: 0.8em;
                                margin-left: 20px; 
                                font-family: sans-serif; 
                                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5), 
                                            2px 2px 4px rgba(255, 255, 255, 0.2); 
                                margin-top: -15px;
                                margin-bottom: 5px;">
                                <i><b><strong>Euclidean Based:</strong> {audio_closeness_display[euc_audio_num]}<br>
                                <strong>Liking:</strong> {av_audio_liking}<br>
                                <strong>Familiarity:</strong> {av_audio_familiarity}</b></i>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"Sound file {sound_file_name} not found. (Sounddatei nicht gefunden)")

            #### MAHALANOBIS ####

            mah_audio_num = 0
            audio_calculator = AudioScoreCalculator(self.user_group_ids)  # Instantiate the calculator

            # Display audio player for each closest sound (Mahalanobis)
            for i, row in closest_sounds_mahalanobis.iterrows():
                av_audio_liking, av_audio_familiarity = audio_calculator.calculate_scores(row)
                
                mah_audio_num += 1

                sound_file_name = row['sound'].strip().lstrip('/')
                sound_file_path = os.path.join(self.audio_files_dir, sound_file_name)
                if os.path.exists(sound_file_path):
                    audio_file = open(sound_file_path, 'rb').read()
                    audio_base64_mahalanobis = base64.b64encode(audio_file).decode()

                    # Define the background color for the audio player container
                    audio_player_color = "#8A2BE2"  # Desired audio player color

                    html_code_mahalanobis = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>Play Audio</title>
                            <style>
                                audio::-webkit-media-controls-panel {{
                                    background-color: {audio_player_color};
                                }}
                                audio::-webkit-media-controls-play-button,
                                audio::-webkit-media-controls-volume-slider,
                                audio::-webkit-media-controls-timeline,
                                audio::-webkit-media-controls-current-time-display,
                                audio::-webkit-media-controls-time-remaining-display {{
                                    color: #FFF;
                                }}
                            </style>
                        </head>
                        <body>
                            <div class="audio-container">
                                <audio id="audioPlayerMahalanobis" src="data:audio/mp3;base64,{audio_base64_mahalanobis}" type="audio/mpeg" controls></audio>   
                            </div>
                        </body>
                    </html>
                    """
                    
                    # Display the audio player in a Streamlit column
                    with col1:
                        components.html(html_code_mahalanobis, height=70, width=320)

                        # Format the information display using Markdown for better styling
                        st.markdown(f"""
                        <div style="
                            font-size: 0.8em;
                            color: #A083F1; 
                            margin-left: 20px; 
                            margin-top: -15px;
                            margin-bottom: 5px;
                            font-family: sans-serif; 
                            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5), 
                                        2px 2px 4px rgba(255, 255, 255, 0.2);">
                            <i><b><strong>Mahalanobis Based:</strong> {audio_closeness_display[mah_audio_num]}<br>
                            <strong>Liking:</strong> {av_audio_liking}<br>
                            <strong>Familiarity:</strong> {av_audio_familiarity}</b></i>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Sound file not found. (Sounddatei nicht gefunden)")

    def get_display_dimension(self, dimension):
        """
        Function to get the dimension in German.
        """
        return f"{dimension} ({self.dimension_display[dimension]})"

    def run(self):
        """
        Runs the SomunicateApp application.
        """
        self.set_style()
        self.show_header()
        data = self.load_and_prepare_data()
        self.groupID_data = self.load_groupID("groupid.csv")
        group_id = self.get_user_group_ids()
        user_ratings, selected_dimensions = self.get_target_dimensions_ratings()
        
        # Check if dimensions are selected and rated
        if selected_dimensions and user_ratings:
            with st.expander("Please Select the Number of Sounds to be Displayed: "):
                top_n = st.slider(f"Display Sound(s)", min_value=1, max_value=10, value = 3, step=1)
            with st.expander(f"Weight Liking and Familiarity"):
                st.session_state.MIN_LIKING = st.slider(f"Weight Min-Liking: ", min_value = 0, max_value = 100, value = 0, step = 1)
                st.session_state.MIN_FAMILIARITY = st.slider(f"Weight Min-Familiarity: ", min_value = 0, max_value = 100, value = 0, step = 1)
            self.display_results(user_ratings, selected_dimensions, top_n, data, group_id)
        
# Run the application
app = SomunicateApp()
app.run()

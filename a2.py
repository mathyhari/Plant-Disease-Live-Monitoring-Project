import streamlit as st
import cv2
import numpy as np
import os


# Expanded dataset with more diseases and corresponding RGB values
disease_dataset = {
    "healthy": np.array([0, 255, 0]),         # Green for healthy plants
    "leaf blight": np.array([255, 0, 0]),       # Red for disease A (leaf blight)
    "powdery mildew": np.array([0, 0, 255]),       # Blue for disease B (powdery mildew)
    "yellow leaf spot": np.array([255, 255, 0]),     # Yellow for disease C (yellow leaf spot)
    "rust": np.array([255, 165, 0]),     # Orange for disease D (rust)
    "leaf curl virus": np.array([128, 0, 128]),     # Purple for disease E (leaf curl virus)
    "fungal rot": np.array([139, 69, 19]),     # Brown for disease F (fungal rot)
    "bacterial wilt": np.array([0, 128, 128]),     # Teal for disease G (bacterial wilt)
    "mosaic virus": np.array([128, 128, 0]),     # Olive for disease H (mosaic virus)
    "leaf scab": np.array([192, 192, 192]),   # Grey for disease I (leaf scab)
    "downy mildew": np.array([0, 255, 255]),     # Cyan for disease J (downy mildew)
}

# Function to predict the disease based on average RGB values in a frame
def predict_disease(frame, dataset):
    avg_color = np.mean(frame, axis=(0, 1))  # Mean across width and height

    # Compare average color to dataset and find the closest match
    min_diff = float("inf")
    predicted_disease = "Unknown"
    
    for disease, color_value in dataset.items():
        diff = np.linalg.norm(avg_color - color_value)  # Euclidean distance
        if diff < min_diff:
            min_diff = diff
            predicted_disease = disease
            
    return predicted_disease

def live_stream_predict():
    stframe = st.empty()  # Placeholder for video stream display
    cap = cv2.VideoCapture(0)  # Accessing the default camera (webcam)

    if not cap.isOpened():
        st.error("Error accessing the camera")
        return

    st.write("Live Stream - Predicting Plant Diseases")
    stop_button = st.button('Stop Live Stream', key='stop_button')
    while st.session_state['live']:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam")
            break

        # Resize frame for processing (optional)
        frame_resized = cv2.resize(frame, (300, 300))

        # Predict disease in the current frame
        predicted_disease = predict_disease(frame_resized, disease_dataset)

        # Display the live frame with prediction
        stframe.image(frame, channels="BGR", caption=f"Predicted Disease: {predicted_disease}")

        # Stop live stream button
        if stop_button:
            st.session_state['live'] = False
            break

    cap.release()

# Streamlit app to process video every 10 seconds
def main():
    st.markdown(
    """
    <style>
    [data-testid="stApp"] {
    background-image: url("https://image.slidesdocs.com/responsive-images/background/green-linear-plant-simple-nature-leaf-powerpoint-background_b2ef89d1c4__960_540.jpg");
    background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.title("Plant Disease Prediction (Live monitoring)")
    st.write("Upload a video of your plant to predict the disease at every 10-second interval.")

    # Sidebar to switch between Video processing and Live Stream modes
    option = st.sidebar.selectbox(
        "Choose an option",
        ("Video (10-second intervals)", "Live Stream")
    )

    if option == "Video (10-second intervals)":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            video_path = os.path.join("temp_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Open video file using OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Error loading video file")
                return

            # Get the FPS (frames per second) of the video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            frame_interval = fps * 10  # Process a frame every 10 seconds

            disease_predictions = []

            st.write("Processing video every 10 seconds...")

            while True:
                # Set the current frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Resize frame for processing (optional)
                frame_resized = cv2.resize(frame, (300, 300))

                # Predict disease for the current frame
                predicted_disease = predict_disease(frame_resized, disease_dataset)
                disease_predictions.append(predicted_disease)

                # Display the frame at this 10-second interval
                st.image(frame_resized, channels="BGR", caption=f"Frame at {frame_count // fps} seconds - Predicted Disease: {predicted_disease}")

                # Move to the next 10-second interval
                frame_count += frame_interval
            
            # Release the video
            cap.release()

            # Calculate the most frequent prediction across all frames processed
            if disease_predictions:
                most_common_disease = max(set(disease_predictions), key=disease_predictions.count)
                st.write(f"Most commonly predicted disease: {most_common_disease}")
            else:
                st.write("No frames were processed.")

    elif option == "Live Stream":
        if 'live' not in st.session_state:
            st.session_state['live'] = False
        if not st.session_state['live']:
            if st.button('Start Live Stream', key="start_button"):
                st.session_state['live'] = True
                live_stream_predict()
        else:
            live_stream_predict()

if __name__ == "__main__":
    main()
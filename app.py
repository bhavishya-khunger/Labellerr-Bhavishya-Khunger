import streamlit as st
import os
import cv2
import json
from tracker import process_video

st.set_page_config(layout="wide")
st.title("🎬 Vehicle and Pedestrian Video Tracker")
st.markdown("Upload a video and this app will track objects using your trained YOLOv8 model and ByteTrack.")

MODEL_PATH = './best.pt' # <-- IMPORTANT: Update this path

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
temp_file_path = None

if uploaded_file is not None:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(temp_file_path)

    if st.button("Start Tracking"):
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at '{MODEL_PATH}'. Please update the path.")
        else:
            st.markdown("### Processing Progress")
            progress_bar = st.progress(0)
            eta_text = st.empty()
            image_placeholder = st.empty()
            
            final_results = {}

            # Loop through the progress updates from the tracker function
            for update in process_video(temp_file_path, MODEL_PATH):
                progress_bar.progress(update["progress"])

                # --- START OF CORRECTION ---
                # Only update the ETA and live frame on progress updates, not the final result.
                if not update["is_done"]:
                    eta_text.text(f"Estimated Time Remaining: {update['eta']} seconds")
                    rgb_frame = cv2.cvtColor(update["annotated_frame"], cv2.COLOR_BGR2RGB)
                    image_placeholder.image(rgb_frame, caption="Live Processing View")
                else:
                    # This is the final update, store it and clear the ETA text
                    final_results = update
                    eta_text.empty()
                # --- END OF CORRECTION ---
            
            st.success("Processing Complete!")
            image_placeholder.empty()
            
            st.markdown("### Sample Annotated Frames")
            sample_paths = final_results.get("sample_paths", [])
            if sample_paths:
                # Create columns to display images side-by-side
                cols = st.columns(len(sample_paths))
                for i, image_path in enumerate(sample_paths):
                    with cols[i]:
                        st.image(image_path, caption=f"Sample Frame {i+1}", use_column_width=True)
            else:
                st.write("No sample frames were generated.")
            
            st.markdown("### Download Results")
            col1, col2 = st.columns(2)

            with col1:
                json_data = final_results.get("json_data", "{}")
                st.download_button(
                    label="Download results.json",
                    data=json_data,
                    file_name="results.json",
                    mime="application/json",
                )
            
            with col2:
                video_path = final_results.get("video_path")
                if video_path and os.path.exists(video_path):
                    with open(video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.download_button(
                        label="Download Annotated Video (.mp4)",
                        data=video_bytes,
                        file_name="annotated_video.mp4",
                        mime="video/mp4"
                    )
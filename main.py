import streamlit as st
import requests
import json
import time
import os
import subprocess
from streamlit_image_comparison import image_comparison

def main():

    command = "uvicorn webapi:app --reload"
    subprocess.Popen(command, shell=True)
    st.set_page_config(layout="centered", page_title="Image Segmentation")
    st.markdown("<h1 style='text-align: center; color: black;'>Car Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Mobile Net V2 and UNET</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        # Using a path relative to the current file
        #current_file_path = os.path.abspath(__file__)
        sample_input = os.path.join('static', 'sample_input.png')
        sample_output = os.path.join('static', 'sample_output.png')
        # Create two columns for image display
        image_comparison(
                img1=sample_input,
                img2=sample_output,
                label1="Sample_Input_image",
                label2="Sample_Processed_image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
        )   
        
    else:

        # Prepare the payload for FastAPI server
        files = {"file": uploaded_file}
        # Make a POST request to the FastAPI server
        start_time = time.time()
        response = requests.post("http://127.0.0.1:8000/upload", files=files)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            input = os.path.join('static', 'Results', 'input.png')
            output = os.path.join('static', 'Results','output.png')
            image_comparison(
                img1=input,
                img2=output,
                label1="Input_image",
                label2="Processed_image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )

            #execution_time = time.time() - start_time
            #st.sidebar.write(f"Time Required {execution_time:.2f} secs.")
        else:
            st.error("Upload failed.")

if __name__ == "__main__":
    main()
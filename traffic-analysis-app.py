import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd

import math
import numpy as np
import cv2

from yolo_results import generate_results




st.set_page_config(layout="wide")


def draw_line_mode(frame):
    detectionLines = []
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("line","freedraw"))

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

            
    image = Image.fromarray(frame)
    width,height = image.size

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        update_streamlit=True,
        height = height,
        width=width,
        drawing_mode=drawing_mode,
        key="canvas1",
    )

	        # Do something interesting with the image data and paths
	        #if canvas_result.image_data is not None:
	        #    st.image(canvas_result.image_data)
	        
    if canvas_result.json_data is not None:

        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        
        res = dict(canvas_result.json_data )
        

        for vals in res['objects']:
            lines = [vals['left']+vals['x1'],vals['top']+vals['y1'],vals['left']+vals['x2'],vals['top']+vals['y2']]
            detectionLines.append(lines)


        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)
    
    return detectionLines    
        
        


###############################################################################



###############################################################################

def main():
    
    uploaded_video = st.sidebar.file_uploader("Background image:", type=["png", "jpg",'mp4'])
    
    if 'vidcap' not in st.session_state:
        st.session_state.vidcap = None 
        
    if 'detectionLines' not in st.session_state:
        st.session_state.detectionLines = None 
	    
	    
    if uploaded_video is not None: # run only when user uploads video
        button = st.button(label = 'get started', key="ta_submit")
        if not button:
            

            vid = uploaded_video.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_video.read()) # save video to disk

            st.markdown(f"""
            ### Files
            - {vid}
            """,
            unsafe_allow_html=True) # display file name

            vidcap = cv2.VideoCapture(vid) # load video from disk
            st.session_state.vidcap = vidcap
            ret, frame = vidcap.read()
            #################################### adding code for allowing the user to add lines########################################
            
            
            st.session_state.detectionLines = draw_line_mode(frame)
            
        else:

            csv = 'Total vehicles detected: 38,lane a-b: 15, lane c-d: 23,Total cars: 35, motorbikes: 0, buses: 0, trucks: 3'

            st.download_button(
                 label="Download data as CSV",
                 data=csv,
                 file_name='traffic_report.csv',
                 mime='text/csv',
             )

            vidcap = st.session_state.vidcap
            detectionLines = st.session_state.detectionLines
            generate_results(vidcap,detectionLines)
            
    
if __name__ == "__main__":

    main()

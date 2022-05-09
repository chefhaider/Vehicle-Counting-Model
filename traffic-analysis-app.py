import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd

import math
import numpy as np
import cv2


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



def readLabelCoordinates(path):
    labelNames = ['car', 'motorbike', 'bus', 'truck']
    txtFile = open(path, "r")
    lineCount = 0
    data = []  # 0: class_name, 1: confidence, 2: left_x, 3: top_y, 4: width, 5: height
    params = []
    while True:
        line = txtFile.readline()
        lineCount += 1
        if lineCount == 1:
            params = line.split(",")
        elif lineCount < 15:
            continue
        if line == 'Objects:\n':
            txtFile.readline()
            line2 = txtFile.readline()
            lineCount += 2
            vehicles = []
            while line2 != '\n':
                lineElements = line2.split(":")
                if lineElements[0] == labelNames[0] or lineElements[0] == labelNames[1] or \
                        lineElements[0] == labelNames[2] or lineElements[0] == labelNames[3]:
                    labels = lineElements[1].split(",")  # 1 or 2
                    if len(labels) == 1:
                        d0 = lineElements[0]
                        d1 = int(lineElements[1].split("%")[0].strip())
                        d2 = lineElements[2].split("t")[0].strip()
                        d3 = lineElements[3].split("w")[0].strip()
                        d4 = lineElements[4].split("h")[0].strip()
                        d5 = lineElements[5].split(")")[0].strip()
                    else:
                        d0 = lineElements[0] + "," + labels[1].strip()
                        d1 = labels[0].split("%")[0].strip() + "," + lineElements[2].split("%")[0].strip()
                        d2 = lineElements[3].split("t")[0].strip()
                        d3 = lineElements[4].split("w")[0].strip()
                        d4 = lineElements[5].split("h")[0].strip()
                        d5 = lineElements[6].split(")")[0].strip()
                    vehicles.append([d0, d1, int(d2), int(d3), int(d4), int(d5)])
                    line2 = txtFile.readline()
                    lineCount += 1
                else:
                    line2 = txtFile.readline()
                    lineCount += 1
            data.append(vehicles)
        if line == "input video stream closed. \n":
            break
    txtFile.close()
    return params, data


def getCenterPoint(left_x, top_y, width, height):
    return (left_x * 2 + width) / 2, (top_y * 2 + height) / 2


def getDistanceFromPointToLine(p1, p2, p3):
    return abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))


def getDistanceFromPointToPoint(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)




def generate_results(video,_detectionLines):



    stframe = st.empty() 



    global x1, y1, drawing
    #video = cv2.VideoCapture(video)
    parameters, labelData = readLabelCoordinates("./dutchLabels.txt") #still need to add paths...
    
    detectionLines = []
    
    for line in _detectionLines:
        detectionLines.append( [int(val) for val in line] )
    
    offset =  20
    velocityOffset = 100
    distanceThreshold = 50
    cameraCoef = 0.06

    #offset = int(parameters[0])             # Predetermined value depending on the video, dutch:50, thai:10, swiss:20
    #velocityOffset = int(parameters[1])     # Predetermined value depending on the video, dutch:100, thai:10, swiss:100
    #distanceThreshold = int(parameters[2])  # Predetermined value depending on the video, dutch:100, thai:40, swiss:50
    #cameraCoef = float(parameters[3].split("\n")[0])           # Predetermined value, dutch:0.04, thai:0.05, swiss:0.068

    frameWidth = int(video.get(3))
    frameHeight = int(video.get(4))
    fps = video.get(5)

    x1 = 0
    y1 = 0
    # [[int(frameWidth / 2), frameHeight], [int(frameWidth / 2), 0]]
    frameCount = 0
    vehicleCount = 0
    lanesCount = [0, 0, 0, 0, 0, 0]
    vehicleTypesCount = [0, 0, 0, 0]
    #detectionLines = []
    previousCentersAndIDs = []
    id = 0
    detectedVehicleIDs = []
    cache = []
    vehicleVelocities = {}
    while True:
        centersAndIDs = []
        ret, frame = video.read()
        
        for dl in detectionLines:        # Plot all detection lines
            cv2.line(frame, (int(dl[0]), int(dl[1])), (int(dl[2]), int(dl[3])), (255, 203, 48), 6)
        
        if not labelData[frameCount]:    # No vehicles in the frame
            print("No vehicle detected in this frame")
        else:
            unavailableIDs = []
            for index, vehicle in enumerate(labelData[frameCount]):
                center = getCenterPoint(vehicle[2], vehicle[3], vehicle[4], vehicle[5])
                sameVehicleDetected = False
                alreadyCounted = False
                for i, c in enumerate(cache):
                    minDistance = distanceThreshold * ((i + 2) / 2)  # Predetermined value depending on the video
                    for cID in c:  # Center point coordinates and ID of a vehicle
                        if cID[2] in unavailableIDs:
                            continue
                        distance = getDistanceFromPointToPoint(center[0], center[1], cID[0], cID[1])
                        if distance < minDistance:   # Same vehicle detected in the previous frame
                            nearestBox = cID[2]
                            minDistance = distance
                            sameVehicleDetected = True
                    if sameVehicleDetected:
                        break

                if not sameVehicleDetected:
                    centersAndIDs.append([center[0], center[1], id])
                    id += 1
                else:
                    centersAndIDs.append([center[0], center[1], nearestBox])
                    unavailableIDs.append(nearestBox)
                if len(centersAndIDs) != 0:
                    vehicleID = centersAndIDs[len(centersAndIDs) - 1][2]
                    # Show vehicle ID
                    cv2.putText(frame, str(vehicleID), (int(centersAndIDs[len(centersAndIDs) - 1][0] + 5),
                        int(centersAndIDs[len(centersAndIDs) - 1][1])), cv2.FONT_HERSHEY_DUPLEX, 1.3, (41, 18, 252), 4)

                cv2.circle(frame, (int(center[0]), int(center[1])), 4, (41, 18, 252), 5)   # Plot center point
                for i, dl in enumerate(detectionLines):
                    p1 = np.array([dl[0], dl[1]])
                    p2 = np.array([dl[2], dl[3]])
                    p3 = np.array([center[0], center[1]])
                    if dl[0] < dl[2]:
                        largerX = dl[2]
                        smallerX = dl[0]
                    else:
                        largerX = dl[0]
                        smallerX = dl[2]
                    if dl[1] < dl[3]:
                        largerY = dl[3]
                        smallerY = dl[1]
                    else:
                        largerY = dl[1]
                        smallerY = dl[3]
                    # Calculate the distance from the vehicle to the detection line  to count the number of vehicles
                    if getDistanceFromPointToLine(p1, p2, p3) < offset and \
                            smallerX - offset < center[0] < largerX + offset and \
                            smallerY - offset < center[1] < largerY + offset:
                        for dvi in detectedVehicleIDs:
                            if dvi == vehicleID:
                                cv2.line(frame, (dl[0], dl[1]), (dl[2], dl[3]), (90, 224, 63), 6)
                                alreadyCounted = True
                                break
                        if not alreadyCounted:
                            detectedVehicleIDs.append(vehicleID)
                            vehicleCount += 1
                            cv2.line(frame, (dl[0], dl[1]), (dl[2], dl[3]), (90, 224, 63), 6)
                            lanesCount[i] += 1
                            if vehicle[0] == 'motorbike':
                                vehicleTypesCount[1] += 1
                            elif vehicle[0] == 'bus':
                                vehicleTypesCount[2] += 1
                            elif vehicle[0] == 'truck':
                                vehicleTypesCount[3] += 1
                            else:
                                vehicleTypesCount[0] += 1

                    # Calculate the distance from the vehicle to the detection line in order to estimate the velocity
                    if getDistanceFromPointToLine(p1, p2, p3) < velocityOffset and \
                            smallerX - velocityOffset < center[0] < largerX + velocityOffset and \
                            smallerY - velocityOffset < center[1] < largerY + velocityOffset:
                        speedometer = []
                        foundInPreviousFrame = False
                        pixelsOverFrames = []
                        for s in range(len(cache) - 1):
                            for c in range(len(cache[s])):
                                if cache[s][c][2] == vehicleID:
                                    if not foundInPreviousFrame:
                                        pixelsOverFrames.append(getDistanceFromPointToPoint(cache[s][c][0], cache[s][c][1],
                                                                                            center[0], center[1]) / (s + 1))
                                        foundInPreviousFrame = True
                                    cacheHit = False
                                    for ss in range(s + 1, len(cache)):
                                        for cc in range(len(cache[ss])):
                                            if cache[ss][cc][2] == vehicleID:
                                                pixelsOverFrames.append(getDistanceFromPointToPoint(cache[s][c][0],
                                                    cache[s][c][1], cache[ss][cc][0], cache[ss][cc][1]) / ss - s)
                                                cacheHit = True
                                                break
                                        if cacheHit:
                                            break
                        if len(pixelsOverFrames) != 0:
                            averageVelocity = sum(pixelsOverFrames) / len(pixelsOverFrames) * fps    # Pixel / second
                            averageVelocity *= cameraCoef * 3.6    # km / h
                            if vehicleID not in vehicleVelocities:
                                vehicleVelocities[vehicleID] = [averageVelocity]
                            else:
                                vehicleVelocities[vehicleID].append(averageVelocity)
                if vehicleID in vehicleVelocities and len(vehicleVelocities[vehicleID]) > 2:
                    grandAverageVelocity = sum(vehicleVelocities[vehicleID]) / len(vehicleVelocities[vehicleID])
                    cv2.putText(frame, str(abs(int(averageVelocity))) + "km/h", (int(vehicle[2]), int(vehicle[3] - 50)),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (41, 18, 252), 4)

        for index, dl in enumerate(detectionLines):
            cv2.putText(frame, "Vehicles: " + str(lanesCount[index]), (int((dl[0] + dl[2]) / 2) + 10,
                  int((dl[1] + dl[3]) / 2) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.3, (209, 21, 77), 4)
        
        
        
        #output video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(gray)
        
        
        frameCount += 1

        cacheSize = 5
        cache.insert(0, centersAndIDs.copy())
        if len(cache) > cacheSize:
            del cache[cacheSize]

        if frameCount == len(labelData) - 2:
            break

    st.write("Total vehicles detected: "+ str(vehicleCount))
    st.write("Total cars: ", str(vehicleTypesCount[0]), " motorbikes: ", str(vehicleTypesCount[1])," buses: ", str(vehicleTypesCount[2]), " trucks: ", str(vehicleTypesCount[3]))
          
          
         

    video.release()



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
            

            vidcap = st.session_state.vidcap
            detectionLines = st.session_state.detectionLines
            generate_results(vidcap,detectionLines)
            
    
if __name__ == "__main__":

    main()
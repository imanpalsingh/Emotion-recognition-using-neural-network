'''
 *
 * Author : - Imanpal Singh <imanpalsingh@gmail.com>
 * Description : -  Emotion Recognition from live camera feed.
 * Technology stack :- Numpy, Pandas, OpenCV, Scikit-Learn
 * Date created : - 11-10-2019
 * Last modified : - 19-11-2019
 * Version : - 0.1.0
 *
'''

'''
 *
 * Change log :

   (0.0.1) : - Dataset: Now using Olevetti Faces Dataset.
   (0.1.0) : - Dataset : Now using Extended cohen Face Dataset.
   
 * File Description : - This file loads the data, preprocesses it and saves it into a csv file format
 
'''

# Driver Function
def Apply():
#################################### IMPORTS #################################

    print("Loading required libraries")

    # Pandas 0.25.0
    import pandas as pd

    # OpenCV 4.1.1
    import cv2

    import glob

    # Numpy 1.17.3
    import numpy as np

    print("Loaded successfully")

    ################################### VARIABLES #################################

    print("Initializing variables and loading models")

    # Loading required Cascades/models
    FACE_CAS = cv2.CascadeClassifier('Models/face1.xml')

    # Defining dataset path
    DATASET = 'Datasets/'

    # Defining emotions on the basis of dataset folder structure
    EMOTIONS = ['happiness','neutral','sadness']

    # List to hold image locations in the directories
    IMAGE_LOCS = []

    # List to hold preprocessed images
    IMAGE_DATA = []

    # List to hold mapped emotions to the images
    EMOTION_MAP = []

    # Variable to hold the emotion of the current iteration
    CURRENT_EMOTION = 0

    #Filename to save Image data to
    FILENAME_IMAGE = 'Datasets//images.csv'

    #Filename to save target values (emotions) to
    FILENAME_TARGET = 'Datasets//target.csv'

    print("Loaded successfully")

    ################################# DATA EXTRACTION ######################################

    print("Extracting image locations")

    # Extracting the image locations
    for emotion in EMOTIONS:
        IMAGE_LOCS.append([img for img in glob.glob(DATASET + emotion +"/*.png")])

    print("Extracted successfully")    

    ############################## DATA EXTRACTION #######################################

    print("Preprocessing Data")

    # For each emotion in Image locations
    for emotion in IMAGE_LOCS:

        # For each image in a emotion
        for images in emotion:

            # Read the image (in grayscale )
            image = cv2.imread(images,cv2.IMREAD_GRAYSCALE)

            # Detecting face in the Image
            face = FACE_CAS.detectMultiScale(image,1.5,5)

            # If face is found
            if(len(face)):
                # Get the x,y values and w,h values
                x,y,w,h = face[0,:]

                # Cropping the face out of the original image (ROI)
                face_cropped = image[y:y+h,x:x+w]

                # Resizing the image
                face_resized = cv2.resize(face_cropped, (64,64),interpolation = cv2.INTER_AREA)
            
                # Dimensionality Reduction
                face_1d = face_resized.reshape(-1).astype('float32')
                
                # Sacling the data
                face_1d = np.divide(face_1d,255.0)

                # Saving the data
                IMAGE_DATA.append(face_1d)

                # Saving the corresponding emotion (label encoded)
                EMOTION_MAP.append(CURRENT_EMOTION)

            
        # Updating the emotion for next iteration
        CURRENT_EMOTION+=1   
            


    print("Preprocessing Successfull")

            
    #################################### DATA SAVING ##############################

    # Creating pandas dataframes
    print("Saving Data")
    Images = pd.DataFrame(IMAGE_DATA)
    Targets = pd.DataFrame(EMOTION_MAP)

    # Writing to csv file
    Images.to_csv(FILENAME_IMAGE)
    Targets.to_csv(FILENAME_TARGET)
    print("Data saved")
    print("Instruction executed sucessfully")


if __name__ =='__main__':

    print(" Warning! This file is not supposed to be run as the main file ")

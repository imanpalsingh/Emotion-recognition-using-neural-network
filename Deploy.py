'''
 *
 * Author : - Imanpal Singh <imanpalsingh@gmail.com>
 * Description : -  Emotion Recognition from live camera feed.
 * Technology stack :- Numpy, Pandas, OpenCV, Scikit-Learn
 * Date created : - 11-10-2019
 * Last modified : - 19-11-2019
 * Version : - 0.0.1
 *
'''
'''
 * Chnage log :
    (0.0.1) : Now Immediately shows the captured picture with the emotion

 * File Description : - This file uses the camera feed and preprocesses it before feeding it to the model to finally recoganise the emotion.
                        
 *
 
'''

def Apply():
    ############################### IMPORTS ############################

    print("Importing required libraries.")

    # OpenCV 4.1.1
    import cv2

    # Numpy version 1.17.3
    import numpy as np

    #Joblib 0.13.2
    import joblib

    import math

    print("Done")

    ############################### VARIABLES/LOADING MODELS #################################

    print("Loading the model")

    #Loading trained algorithm
    model = joblib.load('Models/trained_algorithm.pkl')

    #Loading face_cascade
    FACE_CAS = cv2.CascadeClassifier('Models/face1.xml')

    # Defining emotions mappings to prediction
    EMOTIONS = ['Happiness','Neutral','Sadness']

    IS_SELECTED = False

    print("Done")

    ################################### Video frame #############

    print("Initialising video cam")

    # Starting video camera
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    MainKey=0
    
    print(" When ready with the expression Press 's' ")
    while True:

        #Reading a frame
        _,image = cap.read()

        

        # Converting the image into grayscale
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

       
        if MainKey == 0:
            Key =0
            Key = cv2.waitKey(30)
        if MainKey ==100:
            print("No face was found")
            MainKey =0
            Key=0
        if Key==ord('s'):
            
            
        
            MainKey+=1
             # Detecting face in the Image
            face = FACE_CAS.detectMultiScale(image_gray,1.5,5)
            if(len(face)):
            # Get the x,y values and w,h values
                x,y,w,h = face[0,:]
        
                x_center = x+w//2
                y_center = y+h//2;

                radius = int(math.sqrt( ( x - x_center )**2 + ( y - y_center )**2) )

            #Drawing a circle around the face
                cv2.circle(image,(x_center,y_center),radius,(255,255,255),1)
        
            # Cropping the face out of the original image (ROI)
                face_cropped = image_gray[y:y+h,x:x+w]

            # Resizing the image
                face_resized = cv2.resize(face_cropped, (64,64),interpolation = cv2.INTER_AREA)
            
            # Dimensionality Reduction
                face_1d = face_resized.reshape(-1).astype('float32')
                
            # Sacling the data
                face_1d = np.divide(face_1d,255.0)

            # Calling the predict function
                predicted_emotion = int(model.predict(face_1d.reshape(1,-1)))

            # Putting the text on the photo
                cv2.putText(image, EMOTIONS[predicted_emotion], (x_center, y_center-radius), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                Captured = image.copy()
                IS_SELECTED = True
                print("Face found !")
                break;
                MainKey=0
        flipped = cv2.flip(image,1)
        cv2.imshow("Frame",flipped)

        if Key == ord('q'):
            break;
     #Releasing video cam
    cap.release()

    #Cleaning all windows to avoid memory leak
    cv2.destroyAllWindows()

    if IS_SELECTED == True:

        cv2.imshow('Output',Captured);
        print("Press any key to finish")
        cv2.waitKey()
        cv2.destroyAllWindows()
    
   


if __name__ == '__main__':

    print(" Warning ! This file is not supposed to be run as the main file ")

        


    

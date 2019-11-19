# Emotion-recognition-using-neural-network



<h2>Emotions Recognition ( Happiness, Neutral, Sadness) from camera feed using neural Networks </h2>

Algorithm Multi Layer Perceptron from Scikit-Learn's neural_network module is used. The dataset consists of images taken from the Cohn-Kanade Dataset [Refernce Below] 

<h3> Procedure </h3>

1) To use new Images/data
Put the images in three different folders named as Happy Neutral and angry inside the Dataset folder.

Dataset
./ Angry
./ Neutral
./ Angry

Run the Main File and type command 'load data'. This will load, preprocess and store the data as in csv format (pixel values scaled).

2) To train the model

Run the Main File and type command 'train model'. This will start the nprocess to train model and save the pickle file of the model

3) To run

Run the Main File and type 'run'. This will initiate the process of taking a webcam feed.
To predict, press 's' when ready. The algorithm will try to find a face for next 100 frames. When found it will extract it , preprocess it and predict the emotion.









<h2> Refernces </h2>

- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

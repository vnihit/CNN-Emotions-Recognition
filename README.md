# CNN-Emotions-Recognition
Convolutional Neural Network based Emotions Recognition System

This is an implementation of Convolutional Neural Network based Human Emotions Recognition System. 
 

Dataset: 

The dataset used for this project is the FER-2013 Facial Expression Recognition Challenge from Kaggle. The dataset consists of 28,709 pictures of human faces with seven different emotions (happy, neutral, angry, disgusted, fearful, sad, surprised). 
The dataset and the trained model files are available at - https://drive.google.com/open?id=1tsO3sWSDCYJqKmM7wBy3XNcRS3q1ZVwQ
The Google Drive also contains the copy of the code and the virtual environment.

Usage
The code for training and testing is available in the submitted files. The dependencies are listed in requirements.txt In order to test/run, download and extract the files from Google Drive and follow the instructions-
	•	If using the virtual environment run - pip3 install virtualenv 
	•	If running without the virtual environment install dependencies from the requirements.txt file: pip3 install -r requirements.txt

1) in the directory activate the virtual-environment by typing:
	
	source mlproj/bin/activate

    The virtual environment comes with all dependencies installed for python 3.6.

2) The dataset in the csv file needs to be transformed to generate image and label data. Run-
	
	python3 csv_to_numpy.py
	
3) To train the model -
	
	python3 emo_recogniser.py train

4) To test the trained model (webcam required) -
	
	python3 run_emotion_recognizer.py 


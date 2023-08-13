# Gesture Arithmetic - Landmark Extraction and Support Vector Machine for Arithmetic Calculations

This is an open-source implementation of a gesture detection model that uses the user's webcam and accurately predicts the user's hand gestures to do basic arithmetic like addition and multiplication.

* It currently only supports 14 gestures, the numbers 0-10, the plus symbol (caused by closing the fist and raising the pinky finger), the multiplication symbol (caused by crossing the two index fingers), and the calculate button used to calculate the result. (caused by closing the fist and extending the thumb sideways)

* In models/ are two pickled SVM models, one of which was trained without data augmentation and performs poorly on right-hand gestures (57% accuracy when testing); after applying data augmentation to the landmarks (horizontal flip) and their flattened versions, the test set has 100% accuracy on both left and right hand (shown in train.ipynb).

https://github.com/awesominat/gesture-arithmetic/assets/110934811/e16571d4-714c-4f34-88e3-54ceb0478a91

https://github.com/awesominat/gesture-arithmetic/assets/110934811/a18faffa-4c5c-48e2-97e2-16a4a461896e

## Salient Features
* Waits for 5 loop iterations before confirming input
* Respects PEMDAS order of operations when doing the math
* Supports left and right-hand gestures based on a left-hand training dataset that was augmented to work on the right hand via data augmentation
* Uses Google's MediaPipe library for landmark extraction (21 landmarks) on each hand (42 total landmarks)
* Processes landmarks and passes them through an SVM for classification inspired by Nguyten et. al's 2014 paper
* Has previous implementations of different approaches stored (YoloV* + CNN and mediapipe + NN)
* Reasonably fast (> 30 fps)

## Training on custom gestures/data
More training data can be added by creating two folders called new_train_data/data and placing images of any size in with the naming convention "CLASS INDEX" (for example, "0 4.jpg") and then running train.ipynb.

train.ipynb processes the images and automatically generates a test set. Creating new gestures only requires changing the image_dataset_loader.py's translation dictionary at the beginning following the same format.

## Setup
Step 1. Install requirements.txt
Install everything in the requirements.txt file (preferably through a venv as well)
Python 3.11.4 was used.

Step 2. Run the train.ipynb notebook.
Since the training data isn't provided in the GitHub, a pickled version of the two models is stored in models/ for ease of use. The live webcam detection is in the final cell.

## Pitfalls
A pitfall of the project is that there wasn't any data on rotated gestures/landmarks (i.e., my hand was straight), which causes the models to perform slightly worse if the hand/fingers aren't in the correct orientation. Thus, I plan to apply more data augmentation techniques to both the images and landmarks to make the SVM more robust and less error-prone (15-degree rotations).

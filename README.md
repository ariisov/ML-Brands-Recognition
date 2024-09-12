#Brand Image Recognition with CLIP

#Overview

This project aims to classify product images based on their brand using the CLIP model. The CLIP (Contrastive Language-Image Pretraining) model, developed by OpenAI, enables learning visual concepts from textual descriptions.

#Project Structure

Data Preparation: Images and brand labels are used to train and evaluate the model.
Data Augmentation: The dataset has been augmented to improve model performance and robustness.
Model Training: The CLIP model is used to extract features from images and text, which are then used to train a classification model.
Testing: The trained model is evaluated on a test dataset to assess its performance.
Files and Directories

brands.csv: CSV file containing the image filenames and their corresponding brand names.
brandsImages/: Directory containing images of the products.
#Setup

#Install Dependencies
Ensure you have Python and the necessary libraries installed.

#Prepare Data
First, files are created after the dataset is augmented. To do this, run the file PrepareData.py . Ensure that the augmented_brands.csv file and augmentedBrandsImages directory are in the correct locations specified in the script.
First, files are created after the dataset is augmented. To do this, run the file PrepareData.py . And only after that run the main file TrainModel.py
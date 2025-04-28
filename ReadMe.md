# Neural Machine Translation Project
https://prateeksarna-24.github.io/Medical-Record-Summarizer/
## Overview

This project focuses on Neural Machine Translation (NMT) using a basic Encoder-Decoder architecture. The goal is to translate French sentences into English using a trained model.

## Directory Structure
The folder contains 2 main sub-folders : 
### 1. Data
Data Folder has :
- fra-eng data folder which contains the dataset.
- Contains train and test data which has been created during program execution.
- Contains saved models file.
- Contains Word Index Json Files.
- Contains model architecture figures.
- Plots.
### 2. Project Code Files 
Project folder has 4 files.
- `NMT_Module.py` which contains all the functions used in this project.
- `Part 1 - Data Analysis & Cleaning`
- `Part 2 - Building and Training Encoder Decoder Model`
- `Part 3 - Prediction and Bleu Score`
## Project Structure

### 1. Exploratory Data Analysis & Cleaning

In this part of the project, you performed the following tasks:

- Loaded data from [manythings.org](https://www.manythings.org/anki/).
- Explored and cleaned the data.
- Handled missing values.
- Cleaned and preprocessed text data:
  - Removed extra spaces.
  - Converted English text to lowercase.
  - Expanded contractions.
  - Removed digits and punctuation.
- Visualized and analyzed the length of text by the number of words and characters.
- Created word clouds for English and French words.
- Prepared data for NMT by adding start and end tokens to the target sequence.
- Split the data into training and testing sets.
- Saved cleaned datasets.

### 2. Building Encoder - Decoder Model

In this part, you built the neural network model for machine translation:

- Defined paths for training and testing data.
- Loaded clean training data.
- Preprocessed data for the encoder-decoder model.
- Created functions for converting text sequences to numerical sequences and obtaining padded sequences.
- Built the encoder and decoder models.
- Created a full model by combining the encoder and decoder.
- Trained the model with 15 epochs.
- Saved the encoder, decoder, and full models.
- Saved word indices and common maximum sequence length.

#### Model Architecture

![Model Architecture](Data_Files\\model_architecture.png)



### 3. Prediction and BLEU Score

This section focused on:

- Importing test data.
- Adjusting test data based on the maximum common sequence length.
- Loading word indices and trained models.
- Defining a function for predicting translations.
- Predicting translations for sample sentences.
- Computing BLEU scores for the predicted samples.
- Plotting the BLEU scores over epochs.
- Presenting the conclusion and future work.


## BLEU Score Plot

![BLEU Score Over Epochs](Data_Files/Bleu_score.png)

## Conclusion & Future Work

### Conclusion

- Created a modularized `NMT_Module.py` file for reusable functions.
- Implemented a basic Encoder-Decoder architecture.
- Trained the model with limited parameters.
- Achieved an accuracy of 70%.

### Future Work

The future work includes:

- Implementing an Attention Model.
- Exploring distributed training for a larger model.
- Incorporating MLOps concepts for data pipelining and model deployment.
- Developing a Streamlit web application.

### Submitted By

Prateek Sarna

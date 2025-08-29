# CoffeeBeanAI - Coffee Bean Analysis System

## Author : Randi Hansani

---

# CoffeeBeanAI

A comprehensive machine learning system for coffee bean analysis, featuring both image classification and tabular data prediction capabilities.

- Image classification to verify coffee beans and identify bean types
- Predictive modeling for coffee bean defects and quality assessment
- REST API endpoints for both image-based and data-based predictions

## Features

### Image Classification

- Verify if an image contains coffee beans
- Classify the type of coffee bean
- Confidence scores for predictions

### Tabular Data Prediction

- Predict bean quality based on symptoms and conditions
- Identify potential defects and their causes
- Probability scores for each prediction

## Installation

1.  Clone the repository:

        git clone https://github.com/Radeee21/CoffeeBeans_AI_Backend.git
        cd CoffeeBeans_AI_Backend

2.  Install dependencies:

        pip install -r requirements.txt

3.  Download the pre-trained models and place them in the appropriate directories

## Usage

### Running the API

    python app.py

### Endpoints

#### Image Prediction

**Endpoint:** `/predict_image`

**Method:** POST

**Input:** Image file

**Output:** JSON with classification results

**Example Response:**

{
"is_coffee_bean": true,
"is_coffee_bean_confidence": "98.76%",
"bean_type_confidence": "95.32%",
"predicted_class": "Black Beans"
}

#### Tabular Data Prediction

**Endpoint:** `/predict_tabular`

**Method:** POST

**Input:** JSON with feature values

**Output:** JSON with prediction results

**Example Request:**

{
"symptoms_lable": 5,
"category": 2,
"region": 4,
"dehydration_Duration": 1,
"caught_Rain/Mist": 1
}

**Example Response:**

{
"bean_Quality": {
"prediction": "Bitter or burnt flavor, downgraded quality",
"probability": "87%"
},
"cause_condition": {
"prediction": "Overfermentation or delayed drying",
"probability": "92%"
},
"defect_Name": {
"prediction": "Black Beans",
"probability": "85%"
}
}

## Model Training

The repository includes training scripts for both image classification models:

- `train_classification.py` - Trains the bean type classifier
- `train_verification.py` - Trains the coffee bean verification model
- `train.py` - Trains the tabular data prediction model

## Requirements

- Python 3.11.13+
- Flask
- PyTorch
- Transformers
- scikit-learn
- pandas
- Pillow
- torchvision

**Note:** See `requirements.txt` for specific package versions.


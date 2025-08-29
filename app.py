from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS
import numpy as np
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
import joblib
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

#Hansani
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
VERIFICATION_MODEL_PATH = "ai_models/beans_verification_model.pth"
CLASSIFICATION_MODEL_PATH = "ai_models/beans_classification_model.pth"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model_references = joblib.load("ml_models/model_references.pkl")
ml_models = {
    "bean_Quality": joblib.load(model_references["bean_Quality"]),
    "cause_condition": joblib.load(model_references["cause_condition"]),
    "defect_Name": joblib.load(model_references["defect_Name"])
}

with open("ai_models/bean_label_mappings.json", "r") as f:
    mappings = json.load(f)

def decode(column, value):
    # Map between model target names and JSON keys
    key_mapping = {
        "bean_Quality": "Impact on Bean Quality",
        "cause_condition": "Cause/Condition",
        "defect_Name": "Defect Name"
    }
    
    json_key = key_mapping.get(column, column)
    return mappings.get(json_key, {}).get(str(value), f"Unknown ({value})")

def load_model(model_path):
    """Load a single model from a specific path"""
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model
        num_classes = len(checkpoint['class_names'])
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
         
        return model, checkpoint['class_names']
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")

# Load models
try:
    verification_model, VERIFICATION_CLASSES = load_model(VERIFICATION_MODEL_PATH)
    classification_model, CLASSIFICATION_CLASSES = load_model(CLASSIFICATION_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit(1)
    
    
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file:
        return jsonify({'error': 'Empty file provided'}), 400
    
    try:
        # Read image directly into memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Verification step
        with torch.no_grad():
            verification_outputs = verification_model(image_tensor)
            verification_probs = torch.nn.functional.softmax(verification_outputs.logits, dim=1)
            is_bean = torch.argmax(verification_probs, dim=1).item()
            bean_confidence = verification_probs[0][is_bean].item()
        
        # Apply threshold (90% confidence)
        is_coffee_bean = bool(is_bean) and bean_confidence >= 0.9
        
        result = {
            'is_coffee_bean': is_coffee_bean,
            'is_coffee_bean_confidence': f"{bean_confidence*100:.2f}%",
            'bean_type_confidence': None,
            'predicted_class': None,
        }   
        
        # Classification step only if it's a coffee bean with sufficient confidence
        if is_coffee_bean:
            with torch.no_grad():
                classification_outputs = classification_model(image_tensor)
                classification_probs = torch.nn.functional.softmax(classification_outputs.logits, dim=1)
                pred_class = torch.argmax(classification_probs, dim=1).item()
                type_confidence = classification_probs[0][pred_class].item()
            
            result.update({
                'bean_type_confidence': f"{type_confidence*100:.2f}%",
                'predicted_class': CLASSIFICATION_CLASSES[pred_class]
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    data = request.get_json()

    input_df = pd.DataFrame([{
        "symptoms_lable": data.get("symptoms_lable"),
        "category": data.get("category"),
        "region": data.get("region"),
        "dehydration_Duration": data.get("dehydration_Duration"),
        "caught_Rain/Mist": data.get("caught_Rain/Mist")
    }])

    response = {}
    
    for target_name, model in ml_models.items():
        try:
            # Get prediction
            pred = model.predict(input_df)[0]
            
            # Get probabilities (handling different sklearn versions)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                if isinstance(proba, list):
                    proba = np.array(proba)
                if proba.ndim == 2 and proba.shape[0] == 1:
                    proba = proba[0]
                prob_value = proba[pred] if isinstance(pred, (np.integer, int)) else proba[0][pred]
            else:
                prob_value = 1.0
            
            response[target_name] = {
                "prediction": decode(target_name, pred),
                "probability": f"{prob_value * 100:.0f}%"
            }
            
        except Exception as e:
            response[target_name] = {
                "prediction": "Error in prediction",
                "probability": "0%"
            }

    return jsonify(response)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5050, debug=False)
    app.run(host='172.20.10.2', port=5050, debug=True)
    #  app.run(host='192.168.1.4', port=5050, debug=True)

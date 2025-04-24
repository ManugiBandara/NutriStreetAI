from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), "model", "efficientnetb2_100e_model.h5")
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ["chapati", "dal makhani", "dhokla", "idli", "jalebi"]

# Load the nutritional database
database_path = 'data/nutritional_data.csv'
nutritional_data = pd.read_csv(database_path)

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Rule-based suitability analysis
def get_suggestion(food_name, disease):
    food_info = nutritional_data[nutritional_data['Food Name'] == food_name]
    if food_info.empty:
        return {"error": "Food item not found in the database."}

    origin = food_info['Origin'].values[0]
    food_type = food_info['Food Type'].values[0]
    caution = food_info['Caution'].values[0]
    calories = food_info['Total Calories'].values[0]
    carbs = food_info['Total Carbs'].values[0]
    fats = food_info['Total Fats'].values[0]
    protein = food_info['Total Protein'].values[0]
    sugar = food_info['Total Sugar'].values[0]
    sodium = food_info['Total Sodium'].values[0]
    vitamins = food_info['Vitamin Content'].values[0]
    ingredients = food_info['Ingredients'].values[0].lower()
    cooking_method = food_info['Cooking Method'].values[0].lower()

    suitability = "Unknown"
    explanation = ""

    # Normalize fields
    is_deep_fried = "deep fried" in cooking_method
    is_pan_fried = "pan fried" in cooking_method
    is_steamed_tempered = "steamed" in cooking_method and "tempered" in cooking_method
    is_steamed_only = cooking_method.strip() == "steamed"
    is_boiled = "boiled" in cooking_method
    has_sugar = "sugar" in ingredients

    if disease.lower() == "diabetes":
        if is_deep_fried and has_sugar:
            suitability = "Not Suitable"
            explanation = "Not suitable due to high content of sugar and oil. May spike blood glucose levels rapidly."
        elif is_pan_fried:
            suitability = "Suitable"
            explanation = "Suitable. Wheat flour is rich in nutrients, vitamins and fiber which reduces the sugar absorption."
        elif is_steamed_tempered:
            suitability = "Suitable"
            explanation = "Suitable. Besan gram flour is a pulse flour made from chana dal or chickpea flour which is rich in nutrients and fibers."
        elif is_steamed_only:
            suitability = "Moderate"
            explanation = "Moderate. Black gram lentils and fenugreek seeds include nutrients and antidiabetic effects. Be mindful to consume in less amounts due to higher content of rice. "
        elif is_boiled:
            suitability = "Suitable"
            explanation = "Suitable. Lentils and beans are rich in antioxidants and have low saturated fats."

    elif disease.lower() == "high cholesterol":
        if is_deep_fried and has_sugar:
            suitability = "Not Suitable"
            explanation = "Not suitable due to high sugar and oil content."
        elif is_pan_fried:
            suitability = "Suitable"
            explanation = "Suitable. Wheat flour is rich in nutrients, vitamins and fiber which reduces the cholesterol absorption. Be mindful to consume in lesser amounts if excess oil is added."
        elif is_steamed_tempered:
            suitability = "Moderate"
            explanation = "Moderate. Besan gram flour is a pulse flour made from chana dal or chickpea flour which is rich in nutrients and fibers. Be mindful when consuming in larger amounts since oil and coconut is included."
        elif is_steamed_only:
            suitability = "Suitable"
            explanation = "Suitable. Black gram lentils and fenugreek seeds include nutrients and antidiabetic effects which reduces the risk of heart diseases."
        elif is_boiled:
            suitability = "Not Suitable"
            explanation = "Not suitable due to fresh cream and ghee. Small portions are less harmful as beans and lentils include antioxidants and low saturated fats which lower to risk of diabetes and heart diseases."

    elif disease.lower() in ["both", "diabetes, high cholesterol", "high cholesterol, diabetes"]:
        if is_deep_fried and has_sugar:
            suitability = "Not Suitable"
            explanation = "Not suitable due to high sugar and oil content."
        elif is_pan_fried:
            suitability = "Suitable"
            explanation = "Suitable. Wheat flour is rich in nutrients, vitamins and fiber which reduces the absorption of sugar and cholesterol."
        elif is_steamed_tempered:
            suitability = "Moderate"
            explanation = "Moderate. Besan gram flour is a pulse flour made from chana dal or chickpea flour which is rich in nutrients and fibers. Be mindful when consuming in larger amounts since oil and coconut is included."
        elif is_steamed_only:
            suitability = "Moderate"
            explanation = "Moderate. Black gram lentils and fenugreek seeds include nutrients and antidiabetic effects. Be mindful to consume in less amounts due to higher content of rice."
        elif is_boiled:
            suitability = "Not Suitable"
            explanation = "Not suitable due to fresh cream and ghee. Small portions are less harmful as beans and lentils include antioxidants and low saturated fats which lower to risk of diabetes and heart diseases."

    else:
        suitability = "No rules for this condition."
        explanation = "Guidelines not found for this disease."

    return {
        "Food Name": food_name,
        "Origin": origin,
        "Food Type": food_type,
        "Caution": caution,
        "Total Calories": calories,
        "Total Carbs": carbs,
        "Total Fats": fats,
        "Total Protein": protein,
        "Total Sugar": sugar,
        "Total Sodium": sodium,
        "Vitamin Content": vitamins,
        "Ingredients": ingredients,
        "Suitability": suitability,
        "Suitable Suggestion": explanation
    }

# Landing page 
@app.route('/')
def landing():
    return render_template('landing.html')

# Home page (main page)
@app.route('/home')
def home():
    return render_template('home.html')

# faq page
@app.route('/faq')
def faq():
    return render_template('faq.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html') 

# Classification route
@app.route('/classify', methods=['POST'])
def classify_food():
    selected_disease = request.form['disease']
    food_image = request.files['food_image']

    if food_image:
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], food_image.filename)
        food_image.save(image_path)

        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]

        if selected_disease.lower() == "none":
            food_info = nutritional_data[nutritional_data['Food Name'] == predicted_class]
            if food_info.empty:
                return jsonify({"error": "Food item not found."}), 400

            return render_template('result.html',
                disease="None",
                predicted_food=predicted_class,
                origin=food_info['Origin'].values[0],
                food_type=food_info['Food Type'].values[0],
                caution=food_info['Caution'].values[0],
                calories=food_info['Total Calories'].values[0],
                carbs=food_info['Total Carbs'].values[0],
                fats=food_info['Total Fats'].values[0],
                protein=food_info['Total Protein'].values[0],
                vitamins=food_info['Vitamin Content'].values[0],
                sugar=food_info['Total Sugar'].values[0],
                sodium=food_info['Total Sodium'].values[0],
                suggestion="Be mindful when consuming in large amounts.",
                ingredients=food_info['Ingredients'].values[0],
                image_url=image_path
            )

        # Apply rule-based logic for "Both" condition
        if selected_disease.lower() == "both":
            suggestion = get_suggestion(predicted_class, "both")  # Apply both disease rules
        else:
            # Apply logic for a single disease
            suggestion = get_suggestion(predicted_class, selected_disease)

        return render_template('result.html',
            disease=selected_disease.capitalize(),
            predicted_food=suggestion["Food Name"],
            origin=suggestion["Origin"],
            food_type=suggestion["Food Type"],
            caution=suggestion["Caution"],
            calories=suggestion["Total Calories"],
            carbs=suggestion["Total Carbs"],
            fats=suggestion["Total Fats"],
            protein=suggestion["Total Protein"],
            vitamins=suggestion["Vitamin Content"],
            sugar=suggestion["Total Sugar"],
            sodium=suggestion["Total Sodium"],
            suggestion=suggestion["Suitable Suggestion"],
            ingredients=suggestion["Ingredients"],
            image_url=image_path
        )

    return jsonify({"error": "No image uploaded."}), 400

if __name__ == '__main__':
    app.run(debug=True)

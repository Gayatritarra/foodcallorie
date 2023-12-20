from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage import exposure, io, color
from skimage.transform import resize
import tempfile

app = Flask(__name__)

# Load the trained scikit-learn model
with open('Model1.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_test_image(image_path, target_size=(128, 128)):
    # Load the test image
    image = io.imread(image_path)
    
    # Convert to grayscale
    gray_image = color.rgb2gray(image)
    
    # Resize the image
    resized_image = resize(gray_image, target_size)
    
    # Scale the resized image back to the range 0-255
    scaled_image = (resized_image * 255).astype(np.uint8)
    
    # Extract HOG features
    fd, hog_image = hog(scaled_image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    
    # Rescale the HOG image
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return fd

cal_df = pd.read_csv('calories.csv')
cal_df = cal_df[['FoodItem', 'Cals_per100grams']]

cal_df['FoodItem'] = cal_df['FoodItem'].str.lower()
cal_df['FoodItem'] = cal_df['FoodItem'].str.replace(' ', '_')
cal_df['Cals_per100grams'] = cal_df['Cals_per100grams'].str.replace(' cal', '')
cal_df['Cals_per100grams'] = cal_df['Cals_per100grams'].astype(float)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None  # Initialize prediction result as None
    
    # Get the uploaded image file from the request
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        uploaded_file.save(temp_file.name)
        
        # Preprocess the image for prediction
        processed_image = preprocess_test_image(temp_file.name)
        
        # Make a prediction using the loaded model
        predicted_class = model.predict([processed_image])[0]

        calories_info = cal_df[cal_df['FoodItem'] == predicted_class]

        if not calories_info.empty:
            calories_per_100g = calories_info['Cals_per100grams'].values[0]
            prediction_result = f"The predicted food item is {predicted_class} with {calories_per_100g} calories per 100 grams."
        else:
            prediction_result = f"Calories information not found for {predicted_class}."

    return render_template('index.html', prediction_result=prediction_result)

    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Function to predict image
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the expected input size for the model
    img_resized = cv2.resize(img, (224, 224))  # Change size if needed
    
    # Preprocess image
    img_array = image.img_to_array(img_resized)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Normalize the image
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    # Predict
    prediction = model.predict(img_array)[0][0]
    
    return prediction

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('signLanguageDataModel/sign_language_model.h5')  # Replace with your actual model path

# Define a function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 28x28
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize pixel values
    normalized = resized / 255.0
    # Reshape to (1, 28, 28, 1) for the model
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

# Define a function to get the label from the prediction
def get_label(prediction):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # Labels excluding J and Z
    return labels[np.argmax(prediction)]

# Path to the image to be identified
image_path = 'images/test 04.jpg'  # Replace with the path to your image

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make predictions
prediction = model.predict(preprocessed_image)
label = get_label(prediction)

# Display the result
print(f'Predicted label: {label}')

# Optionally, display the image with the predicted label
img = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted label: {label}')
plt.axis('off')
plt.show()

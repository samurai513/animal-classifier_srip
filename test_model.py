import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Define the custom activation function
def swish_activation(x):
    return tf.keras.backend.sigmoid(x) * x

# Register the custom activation function
tf.keras.utils.get_custom_objects().update({'swish_activation': swish_activation})

# Load the saved model
with tf.keras.utils.custom_object_scope({'swish_activation': swish_activation}):
    model = tf.keras.models.load_model('model.h5')

# Preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  # Assuming model input shape is (256, 256, 3)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to get class names from a text file
def get_class_names(file_path):
    class_names = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            class_names[idx] = line.strip()
    return class_names

# Make predictions on the preprocessed image
def predict_image(image_path, class_names):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    top_classes = np.argsort(predictions[0])[::-1][:10]  # Get indices of top 5 classes
    top_class_names = [class_names[idx] for idx in top_classes]
    top_probabilities = predictions[0][top_classes]
    return top_class_names, top_probabilities

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <image_path>")
    else:
        image_path = sys.argv[1]
        class_names = get_class_names('name.txt')  # Provide the path to your class names text file
        top_classes, top_probabilities = predict_image(image_path, class_names)

        # Plot the graph of probabilities
        plt.figure(figsize=(10, 6))
        plt.bar(top_classes, top_probabilities, color='blue')
        plt.xlabel('Classes')
        plt.ylabel('Probabilities')
        plt.title('Top 10 Predicted Classes and Probabilities')
        plt.xticks(rotation=45)
        plt.show()

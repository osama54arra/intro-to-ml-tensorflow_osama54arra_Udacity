import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import process_image

def predict(image_path, model, top_k=1):
    """
    Predicts the top K flower classes for an image using a trained model.
    
    Args:
        image_path (str): Path to the input image.
        model: Trained Keras model.
        top_k (int): Number of top classes to return.
    
    Returns:
        probs: NumPy array of top K probabilities.
        classes: List of top K class indices as strings.
    """
    # Load and process the image
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image = process_image(test_image)
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Get predictions
    predictions = model.predict(processed_image)
    
    # Get top K probabilities and indices
    top_indices = np.argpartition(predictions[0], -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(predictions[0][top_indices])[::-1]]
    top_probs = predictions[0][top_indices]
    
    classes = [str(idx) for idx in top_indices]
    probs = top_probs
    
    return probs, classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower species from an image using a trained model.')
    parser.add_argument('image', type=str, help='Path to the input image.')
    parser.add_argument('model', type=str, help='Path to the saved Keras model.')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top classes to return (default: 1).')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping class indices to flower names.')
    
    args = parser.parse_args()
    
    # Load the model
    model = tf.keras.models.load_model(args.model)
    
    # Get predictions
    probs, classes = predict(args.image, model, args.top_k)
    
    # Load category names if provided
    class_names = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    
    # Print results
    print(f"Image: {args.image}")
    for i, (prob, cls) in enumerate(zip(probs, classes)):
        if class_names:
            flower_name = class_names[cls]
            print(f"{i+1}: {flower_name} (class {cls}): {prob:.4f}")
        else:
            print(f"{i+1}: class {cls}: {prob:.4f}")

if __name__ == "__main__":
    main()

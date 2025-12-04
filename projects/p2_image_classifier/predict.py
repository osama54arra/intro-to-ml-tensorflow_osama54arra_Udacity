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
    print(f"Arguments parsed: image={args.image}, model={args.model}, top_k={args.top_k}, category_names={args.category_names}", flush=True)
    
    # Load the model
    try:
        print("Loading model...", flush=True)
        model = tf.keras.models.load_model(args.model)
        print("Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return
    
    # Get predictions
    try:
        print("Getting predictions...", flush=True)
        probs, classes = predict(args.image, model, args.top_k)
        print(f"Predictions obtained: {len(probs)} classes", flush=True)
    except Exception as e:
        print(f"Error getting predictions: {e}", flush=True)
        return
    
    # Load category names if provided
    class_names = None
    if args.category_names:
        try:
            print("Loading category names...", flush=True)
            with open(args.category_names, 'r') as f:
                class_names = json.load(f)
            print("Category names loaded.", flush=True)
        except Exception as e:
            print(f"Error loading category names: {e}", flush=True)
            return
    
    # Print results
    print(f"Image: {args.image}", flush=True)
    for i, (prob, cls) in enumerate(zip(probs, classes)):
        if class_names:
            flower_name = class_names[cls]
            print(f"{i+1}: {flower_name} (class {cls}): {prob:.4f}", flush=True)
        else:
            print(f"{i+1}: class {cls}: {prob:.4f}", flush=True)
    print("Prediction complete.", flush=True)

if __name__ == "__main__":
    main()

import ultralytics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
ultralytics.checks()

from ultralytics import YOLO

def select_model(model: str):
    """
    Select a model for object detection.

    Args:
        model (str): The model to use for object detection.

    Returns:
        YOLO: The YOLO model.
    """
    assert isinstance(model, str), "Model must be a string."
    assert model in ["n", "m", "l", "x"], "Model must be one of 'n', 'm', 'l', 'x'."

    model_name = f"yolo11{model}-obb.pt"
    model = YOLO(model_name)

    return model

def preload_model(model: str):
    """
    Preload a model for object detection.

    Args:
        model (str): The model to preload.

    Returns:
        YOLO: The YOLO model.
    """
    return select_model(model)

def process_image(image: Image, model: str):
    """
    Process an image using a CV model.

    Args:
        image (Image from PIL): The image to process.
        model (str): The model to use for processing (n, m, l, x) for different sizes.

    Returns:
        Direct output from the model.
    """
    assert isinstance(image, Image.Image), "Image must be a PIL image."

    # Load a pretrained model
    model = select_model(model)

    # Run inference on the image
    results = model(image)

    return results, results[0].names

def filter_predictions(predictions, probs_preds, xywhrs, threshold=None, class_index=None):
    """
    Filters predictions based on confidence threshold and class index.

    Args:
        predictions (np.ndarray): Array of class predictions.
        probs_preds (np.ndarray): Array of confidence scores.
        xywhrs (np.ndarray): Array of bounding box parameters.
        threshold (float, optional): Confidence threshold. Defaults to None.
        class_index (List(int), optional): Class index to filter by. Defaults to None.

    Returns:
        np.ndarray: Filtered predictions.
        np.ndarray: Filtered confidence scores.
        np.ndarray: Filtered bounding box parameters.
    """
    mask = np.ones_like(probs_preds, dtype=bool)

    if threshold is not None:
        mask &= probs_preds > threshold

    if class_index is not None:
        mask &= np.isin(predictions, class_index)            

    filtered_predictions = predictions[mask]
    filtered_probs_preds = probs_preds[mask]
    filtered_xywhrs = xywhrs[mask]

    return filtered_predictions, filtered_probs_preds, filtered_xywhrs


def draw_bounding_boxes(image, predictions, probs_preds, xywhrs, labels):
    """
    Draws bounding boxes and labels on the image.

    Args:
        image (np.ndarray): The original image.
        predictions (np.ndarray): Array of class predictions.
        probs_preds (np.ndarray): Array of confidence scores.
        xywhrs (np.ndarray): Array of bounding box parameters.
        labels (list): List of class names.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    for cls, conf, xywhr in zip(predictions, probs_preds, xywhrs):
        # Extract box parameters
        x_center, y_center, width, height, angle = xywhr

        # Convert to int for drawing
        x_center_int, y_center_int = int(x_center), int(y_center)
        width_int, height_int = int(width), int(height)
        angle_degrees = angle * 180 / np.pi  # Convert angle to degrees

        # Define the rectangle box
        rect = ((x_center, y_center), (width, height), angle_degrees)
        box = cv2.boxPoints(rect)
        
        # Convert box to int (numpy doesn't support int0)
        box = np.array(box, dtype=int)

        # Draw the bounding box
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Put the class label and confidence score
        label_text = f"{labels[int(cls)]}: {conf:.2f}"
        cv2.putText(image, label_text, (x_center_int, y_center_int - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def calculate_diameters(xywhrs, resolution):
    """
    Calculates diameters and their uncertainty from bounding boxes.

    Args:
        xywhrs (np.ndarray): Array of bounding box parameters.
        resolution (float): Resolution in meters per pixel.

    Returns:
        np.ndarray: Array of diameters.
        float: Standard deviation of diameters (uncertainty).
    """
    wh = xywhrs[:, 2:4]
    diameters = np.mean(wh, axis=1) * resolution
    diameter_uncertainty = np.std(wh, axis=1) * resolution
    return diameters, diameter_uncertainty

def post_process_image(results, resolution: float, threshold: float = None, class_index: int = [0, 1, 2, 6, 7, 8, 9, 10, 11]):
    """
    Post-processes the image to get object count and fuel tank diameters.

    Args:
        results: The direct output from the model.
        resolution (float): Resolution in meters per pixel.
        threshold (float, optional): Confidence threshold. Defaults to None.
        class_index (List(int), optional): Class index to filter by. Defaults to (planes, ships, storage tanks, ground track field, small vehicles, large vehicles, helicopters).

    Returns:
        dict: Dictionary containing object count, diameters, and uncertainty.
        np.ndarray: Array of diameters.
        float: Diameter uncertainty.
        np.ndarray: Image with drawn bounding boxes.
    """
    # Get the first result (assuming you have one image)
    result = results[0]

    # Get the original image
    image = result.orig_img.copy()

    # Get the predictions
    predictions = result.obb.cls.detach().cpu().numpy()
    probs_preds = result.obb.conf.detach().cpu().numpy()
    xywhrs = result.obb.xywhr.detach().cpu().numpy()
    labels = result.names

    # Filter predictions
    predictions, probs_preds, xywhrs = filter_predictions(
        predictions, probs_preds, xywhrs, threshold, class_index
    )

    # Count per class
    unique, counts = np.unique(predictions, return_counts=True)
    total_count = len(predictions)
    object_counts = dict(zip([labels[i] for i in unique], counts))

    # Draw bounding boxes on the image
    image = draw_bounding_boxes(image, predictions, probs_preds, xywhrs, labels)

    # Calculate diameters and uncertainty
    diameters, diameter_uncertainty = calculate_diameters(xywhrs, resolution) if resolution else (None, None)

    # Prepare the result dictionary
    result_dict = {
        'total_count': total_count,
        'object_counts': object_counts,
        'diameters': diameters    
    }

    return result_dict, image


def main():
    # Preload model (if needed)
    model = preload_model("x")

    # Load an image
    img = Image.open("backend/Test Images/boats.jpg")

    # Run the model
    results, labels = process_image(img, "x")
    result_dict, image = post_process_image(results, resolution=0.1, threshold=0.2)

    print(result_dict)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
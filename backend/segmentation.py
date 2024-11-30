import ultralytics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

def process_image(image: Image, model: str, resolution: float = None, threshold: float = None):
    """
    Process an image using a CV model.

    Args:
        image (Image from PIL): The image to process.
        model (str): The model to use for processing (n, m, l, x) for different sizes.
        resolution (float, optional): The resolution of the image in meters per pixels. Defaults to None.
        threshold (float, optional): The threshold for object detection. Defaults to None.

    Returns:
        list: A list of detected objects.
    """
    assert isinstance(image, Image.Image), "Image must be a PIL image."
    assert resolution is None or isinstance(resolution, float), "Resolution must be a float."
    assert threshold is None or isinstance(threshold, float), "Threshold must be a float."

    # Load a pretrained model
    model = select_model(model)

    # Run inference on the image
    results = model(image)

    # Segmented image
    segmented_image = results[0].plot()

    # Get the predictions
    predictions = results[0].obb.cls.detach().cpu().numpy()
    probs_preds = results[0].obb.conf.detach().cpu().numpy()
    xywhrs = results[0].obb.xywhr.detach().cpu().numpy()
    labels = results[0].names

    # Filter predictions
    if threshold is not None:
        predictions = predictions[probs_preds > threshold]
        xywhrs = xywhrs[probs_preds > threshold]
        probs_preds = probs_preds[probs_preds > threshold]

    # const label of the fuel tank
    fuel_tank_label = 2

    # value counts
    unique, counts = np.unique(predictions, return_counts=True)
    object_count = dict(zip([labels[i] for i in unique], counts))

    # Get the tanks width and height
    tanks_pos = xywhrs[predictions == fuel_tank_label]

    # Get the tanks diameter
    ft_diameters_pixels = np.mean(tanks_pos[:, 2:4], axis=1)
    ft_diameters_uncertainty_pixels = np.std(tanks_pos[:, 2:4], axis=1)

    # Scale to meters
    ft_diameters_meters = ft_diameters_pixels * resolution
    ft_diameters_uncertainty_meters = ft_diameters_uncertainty_pixels * resolution

    return object_count, ft_diameters_meters, ft_diameters_uncertainty_meters, segmented_image

img = Image.open("backend/Test Images/boats.jpg")
print(process_image(img, "x", 0.1, 0.5))
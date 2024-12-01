import numpy as np
import torch
from PIL import Image

from unet import UNet

import backend.utils.predict as predict
import gc

def segment_land(image, scale, threshold, resolution, model_path):
    """
    Segments the land in the image using the UNet model.

    Args:
        image (Image from PIL): The image to segment.
        scale (int): The scale factor.
        threshold (float): The threshold for the segmentation.
        resolution (float): The resolution of the image in meters per pixel.
        model_path (str): The path to the model.

    Returns:
        dict: The areas of each class.
        Image from PIL: The segmented image.
    """

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=3, n_classes=7)

    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    seg, mask_indices = predict.predict_img(net=net,
                           full_img=image,
                           scale_factor=scale,
                           out_threshold=threshold,
                           device=device)
    
    mask_indices = mask_indices.cpu().numpy()

    segmented_image = Image.fromarray(seg)

    # Rescale the segmented image to the original size
    segmented_image = segmented_image.resize((image.size[0], image.size[1]))

    # Labels outputed by the model
    labels_map = {0: 'urban land', 1: 'agriculture', 2: 'rangeland', 3: 'forest land', 4: 'water', 5: 'barren land', 6: 'unknown'}

    return_dict = {}
    for i in range(7):
        area = np.sum(mask_indices == i) * resolution
        ratio_area = area / (image.size[0] * image.size[1])
        return_dict[labels_map[i]] = {'area': area, 'ratio_area': ratio_area}

    return return_dict, segmented_image
        
def main():
    # Example usage
    image_path = 'data/test_set_full_set/img_test/1.png'
    image = Image.open(image_path)
    scale = 0.2
    threshold = 0.5
    resolution = 0.5
    model_path = 'weights/CP_epoch30.pth'
    areas, segmented_image = segment_land(image, scale, threshold, resolution, model_path)
    print(areas)
    segmented_image.show()

if __name__ == "__main__":
    main()
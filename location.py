
import os
from dataclasses import dataclass
from PIL import Image
import random
TILE_SAVE_DIR = 'tiles/'
import pickle

class Location:
    name: str
    top_left: tuple[float, float]
    bottom_right: tuple[float, float]

    tiles: dict[tuple[int, int], Image.Image]
    examples: list[tuple[int, int]]
    meters_per_pixel: float

    def __init__(self, name: str, top_left: tuple[float, float], bottom_right: tuple[float, float], zoom: int, tile_size_meters: int):
        self.name = name
        self.tiles = {}
        self.examples = []
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.tile_size_meters = tile_size_meters
        self.zoom = zoom
        self.meters_per_pixel = image_scale(zoom, top_left)


    def fetch(self, zoom=10):
        self.fetch_tiles()
        self.fetch_thumbnail(zoom)
    def fetch_tiles(self):
        current = calculate_new_coordinates(self.top_left, self.tile_size_meters, self.tile_size_meters)
        first_long = current[1]
        while current[0] > self.bottom_right[0]:
            while current[1] < self.bottom_right[1]:
                current = calculate_new_coordinates(current, 0, self.tile_size_meters)
                # download image
                img = fetch_image(current, calculate_new_coordinates(current, self.tile_size_meters, self.tile_size_meters), self.zoom)
                self.tiles[current] = img
            current = current[0], first_long
            current = calculate_new_coordinates(current, self.tile_size_meters, 0)
        
    def load(self):
        self.tiles = {}
        # load all tiles from images
        for file in os.listdir(TILE_SAVE_DIR):
            key = tuple(map(int, file[:-4].split('_')))
            self.tiles[key] = Image.open(os.path.join(TILE_SAVE_DIR, self.name, file))
    def get_example_images(self):
        return [self.tiles[example] for example in self.examples]
    
    def fetch_thumbnail(self, zoom = 10):
        self.thumbnail = fetch_image(self.top_left, self.bottom_right, zoom)

    def get_thumbnail(self):
        return self.tiles[self.examples[0]]
    def get_middle_tile(self):
        mid = self.top_left[0] + (self.bottom_right[0] - self.top_left[0]) / 2, self.top_left[1] + (self.bottom_right[1] - self.top_left[1]) / 2
        img = fetch_image(mid, calculate_new_coordinates(mid, self.tile_size_meters, self.tile_size_meters), self.zoom)
        return img
    def get_real_size(self):
        return calculate_distance_components(self.top_left, self.bottom_right)

    def save_class(self):
        # Create directories if they don't exist
        class_dir = os.path.join(TILE_SAVE_DIR, self.name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'examples': self.examples,
            'tile_size_meters': self.tile_size_meters,
            'zoom': self.zoom,
            'meters_per_pixel': self.meters_per_pixel,
        }
        with open(os.path.join(class_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save images
        tiles_dir = os.path.join(class_dir, 'tiles')
        os.makedirs(tiles_dir, exist_ok=True)
        for key, img in self.tiles.items():
            # Use floating-point format in filenames
            key_str = f"{key[0]:.6f}_{key[1]:.6f}"
            img.save(os.path.join(tiles_dir, f"{key_str}.png"))
        # Save thumbnail
        self.thumbnail.save(os.path.join(class_dir, 'thumbnail.png'))

    @classmethod
    def load_class(cls, name):
        class_dir = os.path.join(TILE_SAVE_DIR, name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"No saved data found for location: {name}")
        
        # Load metadata
        with open(os.path.join(class_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        # Load thumbnail
        
        # Create instance
        instance = cls(
            name=metadata['name'],
            top_left=metadata['top_left'],
            bottom_right=metadata['bottom_right'],
            zoom=metadata['zoom'],
            tile_size_meters=metadata['tile_size_meters']
        )
        instance.examples = metadata['examples']
        instance.meters_per_pixel = metadata['meters_per_pixel']
        instance.thumbnail = Image.open(os.path.join(class_dir, 'thumbnail.png'))
        
        # Load images
        tiles_dir = os.path.join(class_dir, 'tiles')
        instance.tiles = {}
        for file in os.listdir(tiles_dir):
            if file.endswith('.png'):
                # Parse floating-point keys
                key = tuple(map(float, file[:-4].split('_')))
                instance.tiles[key] = Image.open(os.path.join(tiles_dir, file))
        
        return instance
def load_all_locations():
    locations = []
    for name in os.listdir(TILE_SAVE_DIR):
        locations.append(Location.load_class(name))
    return locations



import os
import json
import re
import cv2
from PIL import Image
from datetime import datetime

from sat_down.image_downloading import download_image


default_prefs = {
    'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    'tile_size': 256,
    'channels': 3,
    # 'dir': os.path.join(file_dir, 'all_tiles'),
    'headers': {
        'cache-control': 'max-age=0',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
    },
    # 'tl': '',
    # 'br': '',
    # 'zoom': ''
}
# scale in m/pix
zoom_to_scale = {
    1: 78271.52,
    2: 39135.76,
    3: 19567.88,
    4: 9783.94,
    5: 4891.97,
    6: 2445.98,
    7: 1222.99,
    8: 611.5,
    9: 305.75,
    10: 152.87,
    11: 76.44,
    12: 38.22,
    13: 19.11,
    14: 9.55,
    15: 4.78,
    16: 2.39,
    17: 1.19,
    18: 0.6,
    19: 0.3
}




def take_input(messages):
    inputs = []
    print('Enter "r" to reset or "q" to exit.')
    for message in messages:
        inp = input(message)
        if inp == 'q' or inp == 'Q':
            return None
        if inp == 'r' or inp == 'R':
            return take_input(messages)
        inputs.append(inp)
    return inputs


def fetch_image(top_left, bottom_right, zoom):
    prefs = default_prefs.copy()

    lat1, lon1 = top_left
    lat2, lon2 = bottom_right

    if zoom not in zoom_to_scale:
        raise ValueError('Invalid zoom level ' + zoom) 


    lat1, lon1 = top_left
    lat2, lon2 = bottom_right

    channels = int(prefs['channels'])
    tile_size = int(prefs['tile_size'])
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)

    img = download_image(lat1, lon1, lat2, lon2, zoom, prefs['url'],
        prefs['headers'], tile_size, channels)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f'img_{timestamp}.png'
    return Image.fromarray(img)

vortex_tl = (46.525163, 6.574005)

from math import radians, degrees, sin, cos, atan2, sqrt
def calculate_new_coordinates(lat_lon, meters_south, meters_east):
    R = 6378137
    lat, lon = map(radians, lat_lon)
    new_lat = lat - meters_south / R
    new_lon = lon + meters_east / (R * cos(lat))
    return (degrees(new_lat), degrees(new_lon))


from math import radians, sin, cos, sqrt, atan2

def calculate_distance_components(top_left, bottom_right):
    R = 6378137
    lat1, lon1 = map(radians, top_left)
    lat2, lon2 = map(radians, bottom_right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    north_south_distance = dlat * R
    east_west_distance = dlon * R * cos((lat1 + lat2) / 2)
    
    return (north_south_distance, east_west_distance)

# Example usage
top_left_coords = (48.8566, 2.3522)
bottom_right_coords = (48.853, 2.35)
print(calculate_distance_components(top_left_coords, bottom_right_coords))


def image_scale(zoom, loc):
    lat, _ = loc
    return zoom_to_scale[zoom] * cos(radians(lat))



import os
from dataclasses import dataclass
from PIL import Image
import random
TILE_SAVE_DIR = 'tiles/'

@dataclass
class Location:
    name: str
    tiles: dict[tuple[int, int], Image.Image]
    examples: list[tuple[int, int]]
    meters_per_pixel: float

    def save(self):
        # if the dir exists, ask for permissoin and delete contents
        if os.path.exists(TILE_SAVE_DIR):
            if input(f"Delete contents of {TILE_SAVE_DIR}? (y/n) ") == 'y':
                for file in os.listdir(TILE_SAVE_DIR):
                    os.remove(f"{TILE_SAVE_DIR}{file}")
        else:
            os.makedirs(TILE_SAVE_DIR)
        # save all tiles as images
        for key, tile in self.tiles.items():
            tile.save(f"{TILE_SAVE_DIR}{key[0]}_{key[1]}.png")
    def load(self):
        self.tiles = {}
        # load all tiles from images
        for file in os.listdir(TILE_SAVE_DIR):
            key = tuple(map(int, file[:-4].split('_')))
            self.tiles[key] = Image.open(f"{TILE_SAVE_DIR}{file}")
    def get_example_images(self):
        return [self.tiles[example] for example in self.examples]
    
    def add_more_examples_images(self):
        random.seed(42)
        self.examples += random.sample(list(self.tiles.keys()), 3)
    
    def create_example_images(self):
        self.examples = random.sample(list(self.tiles.keys()), 3)
    
    def get_thumbnail(self):
        return self.tiles[self.examples[0]]

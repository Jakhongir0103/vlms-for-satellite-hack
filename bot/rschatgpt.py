import os
import argparse
import re
from langchain_openai import ChatOpenAI
from PIL import Image
import ast
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()

from backend.segmentation import process_image, post_process_image, preload_model

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.makedirs('image_out', exist_ok=True)

class UnifiedVisionModel:
    def __init__(self, llm):
        self.llm = llm
    
    def image_to_base64(self, image_path):
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_text(self, text_prompt, system_message=None):
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ]
            }
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def analyze_image(self, image_path, prompt=None, system_message=None):
        base64_image = self.image_to_base64(image_path)
        
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This is a satellite image. {prompt if prompt else ''}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self.llm.invoke(messages)
        return response.content

class RSChatGPT:
    def __init__(self, gpt_name, cv_model_type='x'):
        self.llm = ChatOpenAI(
            model=gpt_name,
            max_tokens=1000,
            temperature=0
        )
        self.vision_model = UnifiedVisionModel(self.llm)
        self.cv_model = preload_model(cv_model_type)
        self.system_message = None
        self.object_names = None

    def initialize(self, image_path, question):
        """
        Initialize the system message and object names.
        """
        prompt="Describe what you see in this satellite image very shortly."
        description = self.vision_model.analyze_image(image_path, prompt)

        self.system_message = f"""You are an image analysis assistant. You are given a satellite image, its description, number of objects, and their properties.

Desciption of the image:
{description}

Answer the user's question based on the image, its description, and the objects properties.
"""

        prompt = f"""Among these following objects ['plane', 'ship', 'storage tank', 'ground track field', 'large vehicle', 'small vehicle', 'helicopter'], which ones is the question about?
Note that 'large vehicle' and 'small vehicle' are cars.
Return only a python list of object names, e.g. ['helicopters', 'ground track field']. If the question is not about any of the objects, return an empty list.

Question:
{question}

A list of object names:
"""
        object_names = self.vision_model.analyze_text(prompt)
        object_names = ast.literal_eval(object_names)

        self.object_names = object_names

    def segment_image(self, image_in_path, threshold, resolution=0.1):
        image_in = Image.open(image_in_path)
        result = process_image(image=image_in, model=self.cv_model)
        id_to_label = result.names
        label_to_id = {label: id for id, label in id_to_label.items()}
        result_list, image_out = post_process_image(result, resolution=resolution, threshold=threshold, class_index=[label_to_id[name] for name in self.object_names])

        image_out_path = image_in_path.replace("image_in", "image_out").replace(".png", "_segmented.png")
        Image.fromarray(image_out).save(image_out_path)

        return result_list

    def extract_output(self, text):
        match = re.search(r'\[\[(.*?)\]\]', text)
        if match:
            return match.group(1)
        else:
            return 0

    def run_image(self, image_path, question, result_list):
        """
        Run the image and the result list through the model.
        """
        prompt = f"""Answer the question about the image based on the given objects' properties.
Note that 'large vehicle' and 'small vehicle' are cars. Output only a single number as your output between double square brackets as, e.g [[6]].

Number of objects:
{len(result_list)}

Objects properties:
{chr(10).join([f"{obj['class']}: diameter = {obj['diameter']:.2f} meters, width = {obj['width']:.2f} meters, height = {obj['height']:.2f} meters" for obj in result_list])}

Question:
{question}
"""

        print("Prompt:\n", prompt)
        
        # response = self.vision_model.analyze_image(image_path, prompt, system_message=self.system_message)
        response = self.vision_model.analyze_text(prompt)

        return self.extract_output(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--image_tiles_dir', type=str)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()
    
    bot = RSChatGPT(gpt_name="gpt-4o-2024-11-20")
    bot.initialize(args.image_path, args.question)

    # # All tiles
    # tiles_names = list(os.listdir(args.image_tiles_dir))
    # results_list = []

    # for tile_name in tqdm(tiles_names, desc="Segmenting the tiles"):
    #     tile_path = os.path.join(args.image_tiles_dir, tile_name)
    #     results_list.extend(bot.segment_image(tile_path, args.threshold))
    
    # response = bot.run_image(args.image_path, args.question, results_list)
    # print("Response:\n", response)

    # All tiles
    tiles_names = list(os.listdir(args.image_tiles_dir))

    tile_sample = tiles_names[0]
    tile_path = os.path.join(args.image_tiles_dir, tile_sample)
    result_list = bot.segment_image(tile_path, args.threshold)
    
    response = bot.run_image(args.image_path, args.question, result_list)
    print("Response:\n", response)
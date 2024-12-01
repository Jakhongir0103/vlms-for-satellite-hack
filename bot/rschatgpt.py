import argparse
import ast
import os
import re

from langchain_openai import ChatOpenAI
from PIL import Image

import dotenv
dotenv.load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.segmentation import process_image, post_process_image, preload_model

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.makedirs('image_out', exist_ok=True)

class UnifiedVisionModel:
    def __init__(self, llm):
        self.llm = llm
    
    def image_to_base64(self, image):
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

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

    def analyze_image(self, image, prompt=None, system_message=None):
        base64_image = self.image_to_base64(image)
        
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else "You are a helpful visual assistant. If the provided context does not contain all the necessary information, analyze the image and answer the question yourself."
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

    def initialize(self, question):
        """
        Initialize the object names.
        """
        prompt = f"""Among these following objects ['plane', 'ship', 'storage tank', 'ground track field', 'large vehicle', 'small vehicle', 'helicopter'], which ones is the question about?
Note that 'large vehicle' and 'small vehicle' are cars.
Return only a list of object names, e.g. ['helicopters', 'ground track field']. If the question is not about any of the objects, return an empty list.

Question:
{question}

A list of object names:
"""
        object_names = self.vision_model.analyze_text(prompt)
        object_names = ast.literal_eval(object_names)

        print("Object names:\n", object_names)

        self.object_names = object_names

    def segment_image(self, image_in, threshold, resolution=0.1):
        result = process_image(image=image_in, model=self.cv_model)
        id_to_label = result.names
        label_to_id = {label: id for id, label in id_to_label.items()}
        result_list, image_out = post_process_image(result, resolution=resolution, threshold=threshold, class_index=[label_to_id[name] for name in self.object_names])

        image_out = Image.fromarray(image_out)

        return result_list, image_out

    def extract_output(self, text):
        match = re.search(r'\[\[(.*?)\]\]', text)
        if match:
            return match.group(1)
        else:
            return 0

    def run_text(self, question, result_list):
        """
        Run the image and the result list through the model.
        """
        obj_count = {class_name: sum(1 for obj in result_list if obj['class'] == class_name) for class_name in self.object_names}
        prompt = f"""Answer the question about an image based on the given objects' properties.
Note that 'large vehicle' and 'small vehicle' are cars. Output only a single number as your output between double square brackets as, e.g [[6]].

Number of objects:
{chr(10).join([f"{class_name}: {count}" for class_name, count in obj_count.items()])}

Objects properties:
{chr(10).join([f"{obj['class']}: diameter = {obj['diameter']:.2f} meters, width = {obj['width']:.2f} meters, height = {obj['height']:.2f} meters" for obj in result_list])}

Question:
{question}
"""

        print("Prompt:\n", prompt)

        response = self.vision_model.analyze_text(prompt)
        return self.extract_output(response)

    def run_image(self, image, question, result_list):
        """
        Run the image and the result list through the model.
        """
        obj_count = {class_name: sum(1 for obj in result_list if obj['class'] == class_name) for class_name in self.object_names}
        prompt = f"""Answer the question about the image based on the given objects' properties.
Note that 'large vehicle' and 'small vehicle' are cars. Output only a single number as your output between double square brackets as, e.g [[6]].

Number of objects:
{chr(10).join([f"{class_name}: {count}" for class_name, count in obj_count.items()])}

Objects properties:
{chr(10).join([f"{obj['class']}: diameter = {obj['diameter']:.2f} meters, width = {obj['width']:.2f} meters, height = {obj['height']:.2f} meters" for obj in result_list])}

Question:
{question}
"""

        print("Prompt:\n", prompt)

        response = self.vision_model.analyze_image(image, prompt)
        return self.extract_output(response)

def main(image: Image.Image, question: str, threshold: float, resolution: float ):
    bot = RSChatGPT(gpt_name="gpt-4o-mini")
    bot.initialize(question)

    result_list, image_out = bot.segment_image(image, threshold, resolution)
    response = bot.run_image(image, question, result_list)

    return response, image_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()

    bot = RSChatGPT(gpt_name="gpt-4o-mini")
    bot.initialize(args.question)

    image = Image.open(args.image_path)

    result_list, image_out = bot.segment_image(image, args.threshold)
    response = bot.run_image(image, args.question, result_list)

    print("Response:\n", response)
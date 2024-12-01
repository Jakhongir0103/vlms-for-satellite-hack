import argparse
import ast
import os
import re

from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.segmentation import process_image, post_process_image, preload_model
from backend.predict_field_segmentation import segment_land, preload_model as preload_field_segmentation_model
import ollama

os.makedirs('image_out', exist_ok=True)

class LLMModel:
    def __init__(self):
        self.client = ollama.Client(host='http://localhost:11434')
    
    def analyze_text(self, prompt):
        if isinstance(prompt, dict):
            messages = prompt.get('content') if 'content' in prompt else str(prompt)
        else:
            messages = prompt
        
        response = self.client.generate(model='llama2', prompt=messages)
        return response['response']

class RSChatGPT:
    def __init__(self, cv_model_type='x', field_segmentation_model_path='../backend/weights/CP_epoch30.pth'):
        self.llm_model = LLMModel()
        self.cv_model = preload_model(cv_model_type)
        self.field_net = preload_field_segmentation_model(field_segmentation_model_path)
        self.object_names = ['plane', 'ship', 'storage tank', 'ground track field', 'large vehicle', 'small vehicle', 'helicopter']
        self.field_names = ['urban land', 'agriculture', 'rangeland', 'forest land', 'water', 'barren land']
        self.system_message = None
        self.detected_entities = None
        self.question_type = None

    def initialize(self, question):
        """
        Initialize the object names.
        """
        prompt = f"""Given the question, which ones of the following entities is the question about?
Note that 'large vehicle' and 'small vehicle' are cars.
Return only a list of object names, e.g. ['helicopters', 'ground track field']. If the question is not about any of the entities, return an empty list.

Entities: 
{self.object_names + self.field_names}

Question:
{question}

A list of entity names:
"""
        detected_entities = self.llm_model.analyze_text(prompt)
        match = re.search(r'\[(.*?)\]', detected_entities)
        if match:
            detected_entities = '[' + match.group(1) + ']'
        else:
            detected_entities = '[]'
        detected_entities = ast.literal_eval(detected_entities)

        if [entity for entity in detected_entities if entity in self.object_names]:
            self.question_type = 'object'
        elif [entity for entity in detected_entities if entity in self.field_names]:
            self.question_type = 'field'
        else:
            self.question_type = None

        print("Detected entities:\n", detected_entities)

        self.detected_entities = detected_entities

    def segment_object(self, image_in, threshold=0.5, resolution=0.1):
        result = process_image(image=image_in, model=self.cv_model)
        id_to_label = result.names
        label_to_id = {label: id for id, label in id_to_label.items()}

        result_list, image_out = post_process_image(result, resolution=resolution, threshold=threshold, class_index=[label_to_id[name] for name in self.object_names])

        image_out = Image.fromarray(image_out)

        return result_list, image_out

    def segment_field(self, image_in, threshold=0.5, resolution=0.1, scale=0.2):
        areas, image_out = segment_land(image=image_in, threshold=threshold, resolution=resolution, scale=scale, net=self.field_net)
        del areas['unknown']

        return areas, image_out

    def extract_output(self, text):
        match = re.search(r'\[\[(.*?)\]\]', text)
        if match:
            return match.group(1)
        else:
            return 0

    def run_text(self, question, results):
        """
        Run the image and the result list through the model.
        """
        if self.question_type == 'object':
            obj_count = {class_name: sum(1 for obj in results if obj['class'] == class_name) for class_name in self.object_names}
            prompt = f"""Answer the question about an image based on the given objects' properties.
Note that 'large vehicle' and 'small vehicle' are cars. Output only a single number as your output between double square brackets as, e.g [[6]].

Number of objects:
{chr(10).join([f"{class_name}: {count}" for class_name, count in obj_count.items()])}

Objects properties:
{chr(10).join([f"{obj['class']}: diameter = {obj['diameter']:.2f} meters, width = {obj['width']:.2f} meters, height = {obj['height']:.2f} meters" for obj in results])}

Question:
{question}
"""
        elif self.question_type == 'field':
            prompt = f"""Answer the question about an image based on the given fields' areas.
Output only a single number as your output between double square brackets as, e.g [[6]].

Area of each type of field:
{chr(10).join([f"{field}: area = {areas['area']:.2f} square meters, area ratio = {areas['ratio_area']:.4f}" for field, areas in results.items()])}

Question:
{question}
"""
        else:
            prompt = f"""Answer the question about an image.
Output only a single number as your output between double square brackets as, e.g [[6]].

Question:
{question}
"""

        print("Prompt:\n", prompt)

        response = self.llm_model.analyze_text(prompt)
        return self.extract_output(response)

def initilize_bot(question: str):
    bot = RSChatGPT()
    bot.initialize(question)

    return bot

def analyze_image(bot, image, question, threshold, resolution):
    if bot.question_type == 'object':
        results, image_out = bot.segment_object(image, threshold, resolution)
    elif bot.question_type == 'field':
        results, image_out = bot.segment_field(image, threshold, resolution)
    else:
        results = None
        image_out = image

    response = bot.run_text(question, results)

    return response, image_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--resolution', type=float, required=True)
    args = parser.parse_args()

    bot = initilize_bot(args.question)

    image = Image.open(args.image_path)
    response, image_out = analyze_image(bot, image, args.question, args.threshold, args.resolution)

    image_out.show()
    print("Response:\n", response)
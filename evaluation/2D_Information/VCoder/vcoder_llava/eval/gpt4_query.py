import base64
import requests
import os
import argparse
from vcoder_llava.questions import QUESTIONS
import random
import glob
from tqdm import tqdm
import time

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def query_gpt4(image_path):
  # Getting the base64 string
  base64_image = encode_image(image_path)
  ques = "What entities can be seen in the image? Your answer should be in the format: 'The objects present in the image are: ...' and then just list the objects with their counts (in words) before them in paragraph format." \
          "For example if there are 14 people, two dogs, and three chairs in an image, you should respond: The objects present in are: fourteen people, two dogs, three chairs."

  payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": ques,
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return response.json()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image-folder", type=str, default="")
  parser.add_argument("--output-file", type=str, default="output")
  args = parser.parse_args()

  if os.path.exists("done_ims.txt"):
    with open("done_ims.txt", 'r') as f:
      ims = f.readlines()
  else:
     ims = []
  done_ims = [i.strip("\n") for i in ims]
  print(done_ims)
  
  os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
  images = glob.glob(os.path.join(args.image_folder, "*.jpg"))
  error_imgs = []
  
  for image in tqdm(images, total=len(images)):
    skip = False
    fail = True
    if image in done_ims:
      continue
    print("Running image %s" % image)
    while fail:
      try:
        answer = query_gpt4(image)
        answer = answer["choices"][0]["message"]["content"]
        with open(f'done_ims.txt', 'a') as f:
          f.write(f'{image}\n')
        fail = False
      except:
        fail = True
        print(answer)
        if answer['error']['message'] == "Your input image may contain content that is not allowed by our safety system.":
            break
            skip = True
        else:
            time.sleep(900)
    if skip:
        continue
    with open(f'{args.output_file}', 'a') as f:
        f.write(f'Image: {image.split("/")[-1]}\n')
        f.write(f'<<ANSWER>>: {answer}\n')
        f.write('-------------------------------------------------------\n')
        
  
  print(f"Error images: {error_imgs}")

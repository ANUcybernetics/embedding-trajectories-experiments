import requests
import json
import time
import os

# Set your Replicate API token
# os.environ["REPLICATE_API_TOKEN"] = "your_api_token_here"

# latest versions as at 2024-08-13
# flux schnell
IMAGE_MODEL_VERSION="f2ab8a5bfe79f02f0789a146cf5e73d2a4ff2684a98c2b303d1e1ff3814271db"
# flux dev
# IMAGE_MODEL_VERSION="2d27e06ff0ee2494eb3ae8e3da0be637763f59bf6a488b32b2f89cd928c3faba"

# actually this is now BLIP
CAPTION_MODEL_VERSION="2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"
# LLAVA
# CAPTION_MODEL_VERSION="80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"

# ImageBind
EMBEDDING_MODEL_VERSION="0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304"

def latest_model_version(model_name):
    url = f"https://api.replicate.com/v1/models/{model_name}"
    headers = {
        "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['latest_version']['id']
    else:
        raise Exception(f"Failed to fetch latest version: {response.status_code}")

def generate_image(prompt):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json"
    }
    data = {
        "version": IMAGE_MODEL_VERSION,
        "input": {"prompt": prompt, "output_format": "jpg"},

    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    while result['status'] != 'succeeded':
        time.sleep(1)
        response = requests.get(f"{url}/{result['id']}", headers=headers)
        result = response.json()

    return result['output'][0]

def caption_image(image_url):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json"
    }
    data = {
        "version": CAPTION_MODEL_VERSION,
        "input": {"image": image_url, "prompt": "Describe this image in detail"}
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    while result['status'] != 'succeeded':
        time.sleep(1)
        response = requests.get(f"{url}/{result['id']}", headers=headers)
        result = response.json()

    return "".join(result['output'])

def calculate_embedding(type, text_or_url):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json"
    }
    data = {
        "version": EMBEDDING_MODEL_VERSION,
        "input": {
            "modality": type,
            "text_input" if type == "text" else "input": text_or_url
        }
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    while result['status'] != 'succeeded':
        time.sleep(1)
        response = requests.get(f"{url}/{result['id']}", headers=headers)
        result = response.json()

    embedding = result['output']
    return embedding

def main():
    prompt = input("Enter initial prompt: ")

    output_data = []

    for i in range(10):
        print(f"Iteration {i+1}")
        image_url = generate_image(prompt)
        print(f"Generated image: {image_url}")

        image_embedding = calculate_embedding("vision", image_url)
        output_data.append([i, "image"] + image_embedding)

        caption = caption_image(image_url)
        print(f"Image caption: {caption}")

        caption_embedding = calculate_embedding("text", caption)
        output_data.append([i, "text"] + caption_embedding)

        prompt = caption
        print("\n")

    with open("data/output.json", "w") as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()

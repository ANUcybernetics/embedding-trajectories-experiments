import datetime
import json
import os
import sys
import time

import requests

# get Replicate API token
try:
    REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]
except KeyError:
    print("Error: REPLICATE_API_TOKEN not found in environment variables.")
    print("Please set the REPLICATE_API_TOKEN environment variable and try again.")
    sys.exit(1)

# latest versions as at 2024-08-13
# flux schnell
IMAGE_MODEL_VERSION = "f2ab8a5bfe79f02f0789a146cf5e73d2a4ff2684a98c2b303d1e1ff3814271db"
# flux dev
# IMAGE_MODEL_VERSION="2d27e06ff0ee2494eb3ae8e3da0be637763f59bf6a488b32b2f89cd928c3faba"

# actually this is now BLIP
CAPTION_MODEL_VERSION = (
    "2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"
)
# LLAVA
# CAPTION_MODEL_VERSION="80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"

# ImageBind
EMBEDDING_MODEL_VERSION = (
    "0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304"
)


def latest_model_version(model_name):
    url = f"https://api.replicate.com/v1/models/{model_name}"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["latest_version"]["id"]
    else:
        raise Exception(f"Failed to fetch latest version: {response.status_code}")


def generate_image(prompt):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "version": IMAGE_MODEL_VERSION,
        "input": {"prompt": prompt, "output_format": "jpg"},
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    while result["status"] != "succeeded":
        time.sleep(1)
        response = requests.get(f"{url}/{result['id']}", headers=headers)
        result = response.json()

    return result["output"][0]


def caption_image(image_url):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "version": CAPTION_MODEL_VERSION,
        "input": {"image": image_url, "prompt": "Describe this image in detail"},
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    while result["status"] != "succeeded":
        time.sleep(1)
        response = requests.get(f"{url}/{result['id']}", headers=headers)
        result = response.json()

    return "".join(result["output"])


def calculate_embedding(type, text_or_url):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "version": EMBEDDING_MODEL_VERSION,
        "input": {
            "modality": type,
            "text_input" if type == "text" else "input": text_or_url,
        },
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    while result["status"] != "succeeded":
        time.sleep(1)
        response = requests.get(f"{url}/{result['id']}", headers=headers)
        result = response.json()

    embedding = result["output"]
    return embedding


def main():
    num_iterations = 100  # should be even
    prompt = input("Enter initial prompt: ")

    # Get the current timestamp in ISO8601 format
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    output_data = []

    # all the index maths assumes just 2 models in sequence
    # not necessary to make it more general at this stage
    for i in range(int(num_iterations / 2)):
        print(f"Sequence number {2*i}")
        image_url = generate_image(prompt)
        print(f"Generated image: {image_url}")

        image_embedding = calculate_embedding("vision", image_url)
        output_data.append(
            {
                "seq_no": 2 * i,
                "type": "image",
                "input": prompt,
                "embedding": image_embedding,
            }
        )

        print(f"Sequence number {2*i+1}")
        # BLIP prefixes the string with "Caption: " - remove this
        caption = caption_image(image_url)
        if caption.startswith("Caption: "):
            caption = caption[8:]
        print(f"Image caption: {caption}")

        caption_embedding = calculate_embedding("text", caption)
        output_data.append(
            {
                "seq_no": 2 * i + 1,
                "type": "text",
                "input": image_url,
                "embedding": caption_embedding,
            }
        )

        prompt = caption
        print("\n")

    # Read existing data from the file
    try:
        with open("data/output.json", "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    # Add new data to existing data
    existing_data[timestamp] = output_data

    # Write updated data back to the file
    with open("data/output.json", "w") as f:
        json.dump(existing_data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()

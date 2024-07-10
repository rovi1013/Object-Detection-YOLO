import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from PIL import Image
from io import BytesIO


def fetch_image_urls(query, max_images):
    image_urls = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    base_url = "https://unsplash.com/s/photos/"
    url = f"{base_url}{quote(query)}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    image_elements = soup.find_all("img", {"srcset": True})

    for img in image_elements:
        if len(image_urls) >= max_images:
            break
        image_url = img['src']
        image_urls.append(image_url)

    return image_urls


def filter_valid_images(image_urls, max_images):
    valid_images = []
    for url in image_urls:
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                if image.width >= 150 and image.height >= 150:
                    valid_images.append(url)
        except Exception as e:
            print(f"Error processing image from URL {url}: {e}")
        if len(valid_images) >= max_images:
            break
    return valid_images


def download_images(image_urls, folder_path, image_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, url in enumerate(image_urls):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(folder_path, f"{image_name}_{i + 1}.jpg")
            with open(file_path, 'wb') as f:
                for chunk in response:
                    f.write(chunk)


def main():
    search_queries = [
        "Wood Pigeon",
        "Rock Dove",
        "House Sparrow",
        "Eurasian Blackbird",
        "European Robin",
        "Carrion Crow",
        "Eurasian Wren",
        "Eurasian Jay",
        "Great Tit",
        "Blue Tit",
        "Eurasian Magpie"
    ]
    max_images = 500
    main_folder = "birds_dataset"

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    for query in search_queries:
        query_folder = os.path.join(main_folder, query.replace(" ", "_"))
        image_urls = fetch_image_urls(query, max_images * 2)  # Fetch more images to ensure enough valid ones
        print(f"Fetched {len(image_urls)} images for '{query}'")  # Debugging info

        # Filter valid images based on resolution
        valid_image_urls = filter_valid_images(image_urls, max_images)

        # Output the number of images found for verification
        print(f"Found {len(valid_image_urls)} valid images for '{query}'")

        download_images(valid_image_urls, query_folder, query.replace(" ", "_").lower())


if __name__ == "__main__":
    main()

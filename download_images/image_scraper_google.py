import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


def fetch_image_urls(query, max_images):
    image_urls = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    num_per_page = 20  # 20 images per page
    num_pages = max_images // num_per_page + 1

    for page in range(num_pages):
        start = page * num_per_page
        url = f"https://www.google.com/search?hl=en&q={quote(query)}&tbm=isch&start={start}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_elements = soup.find_all("img", {"src": re.compile("gstatic.com")})

        for img in image_elements:
            if len(image_urls) >= max_images:
                break
            image_url = img.get("src")
            image_urls.append(image_url)

        if len(image_elements) < num_per_page:
            break  # Exit if there are fewer images than expected on the page

    return image_urls


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
    search_queries = {
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
    }
    max_images = 500
    main_folder = "birds_dataset"

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    for query in search_queries:
        query_folder = os.path.join(main_folder, query.replace(" ", "_"))
        image_urls = fetch_image_urls(query, max_images)
        download_images(image_urls, query_folder, query.replace(" ", "_").lower())


if __name__ == "__main__":
    main()

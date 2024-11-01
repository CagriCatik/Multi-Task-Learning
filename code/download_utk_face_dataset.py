import requests
import zipfile
import os

# URL of the file to be downloaded
url = 'https://hydranets-data.s3.eu-west-3.amazonaws.com/UTKFace.zip'
output_dir = 'UTKFace'  # Directory where the extracted files will be stored
zip_path = 'UTKFace.zip'  # Path to save the downloaded zip file

def download_file(url, save_path):
    """Download the file from the URL and save it to the specified path."""
    print(f"Downloading file from {url}...")
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)
    print(f"File downloaded: {save_path}")

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed.")

def main():
    # Check if the directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download the zip file
    download_file(url, zip_path)

    # Extract the zip file
    extract_zip(zip_path, output_dir)

    # Optionally, remove the zip file after extraction
    os.remove(zip_path)
    print("Zip file removed after extraction.")

if __name__ == '__main__':
    main()

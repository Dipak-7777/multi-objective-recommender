import os
import yaml
import pandas as pd
import requests
from pathlib import Path

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_file(url, save_path):
    print(f"Downloading {url}...")
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {save_path}")

def main():
    config = load_config()
    raw_dir = Path(config['paths']['raw_data_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    files_to_get = {
        'ratings.csv': config['datasets']['ratings_url'],
        'metadata.csv': config['datasets']['metadata_url']
    }
    for filename, url in files_to_get.items():
        save_path = raw_dir / filename
        download_file(url, save_path)
    print("\n? Data ingestion complete. Raw files are now in data/raw/")

if __name__ == '__main__':
    main()

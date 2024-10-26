import os
from scripts.download_dataset import download_and_save_dataset
from scripts.convert_arrow_to_parquet import convert_arrow_to_parquet

class DatasetManager:
    def __init__(self, dataset_name, save_path):
        self.dataset_name = dataset_name
        self.save_path = save_path

    def ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def download_dataset(self):
        download_and_save_dataset(self.dataset_name, self.save_path)

    def convert_dataset(self, output_path):
        self.ensure_directory_exists(output_path)
        convert_arrow_to_parquet(self.save_path, output_path)

    def execute(self, choice, output_path=None):
        if choice == "download":
            self.download_dataset()
        elif choice == "convert" and output_path:
            self.convert_dataset(output_path)
        else:
            raise ValueError("Invalid choice or missing output_path for conversion")

if __name__ == "__main__":
    dataset_name = "Magpie-Align/Magpie-Reasoning-150K"
    save_path = f"data/raw/{dataset_name}"
    output_path = f"data/processed/{dataset_name}"

    manager = DatasetManager(dataset_name, save_path)
    
    manager.execute("download")
    manager.execute("convert", output_path)

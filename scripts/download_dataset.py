from datasets import load_dataset

def download_and_save_dataset(dataset_name: str, save_path: str):
    """
    Downloads a dataset from Hugging Face and saves it to the specified local path.

    Args:
        dataset_name (str): The Hugging Face dataset identifier (e.g., "squad" or "imdb").
        save_path (str): Local path where the dataset will be saved.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Save the dataset to the specified path
    dataset.save_to_disk(save_path)
    print(f"Dataset '{dataset_name}' downloaded and saved to '{save_path}'.")
from datasets import load_from_disk
import pandas as pd

def convert_arrow_to_parquet(dataset_path: str, output_dir: str):
    """
    Converts the dataset from Arrow format to Parquet.

    Args:
        dataset_path (str): Path to the directory containing the Arrow dataset.
        output_dir (str): Directory where the Parquet files will be saved.
    """
    # Load the dataset from the specified path
    dataset = load_from_disk(dataset_path)

    # Save each split as a Parquet file
    for split in dataset.keys():
        df = dataset[split].to_pandas()  # Convert to Pandas DataFrame
        df.to_parquet(f"{output_dir}/help_steer_{split}.parquet", index=False)
        print(f"Saved {split} split to {output_dir}/help_steer_{split}.parquet")

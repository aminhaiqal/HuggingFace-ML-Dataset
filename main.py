import argparse
from src.preprocessing.data_collection import DataCollector
from src.preprocessing.data_exporation import DataExplorer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Manager")
    parser.add_argument("--collect", action="store_true", help="Download and convert the dataset")
    parser.add_argument("--explore", action="store_true", help="Explore the dataset")

    args = parser.parse_args()

    dataset_name = "Magpie-Align/Magpie-Reasoning-150K"
    save_path = f"data/raw/{dataset_name}"
    output_path = f"data/processed/{dataset_name}"

    manager = DataCollector(dataset_name, save_path)

    if args.collect:
        manager.execute("download")
        manager.execute("convert_parquet", output_path)
    elif args.explore:
        # Add your exploration code here
        dataset_paths = [
            'data/processed/KingNish/reasoning-base-20k',
            #'data/processed/Magpie-Align/Magpie-Reasoning-150K',
            #'data/processed/SkunkworksAI/reasoning-0.01'
        ]
        explorer = DataExplorer(dataset_paths)
        explorer.basic_info()
        explorer.summary_statistics()
        explorer.missing_values()
    else:
        print("Please specify --collect or --explore")

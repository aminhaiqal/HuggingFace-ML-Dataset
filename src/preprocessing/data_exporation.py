import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self, dataset_paths):
        """
        Initializes the DataExplorer with paths to datasets.
        
        Args:
            dataset_paths (list): List of paths to the Parquet datasets.
        """
        self.datasets = [pd.read_parquet(path) for path in dataset_paths]

    def basic_info(self):
        """Prints the shape, columns, and data types for each dataset."""
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"Dataset {i} Shape: {dataset.shape}")
            print(f"Columns: {dataset.columns.tolist()}")
            print(f"Data Types:\n{dataset.dtypes}\n")

    def summary_statistics(self):
        """Prints summary statistics and sample rows for each dataset."""
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"Dataset {i} Summary Statistics:\n{dataset.describe(include='all')}\n")
            print(f"Sample Rows from Dataset {i}:\n{dataset.head()}\n")

    def missing_values(self):
        """Prints the missing values for each dataset."""
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"Missing Values in Dataset {i}:\n{dataset.isnull().sum()}\n")

    def visualize_distribution(self, column):
        """
        Visualizes the distribution of a specified numerical column in each dataset.
        
        Args:
            column (str): The name of the numerical column to visualize.
        """
        for i, dataset in enumerate(self.datasets, start=1):
            plt.figure(figsize=(10, 6))
            sns.histplot(dataset[column], kde=True)
            plt.title(f'Dataset {i} Distribution of {column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()

    def categorical_analysis(self):
        """Visualizes counts of categorical features in each dataset."""
        for i, dataset in enumerate(self.datasets, start=1):
            for col in dataset.select_dtypes(include=['object']).columns:
                plt.figure(figsize=(10, 6))
                dataset[col].value_counts().plot(kind='bar')
                plt.title(f'Dataset {i} - {col} Value Counts')
                plt.xlabel(col)
                plt.ylabel('Counts')
                plt.xticks(rotation=45)
                plt.show()

    def correlation_analysis(self):
        """Generates correlation heatmaps for numerical features in each dataset."""
        for i, dataset in enumerate(self.datasets, start=1):
            plt.figure(figsize=(10, 8))
            sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title(f'Dataset {i} Correlation Heatmap')
            plt.show()

    def conversation_length_analysis(self, column):
        """
        Analyzes the length of conversations in a specified column.
        
        Args:
            column (str): The name of the column containing conversation data.
        """
        for i, dataset in enumerate(self.datasets, start=1):
            dataset['conversation_length'] = dataset[column].apply(lambda x: len(x.split()))
            plt.figure(figsize=(10, 6))
            sns.histplot(dataset['conversation_length'], bins=30, kde=True)
            plt.title(f'Dataset {i} Conversation Length Distribution')
            plt.xlabel('Length of Conversation (in words)')
            plt.ylabel('Frequency')
            plt.show()

# Example Usage
if __name__ == "__main__":
    dataset_paths = [
        'data/processed/KingNish/reasoning-base-20k',
        #'data/processed/Magpie-Align/Magpie-Reasoning-150K',
        #'data/processed/SkunkworksAI/reasoning-0.01'
    ]
    
    explorer = DataExplorer(dataset_paths)
    explorer.basic_info()
    #explorer.summary_statistics()
    #explorer.missing_values()
    #explorer.visualize_distribution('your_numerical_column')  # Replace with your numerical column
    #explorer.categorical_analysis()
    #explorer.correlation_analysis()
    #explorer.conversation_length_analysis('conversations')  # Replace with your conversation column

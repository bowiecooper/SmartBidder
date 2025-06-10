import os
from data_generator import AdAuctionDataGenerator
import pandas as pd
from datetime import datetime
import numpy as np

def generate_and_save_dataset(n_samples: int = 100000, output_dir: str = 'data'):
    """
    Generate a large dataset and save it to CSV files.
    
    Args:
        n_samples: Number of samples to generate
        output_dir: Directory to save the data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = AdAuctionDataGenerator(seed=42)
    
    # Generate data
    print(f"Generating {n_samples} samples...")
    df = generator.generate_auction_data(n_samples=n_samples)
    
    # Add some derived features
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Calculate some derived metrics
    df['cpc'] = df['winning_bid'] / (df['click_through_rate'] * 1000)  # Cost per click
    df['cpa'] = df['cpc'] / df['conversion_rate']  # Cost per acquisition
    
    # Save full dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_path = os.path.join(output_dir, f'ad_auction_data_{timestamp}.csv')
    df.to_csv(full_path, index=False)
    print(f"Full dataset saved to: {full_path}")
    
    # Save a smaller sample for quick testing
    sample_path = os.path.join(output_dir, 'ad_auction_data_sample.csv')
    df.sample(n=1000, random_state=42).to_csv(sample_path, index=False)
    print(f"Sample dataset saved to: {sample_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nNumerical columns statistics:")
    print(df.describe())
    
    print("\nCategorical columns value counts:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head())

if __name__ == "__main__":
    generate_and_save_dataset()
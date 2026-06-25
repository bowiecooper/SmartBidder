"""Generate and save a large synthetic auction dataset to CSV.

Training (`python -m smartbidder.train`) generates data in-memory, so this script is
only needed if you want a CSV on disk for exploration/notebooks.
"""

import os
from datetime import datetime

from data_generator import AdAuctionDataGenerator


def generate_and_save_dataset(n_samples: int = 100_000, output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {n_samples:,} samples...")
    df = AdAuctionDataGenerator(seed=42).generate_auction_data(n_samples=n_samples)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"ad_auction_data_{timestamp}.csv")
    df.to_csv(full_path, index=False)
    print(f"Full dataset saved to: {full_path}")

    sample_path = os.path.join(output_dir, "ad_auction_data_sample.csv")
    df.sample(n=min(1000, len(df)), random_state=42).to_csv(sample_path, index=False)
    print(f"Sample dataset saved to: {sample_path}")

    print(f"\nTotal samples: {len(df):,}")
    print(f"Overall CTR: {df['clicked'].mean():.3%}")
    print(f"Overall conversion rate: {df['converted'].mean():.3%}")


if __name__ == "__main__":
    generate_and_save_dataset()

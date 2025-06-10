import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
from typing import List, Dict, Optional
import random

class AdAuctionDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        self.faker = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Define possible audience segments
        self.audience_segments = [
            'tech_enthusiasts', 'fashion_shoppers', 'sports_fans',
            'travel_enthusiasts', 'food_lovers', 'business_professionals',
            'students', 'parents', 'gaming_community'
        ]
        
        # Define possible ad categories
        self.ad_categories = [
            'electronics', 'fashion', 'sports', 'travel', 'food',
            'business', 'education', 'family', 'gaming'
        ]
        
        # Define possible device types
        self.device_types = ['mobile', 'desktop', 'tablet']
        
        # Define possible time slots (hour of day)
        self.time_slots = list(range(24))

    def generate_auction_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate simulated ad auction data."""
        data = {
            'timestamp': self._generate_timestamps(n_samples),
            'auction_id': [f'AUCTION_{i:06d}' for i in range(n_samples)],
            'base_cpm': self._generate_base_cpm(n_samples),
            'winning_bid': self._generate_winning_bids(n_samples),
            'audience_segment': self._generate_audience_segments(n_samples),
            'ad_category': self._generate_ad_categories(n_samples),
            'device_type': self._generate_device_types(n_samples),
            'hour_of_day': self._generate_time_slots(n_samples),
            'day_of_week': self._generate_day_of_week(n_samples),
            'click_through_rate': self._generate_ctr(n_samples),
            'conversion_rate': self._generate_conversion_rate(n_samples),
            'ad_position': self._generate_ad_positions(n_samples),
            'ad_size': self._generate_ad_sizes(n_samples),
            'user_location': self._generate_locations(n_samples),
            'user_age': self._generate_user_ages(n_samples),
            'user_gender': self._generate_user_genders(n_samples)
        }
        
        return pd.DataFrame(data)

    def _generate_timestamps(self, n_samples: int) -> List[datetime]:
        """Generate timestamps for the last 30 days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return [self.faker.date_time_between(start_date, end_date) for _ in range(n_samples)]

    def _generate_base_cpm(self, n_samples: int) -> np.ndarray:
        """Generate base CPM values (Cost Per Mille)."""
        return np.random.lognormal(mean=2.5, sigma=0.5, size=n_samples)

    def _generate_winning_bids(self, n_samples: int) -> np.ndarray:
        """Generate winning bid prices."""
        return np.random.lognormal(mean=3.0, sigma=0.6, size=n_samples)

    def _generate_audience_segments(self, n_samples: int) -> List[str]:
        """Generate audience segments with realistic distribution."""
        return random.choices(self.audience_segments, k=n_samples)

    def _generate_ad_categories(self, n_samples: int) -> List[str]:
        """Generate ad categories with realistic distribution."""
        return random.choices(self.ad_categories, k=n_samples)

    def _generate_device_types(self, n_samples: int) -> List[str]:
        """Generate device types with realistic distribution."""
        weights = [0.6, 0.3, 0.1]  # Mobile, Desktop, Tablet
        return random.choices(self.device_types, weights=weights, k=n_samples)

    def _generate_time_slots(self, n_samples: int) -> List[int]:
        """Generate hour of day with realistic distribution."""
        return random.choices(self.time_slots, k=n_samples)

    def _generate_day_of_week(self, n_samples: int) -> List[int]:
        """Generate day of week (0-6, where 0 is Monday)."""
        return random.choices(range(7), k=n_samples)

    def _generate_ctr(self, n_samples: int) -> np.ndarray:
        """Generate click-through rates."""
        return np.random.beta(2, 100, size=n_samples)  # Mean around 2%

    def _generate_conversion_rate(self, n_samples: int) -> np.ndarray:
        """Generate conversion rates."""
        return np.random.beta(1, 200, size=n_samples)  # Mean around 0.5%

    def _generate_ad_positions(self, n_samples: int) -> List[str]:
        """Generate ad positions."""
        positions = ['top', 'sidebar', 'content', 'bottom']
        weights = [0.3, 0.2, 0.3, 0.2]
        return random.choices(positions, weights=weights, k=n_samples)

    def _generate_ad_sizes(self, n_samples: int) -> List[str]:
        """Generate ad sizes."""
        sizes = ['300x250', '728x90', '160x600', '320x50']
        weights = [0.4, 0.3, 0.2, 0.1]
        return random.choices(sizes, weights=weights, k=n_samples)

    def _generate_locations(self, n_samples: int) -> List[str]:
        """Generate user locations."""
        return [self.faker.country_code() for _ in range(n_samples)]

    def _generate_user_ages(self, n_samples: int) -> List[int]:
        """Generate user ages with realistic distribution."""
        return np.random.normal(35, 10, n_samples).astype(int).clip(18, 65)

    def _generate_user_genders(self, n_samples: int) -> List[str]:
        """Generate user genders."""
        return random.choices(['M', 'F', 'O'], weights=[0.48, 0.48, 0.04], k=n_samples)

if __name__ == "__main__":
    # Example usage
    generator = AdAuctionDataGenerator()
    df = generator.generate_auction_data(n_samples=1000)
    print("Generated dataset shape:", df.shape)
    print("\nSample of generated data:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_data(n_samples: int = 100_000, fraud_ratio: float = 0.005, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    # 0 for benign, 1 for fraud
    y = np.random.binomial(1, fraud_ratio, n_samples)

    # generate base features
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='min')
    user_ids = np.random.randint(1, 1000, n_samples)
    merchant_ids = np.random.randint(1, 500, n_samples)

    # generate amounts with different distributions for normal vs fraud
    amounts = np.where(
        y == 0,
        np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples), # normal tx
        np.random.lognormal(mean=5.0, sigma=1.5, size=n_samples)  # fraud tx
    )

    df = pd.DataFrame({
        'timestamp': timestamps,
        'user_id': user_ids,
        'merchant_id': merchant_ids,
        'amount': amounts,
        'is_fraud': y
    })

    # Sort by time to emulate real transaction log
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # avoid data leakage across time splits
    split_idx = int(len(df) * train_ratio)

    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    return df_train, df_val

def load_or_generate_data(filepath: str = None) -> pd.DataFrame:
    if filepath:
        try:
            return pd.read_csv(filepath, parse_dates=['timestamp'])
        except FileNotFoundError:
            print(f"File {filepath} not found. Falling back to synthetic generator.")

    return generate_synthetic_data()

if __name__ == "__main__":
    df = load_or_generate_data()
    print(f"Data shape: {df.shape}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.4f}")
    df_train, df_val = time_based_split(df)
    print(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}")

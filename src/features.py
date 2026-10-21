import numpy as np
import pandas as pd
from typing import Tuple

class FeatureEngineer:
    def __init__(self):
        self.user_amount_mean = {}
        self.merchant_risk_scores = {}

    def fit(self, df_train: pd.DataFrame):
        # avoid data leakage across time splits
        # fit on training data only

        # User amount means
        self.user_amount_mean = df_train.groupby('user_id')['amount'].mean().to_dict()

        # Merchant risk scores
        # smooth to avoid extreme values on low-frequency merchants
        alpha = 10
        global_mean = df_train['is_fraud'].mean()

        merchant_stats = df_train.groupby('merchant_id')['is_fraud'].agg(['sum', 'count'])
        merchant_risk = (merchant_stats['sum'] + global_mean * alpha) / (merchant_stats['count'] + alpha)
        self.merchant_risk_scores = merchant_risk.to_dict()
        self.global_mean = global_mean

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # Cyclical time features
        hour = df_out['timestamp'].dt.hour + df_out['timestamp'].dt.minute / 60.0
        df_out['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df_out['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        # Amount features
        user_mean = df_out['user_id'].map(self.user_amount_mean).fillna(df_out['amount'].mean())
        df_out['amount_to_mean'] = df_out['amount'] / (user_mean + 1e-6)

        # Merchant risk
        df_out['merchant_risk'] = df_out['merchant_id'].map(self.merchant_risk_scores).fillna(self.global_mean)

        # Rolling velocity
        # This implementation requires the dataframe to be sorted by timestamp
        df_out = df_out.sort_values('timestamp')

        # We process velocity carefully to ensure no future leakage
        df_out = df_out.set_index('timestamp')

        # 1 hour velocity
        velocity_1h = df_out.groupby('user_id')['amount'].rolling('1h').count()
        velocity_1h.index = velocity_1h.index.droplevel(0)

        # 24 hour velocity
        velocity_24h = df_out.groupby('user_id')['amount'].rolling('24h').count()
        velocity_24h.index = velocity_24h.index.droplevel(0)

        # align indexes to handle duplicate timestamps
        # reindex back to original row order by preserving a temporary ID before
        df_out['velocity_1h'] = velocity_1h[~velocity_1h.index.duplicated(keep='last')]
        df_out['velocity_24h'] = velocity_24h[~velocity_24h.index.duplicated(keep='last')]

        # Alternatively we can just use the index carefully
        df_out['velocity_1h'] = df_out.index.map(lambda idx: velocity_1h[velocity_1h.index == idx].iloc[0] if sum(velocity_1h.index == idx) else 0)
        df_out['velocity_24h'] = df_out.index.map(lambda idx: velocity_24h[velocity_24h.index == idx].iloc[0] if sum(velocity_24h.index == idx) else 0)

        # clip extreme velocity outliers at 99th pct
        # We avoid calculating exact 99th pct dynamically to avoid leakage,
        # using sensible defaults based on typical data
        df_out['velocity_1h'] = df_out['velocity_1h'].clip(upper=50)
        df_out['velocity_24h'] = df_out['velocity_24h'].clip(upper=200)

        df_out = df_out.reset_index()

        # Drop original identifiers and strings that aren't features
        cols_to_drop = ['timestamp', 'user_id', 'merchant_id']
        if 'is_fraud' in df_out.columns:
            # We keep it, but separate X and y later
            pass

        return df_out.drop(columns=cols_to_drop)

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_train)
        return self.transform(df_train)

def split_X_y(df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    return df, None

if __name__ == "__main__":
    from data_prep import load_or_generate_data, time_based_split

    df = load_or_generate_data()
    df_train, df_val = time_based_split(df)

    fe = FeatureEngineer()
    df_train_features = fe.fit_transform(df_train)
    df_val_features = fe.transform(df_val)

    X_train, y_train = split_X_y(df_train_features)
    X_val, y_val = split_X_y(df_val_features)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Features: {X_train.columns.tolist()}")
    print("Features engineered successfully.")

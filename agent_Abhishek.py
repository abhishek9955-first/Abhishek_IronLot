import os
import joblib
import pandas as pd
import warnings
import numpy as np


class LiveAuctionAgent:

    def __init__(self):

        # Starting bankroll
        self.bankroll = 500000.0

        # Store predicted value
        self.predicted_value = 0.0

        # Current file path
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Load trained model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = joblib.load(
                os.path.join(base_path, "model_Abhishek.pkl")
            )

    # =========================================
    # ANALYZE ITEM
    # =========================================
    def analyze_item(self, item_features: dict):

        # Convert dictionary to dataframe
        car_df = pd.DataFrame([item_features])

        # =========================================
        # DATA VALIDATION & TYPE COERCION (FIXED)
        # =========================================
        
        cat_features = ["make", "model", "trim", "body", "state", "color", "interior", "transmission"]
        num_features = ["year", "condition", "odometer"]
        
        # Hardcoded medians from training to handle missing data safely
        num_defaults = {"year": 2015, "condition": 3.0, "odometer": 80000}

        # 1. Guarantee categorical columns exist
        for col in cat_features:
            if col not in car_df.columns:
                car_df[col] = "unknown"
                
        # 2. Guarantee numeric columns exist and cast types
        for col in num_features:
            if col not in car_df.columns:
                car_df[col] = num_defaults[col]
            else:
                # Force numeric type to prevent strings from breaking math operations
                car_df[col] = pd.to_numeric(car_df[col], errors='coerce').fillna(num_defaults[col])

        # =========================================
        # DATA CLEANING
        # =========================================

        # Categorical features - clean them
        for col in cat_features:
            car_df[col] = (
                car_df[col]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .str.strip()
            )

        # =========================================
        # FEATURE ENGINEERING
        # =========================================

        # Vehicle age
        car_df["vehicle_age"] = 2025 - car_df["year"]

        # Usage intensity
        car_df["usage_intensity"] = (
            car_df["odometer"] /
            car_df["vehicle_age"].clip(lower=1)
        )

        # Age x mileage interaction
        car_df["age_x_mileage"] = (
            car_df["vehicle_age"] *
            car_df["odometer"]
        )

        # =========================================
        # CONVERT TO CATEGORY DTYPE
        # =========================================

        for col in cat_features:
            car_df[col] = car_df[col].astype("category")

        # =========================================
        # SELECT FEATURES
        # =========================================

        features = [
            "year", "condition", "odometer",
            "vehicle_age", "usage_intensity", "age_x_mileage",
            "make", "model", "trim", "body", "state", "color", "interior", "transmission"
        ]

        car_df = car_df[features]

        # =========================================
        # PREDICTION
        # =========================================

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            prediction = self.model.predict(car_df)

        # Reverse log transformation to get actual price
        self.predicted_value = np.expm1(float(prediction[0]))

    # =========================================
    # PLACE BID
    # =========================================
    def place_bid(self, current_highest_bid: float) -> float:

        # Safe bidding limit
        my_max_price = self.predicted_value * 0.90

        # Deterministic increment
        my_next_bid = current_highest_bid + 50.0

        # Bid only if profitable
        if (
            my_next_bid <= my_max_price
            and my_next_bid <= self.bankroll
        ):

            return my_next_bid

        # Otherwise skip
        return 0.0

    # =========================================
    # AUCTION RESULT
    # =========================================
    def auction_result(
        self,
        won,
        winning_bid,
        actual_price,
        current_bankroll
    ):

        # Update bankroll
        self.bankroll = current_bankroll


# =========================================
# TESTING
# =========================================

if __name__ == "__main__":
    agent = LiveAuctionAgent()

    sample_item = {
        "year": 2018,
        "condition": 4,
        "odometer": 50000,
        "make": "Toyota",
        "model": "Corolla",
        "trim": "LE",
        "body": "Sedan",
        "transmission": "Automatic",
        "state": "CA",
        "color": "White",
        # Notice "interior" is missing here to test the fix. It will default to "unknown" instead of crashing.
    }

    # Analyze
    agent.analyze_item(sample_item)

    # Print predicted value
    print("Predicted value:", agent.predicted_value)

    # Place bid
    bid = agent.place_bid(10000)

    print("Next bid:", bid)
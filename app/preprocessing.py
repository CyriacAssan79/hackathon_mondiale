import numpy as np
import pandas as pd


def preprocess_input(data: dict, selected_features: list) -> pd.DataFrame:
    """
    Prétraitement identique à celui utilisé
    pendant l'entraînement.
    """

    # ===================================
    # Création du DataFrame
    # ===================================

    df = pd.DataFrame([data])

    # ===================================
    # Conversion traffic_density
    # ===================================

    traffic_mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    if df["traffic_density"].dtype == "object":
        df["traffic_density"] = (
            df["traffic_density"]
            .map(traffic_mapping)
            .fillna(1)
        )

    # ===================================
    # Feature Engineering
    # ===================================

    df["is_weekend"] = (
        df["day_of_week"]
        .apply(lambda x: 1 if x in [5, 6] else 0)
    )

    # Heure cyclique
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Direction cyclique
    df["heading_sin"] = np.sin(
        np.deg2rad(df["heading"])
    )

    df["heading_cos"] = np.cos(
        np.deg2rad(df["heading"])
    )

    epsilon = 1e-6

    # Ratio vitesse / accélération
    df["speed_acceleration_ratio"] = (
        df["speed"] /
        (df["acceleration"].abs() + epsilon)
    )

    df["speed_acceleration_ratio"] = (
        df["speed_acceleration_ratio"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # Distance / Temps
    df["distance_per_time"] = (
        df["remaining_distance"] /
        (df["remaining_time"] + epsilon)
    )

    df["distance_per_time"] = (
        df["distance_per_time"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # ===================================
    # One Hot Encoding
    # ===================================

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    # ===================================
    # Road Complexity
    # ===================================

    road_type_cols = [
        col
        for col in df.columns
        if col.startswith("road_type_")
    ]

    if len(road_type_cols) > 0:

        road_score = df[road_type_cols].sum(axis=1)

        df["road_complexity"] = (
            road_score *
            df["traffic_density"]
        )

    else:

        df["road_complexity"] = df["traffic_density"]

    # ===================================
    # Suppression colonnes
    # ===================================

    df.drop(
        columns=[
            "hour",
            "day_of_week",
            "heading"
        ],
        inplace=True,
        errors="ignore"
    )

    # ===================================
    # Même ordre que l'entraînement
    # ===================================

    df = df.reindex(
        columns=selected_features,
        fill_value=0
    )

    return df
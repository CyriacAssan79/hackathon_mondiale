from pathlib import Path
import joblib

# Dossier contenant les modèles
MODELS_DIR = Path(__file__).parent / "models"

MODEL_PATH = MODELS_DIR / "extra_trees_regressor.joblib"
FEATURES_PATH = MODELS_DIR / "selected_features.joblib"


class ModelLoader:
    """
    Charge le modèle et les variables une seule fois
    au démarrage de l'application.
    """

    def __init__(self):
        self.model = None
        self.selected_features = None

    def load(self):
        """Charge les fichiers .joblib"""

        self.model = joblib.load(MODEL_PATH)
        self.selected_features = joblib.load(FEATURES_PATH)

        print("✅ Modèle chargé avec succès")
        print(f"✅ Nombre de variables : {len(self.selected_features)}")

    def get_model(self):
        return self.model

    def get_features(self):
        return self.selected_features


# Instance globale
model_loader = ModelLoader()
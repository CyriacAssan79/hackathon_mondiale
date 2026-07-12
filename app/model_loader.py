from huggingface_hub import hf_hub_download
import joblib


# ==============================
# Configuration Hugging Face
# ==============================
REPO_ID = "Cyriac2002/extra_trees_regressor"

MODEL_FILENAME = "extra_trees_regressor.joblib"
FEATURES_FILENAME = "selected_features.joblib"


class ModelLoader:
    """
    Télécharge et charge le modèle ainsi que les variables
    une seule fois au démarrage de l'API.
    """

    def __init__(self):
        self.model = None
        self.selected_features = None
        self.loaded = False
        self.error = None

    def load(self):
        if self.loaded:
            return

        try:
            print("📥 Téléchargement des fichiers depuis Hugging Face...")

            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
            )

            features_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=FEATURES_FILENAME,
            )

            print("📦 Chargement du modèle...")

            self.model = joblib.load(model_path)
            self.selected_features = joblib.load(features_path)
            self.loaded = True
            self.error = None

            print("✅ Modèle chargé avec succès")
            print(f"✅ Nombre de variables : {len(self.selected_features)}")

        except Exception as exc:
            self.model = None
            self.selected_features = None
            self.loaded = False
            self.error = str(exc)
            print(f"⚠️ Impossible de charger le modèle au démarrage : {exc}")

    def get_model(self):
        self.load()
        return self.model

    def get_features(self):
        self.load()
        return self.selected_features

    def is_ready(self):
        self.load()
        return self.model is not None and self.selected_features is not None


# Instance globale
model_loader = ModelLoader()
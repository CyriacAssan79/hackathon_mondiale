from app.model_loader import model_loader
from app.preprocessing import preprocess_input


class PredictionService:

    def predict(self, data: dict):

        print("1 - Début prediction")

        model = model_loader.get_model()

        print("2 - Modèle chargé")

        selected_features = model_loader.get_features()

        print("3 - Features chargées")

        if model is None or selected_features is None:
            raise RuntimeError(
                "Le modèle n'est pas prêt. Vérifiez que le serveur a chargé le modèle au démarrage."
            )

        X = preprocess_input(
            data,
            selected_features
        )

        print("4 - Préprocessing terminé")
        print(X)

        prediction = model.predict(X)

        print("5 - Prediction terminée")

        return {
            "next_latitude": float(prediction[0][0]),
            "next_longitude": float(prediction[0][1])
        }


prediction_service = PredictionService()
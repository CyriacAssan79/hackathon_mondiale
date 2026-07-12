from app.model_loader import model_loader
from app.preprocessing import preprocess_input


class PredictionService:

    def predict(self, data: dict):

        model = model_loader.get_model()
        selected_features = model_loader.get_features()

        if model is None or selected_features is None:
            raise RuntimeError("Le modèle n'est pas disponible pour le moment.")

        X = preprocess_input(
            data,
            selected_features
        )

        prediction = model.predict(X)

        return {
            "next_latitude": float(prediction[0][0]),
            "next_longitude": float(prediction[0][1])
        }


prediction_service = PredictionService()
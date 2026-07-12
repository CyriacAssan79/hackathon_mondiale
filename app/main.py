from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)

from app.predictor import prediction_service

# ==================================================
# Création de l'application
# ==================================================

app = FastAPI(
    title="RiskMap AI API",
    description="API de prédiction de la prochaine position GPS d'un automobiliste",
    version="1.0.0"
)

# ==================================================
# Routes
# ==================================================

@app.get("/")
def root():

    return {
        "message": "RiskMap AI API",
        "status": "running"
    }


@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    return {
        "status": "healthy"
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():

    return {
        "model_name": "Extra Trees Regressor",
        "version": "1.0.0",
        "target": "Next GPS Position"
    }


@app.post(
    "/predict-position",
    response_model=PredictionResponse
)
def predict_position(request: PredictionRequest):

    try:

        prediction = prediction_service.predict(
            request.model_dump()
        )

        return prediction

    except RuntimeError as e:

        return JSONResponse(
            status_code=503,
            content={
                "error": str(e)
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
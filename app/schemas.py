from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Données envoyées par Base44 pour prédire
    la prochaine position GPS.
    """

    latitude: float = Field(..., description="Latitude actuelle")
    longitude: float = Field(..., description="Longitude actuelle")

    speed: float = Field(..., ge=0, description="Vitesse (km/h)")

    heading: float = Field(
        ...,
        ge=0,
        le=360,
        description="Direction du véhicule en degrés"
    )

    acceleration: float = Field(
        ...,
        description="Accélération actuelle"
    )

    remaining_distance: float = Field(
        ...,
        ge=0,
        description="Distance restante jusqu'à destination (m)"
    )

    remaining_time: float = Field(
        ...,
        ge=0,
        description="Temps restant (secondes)"
    )

    road_type: str = Field(
        ...,
        description="Type de route"
    )

    traffic_density: float = Field(
        ...,
        description="Densité du trafic"
    )

    weather: str = Field(
        ...,
        description="Conditions météorologiques"
    )

    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Heure actuelle"
    )

    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="0=Lundi ... 6=Dimanche"
    )


class PredictionResponse(BaseModel):
    """
    Réponse retournée par le modèle.
    """

    next_latitude: float
    next_longitude: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    target: str
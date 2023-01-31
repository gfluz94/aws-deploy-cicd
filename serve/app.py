from fastapi import FastAPI
from starlette.requests import Request
from mangum import Mangum

from serve.data import FeatureStoreDataRequest
from serve.predictor import PredictorService

MODELS_PATH = "models"
USERID_COLUMN = "uuid"
MERCHANT_GROUPS = [
    "Clothing & Shoes",
    "Intangible products",
    "Food & Beverage",
    "Erotic Material",
    "Entertainment",
]
LOG_TRANSFORM_COLUMNS = [
    "max_paid_inv_0_24m",
    "sum_capital_paid_account_0_12m",
]
predictor = PredictorService(
    model_path=MODELS_PATH,
    user_id_col=USERID_COLUMN,
    merchant_groups=MERCHANT_GROUPS,
    log_transform_cols=LOG_TRANSFORM_COLUMNS,
)
app = FastAPI()
handler = Mangum(app)


@app.get("/")
def home():
    return "Home"


@app.post("/predict")
def predict(request: Request, features: FeatureStoreDataRequest):
    if request.method == "POST":
        service_input = features.__dict__
        predictions = predictor.predict(service_input)
        return predictions
    return "No POST request found!"

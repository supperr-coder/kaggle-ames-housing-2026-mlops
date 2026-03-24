"""
FastAPI app exposing a /predict endpoint for the Ames Housing model.

Usage:
    uvicorn src.api:app --reload

Endpoints:
    GET  /         - health check
    POST /predict  - returns predicted log sale price + estimated sale price
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = "models/lgbm_model.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"

app = FastAPI(title="Ames Housing Price Predictor")

# Load model and feature list once at startup
model = joblib.load(MODEL_PATH)
with open(FEATURE_COLS_PATH) as f:
    feature_cols = json.load(f)


class HouseFeatures(BaseModel):
    Lot_Frontage: Optional[float] = 0.0
    Overall_Qual: Optional[float] = 0.0
    Overall_Cond: Optional[float] = 0.0
    Bsmt_Qual: Optional[float] = 0.0
    Bsmt_Cond: Optional[float] = 0.0
    Bsmt_Exposure: Optional[float] = 0.0
    BsmtFin_Type_1: Optional[float] = 0.0
    BsmtFin_SF_1: Optional[float] = 0.0
    BsmtFin_Type_2: Optional[float] = 0.0
    Bsmt_Unf_SF: Optional[float] = 0.0
    Total_Bsmt_SF: Optional[float] = 0.0
    Heating_QC: Optional[float] = 0.0
    Flr_1st_SF: Optional[float] = 0.0  # 1st_Flr_SF
    Flr_2nd_SF: Optional[float] = 0.0  # 2nd_Flr_SF
    Gr_Liv_Area: Optional[float] = 0.0
    Full_Bath: Optional[float] = 0.0
    Bedroom_AbvGr: Optional[float] = 0.0
    Kitchen_Qual: Optional[float] = 0.0
    TotRms_AbvGrd: Optional[float] = 0.0
    Functional: Optional[float] = 0.0
    Fireplaces: Optional[float] = 0.0
    Fireplace_Qu: Optional[float] = 0.0
    Garage_Cars: Optional[float] = 0.0
    Garage_Area: Optional[float] = 0.0
    Garage_Qual: Optional[float] = 0.0
    Garage_Cond: Optional[float] = 0.0
    Pool_QC: Optional[float] = 0.0
    Fence: Optional[float] = 0.0
    HouseAge: Optional[float] = 0.0
    Log_LotArea: Optional[float] = 0.0
    RemodAge: Optional[float] = 0.0
    GarageRatio: Optional[float] = 0.0
    RoomsPerArea: Optional[float] = 0.0
    TotalSF: Optional[float] = 0.0
    TotalBath: Optional[float] = 0.0
    Misc_Val_T: Optional[float] = 0.0
    Pool_Area_T: Optional[float] = 0.0
    Low_Qual_Fin_SF_T: Optional[float] = 0.0
    Ssn_Porch_3T: Optional[float] = 0.0   # 3Ssn_Porch_T
    BsmtFin_SF_2_T: Optional[float] = 0.0
    Enclosed_Porch_T: Optional[float] = 0.0
    Screen_Porch_T: Optional[float] = 0.0
    Mas_Vnr_Area_T: Optional[float] = 0.0
    Open_Porch_SF_T: Optional[float] = 0.0
    Wood_Deck_SF_T: Optional[float] = 0.0
    Year_Built: Optional[float] = 0.0
    Year_Remod_Add: Optional[float] = 0.0  # Year_Remod/Add
    Yr_Sold: Optional[float] = 0.0
    Garage_Yr_Blt: Optional[float] = 0.0
    MoSold_sin: Optional[float] = 0.0
    Mo_Sold: Optional[float] = 0.0
    MoSold_cos: Optional[float] = 0.0

    def to_model_input(self) -> pd.DataFrame:
        """Map API field names back to the original feature column names."""
        mapping = {
            "Flr_1st_SF": "1st_Flr_SF",
            "Flr_2nd_SF": "2nd_Flr_SF",
            "Ssn_Porch_3T": "3Ssn_Porch_T",
            "Year_Remod_Add": "Year_Remod/Add",
        }
        data = self.model_dump()  # model_dump() is inherited from BaseModel, returns dict of field values
        renamed = {mapping.get(k, k): v for k, v in data.items()}
        return pd.DataFrame([renamed])[feature_cols]


class PredictionResponse(BaseModel):
    log_sale_price: float
    estimated_sale_price: float

# @ get("/") refers to the root URL, which we can use for a simple health check to confirm the API
# is running and the model is loaded correctly when accessing the URL.
@app.get("/")
def health_check():
    return {"status": "ok", "model": "lgbm", "features": len(feature_cols)}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        X = features.to_model_input()
        log_price = float(model.predict(X)[0])
        sale_price = float(np.expm1(log_price))
        return PredictionResponse(
            log_sale_price=round(log_price, 4),
            estimated_sale_price=round(sale_price, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

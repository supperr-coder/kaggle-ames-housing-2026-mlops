"""
Tests for the FastAPI /predict endpoint.

The model and feature_cols.json are mocked so no real model file is needed —
this lets the tests run in CI without access to the models/ directory.
"""

import builtins
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

FEATURE_COLS = [
    "Lot_Frontage", "Overall_Qual", "Overall_Cond", "Bsmt_Qual", "Bsmt_Cond",
    "Bsmt_Exposure", "BsmtFin_Type_1", "BsmtFin_SF_1", "BsmtFin_Type_2",
    "Bsmt_Unf_SF", "Total_Bsmt_SF", "Heating_QC", "1st_Flr_SF", "2nd_Flr_SF",
    "Gr_Liv_Area", "Full_Bath", "Bedroom_AbvGr", "Kitchen_Qual", "TotRms_AbvGrd",
    "Functional", "Fireplaces", "Fireplace_Qu", "Garage_Cars", "Garage_Area",
    "Garage_Qual", "Garage_Cond", "Pool_QC", "Fence", "HouseAge", "Log_LotArea",
    "RemodAge", "GarageRatio", "RoomsPerArea", "TotalSF", "TotalBath",
    "Misc_Val_T", "Pool_Area_T", "Low_Qual_Fin_SF_T", "3Ssn_Porch_T",
    "BsmtFin_SF_2_T", "Enclosed_Porch_T", "Screen_Porch_T", "Mas_Vnr_Area_T",
    "Open_Porch_SF_T", "Wood_Deck_SF_T", "Year_Built", "Year_Remod/Add",
    "Yr_Sold", "Garage_Yr_Blt", "MoSold_sin", "Mo_Sold", "MoSold_cos"
]
# ISSUE: api.py loads the model from models/ which is excluded from git.
# SOLUTION: Mocking 
# Mock the model to return a FIXED log price of 12.0 (~$162,754)
mock_model = MagicMock()
mock_model.predict.return_value = [12.0]

# Patch joblib.load and open before src.api is imported so the module-level
# model loading doesn't fail looking for real files

# We only want to mock open() for feature_cols.json, NOT for all files.
# If we mock builtins.open globally, it breaks other libraries (e.g. dateutil)
# that need to open real files during import. So we use a selective mock that
# checks the filename and only fakes the feature_cols.json read.
_real_open = builtins.open

def _selective_open(*args, **kwargs):
    if len(args) > 0 and "feature_cols" in str(args[0]):
        return StringIO(json.dumps(FEATURE_COLS))
    return _real_open(*args, **kwargs)

with patch("joblib.load", return_value=mock_model), \
     patch("builtins.open", side_effect=_selective_open):
    from src.api import app



from fastapi.testclient import TestClient

client = TestClient(app)

# Sends a GET / request and checks the response is 200 OK with the right fields.
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "lgbm"
    assert data["features"] == len(FEATURE_COLS)

#  Sends a POST /predict with some house features and checks that log_sale_price 
# and estimated_sale_price come back correctly. Since we mocked the model to always 
# return 12.0, we know exactly what to expect.
def test_predict_returns_valid_response():
    payload = {
        "Overall_Qual": 7,
        "Gr_Liv_Area": 1500,
        "TotalSF": 2000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "log_sale_price" in data
    assert "estimated_sale_price" in data
    assert data["log_sale_price"] == 12.0
    assert data["estimated_sale_price"] > 0

# edge case test: if the client sends an empty JSON payload, shouldn't crash
def test_predict_defaults_missing_fields():
    """Sending an empty payload should use defaults (all 0.0) and still return a prediction."""
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert "estimated_sale_price" in response.json()

from fastapi import FastAPI
import pandas as pd
import json
import numpy as np
import backend_services
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Change "*" to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")  # Test route to check if API is working
def read_root():
    return {"message": "API is running"}


@app.get("/columns")
def get_columns():
    """Returns the column names of the DataFrame as a JSON response."""
    columns = backend_services.get_columns_names()
    return json.dumps(columns)

class RankingRequest(BaseModel):
    constraints: list
    method: str
    columns: list
    num_ret_tuples: int
    num_of_rankings: int
    num_of_samples: int = None
    k_samples: int = None

@app.post("/ranking")
def get_ranking(data: RankingRequest):
    """Computes ranking based on the given parameters."""
    result = backend_services.sort_data(
        data.constraints,
        data.method,
        data.columns,
        data.num_ret_tuples,
        data.num_of_rankings,
        data.num_of_samples,
        data.k_samples
    )
    return result  # FastAPI will automatically convert this to JSON


class StabilityRequest(BaseModel):
    W1: float
    W2: float
    columns: list



@app.post("/stability")
def get_stability(data: StabilityRequest):
    """Computes ranking stability based on the given parameters."""
    W1 = float(data.W1)
    W2 = float(data.W2)

    print(f"DEBUG: Converted W1={W1}, W2={W2}")

    result = backend_services.get_ranking_stability(
        W1,
        W2,
        data.columns
    )
    return result




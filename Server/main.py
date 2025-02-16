from fastapi import FastAPI
import pandas as pd
import json
import numpy as np
import backend_services
from pydantic import BaseModel

app = FastAPI()


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



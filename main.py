import os
import torch
import xarray as xr
import numpy as np
from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Dict

from data.ingestion import CopernicusIngestor
from models.transformer import SwinPredictor
from engine.boids import PhytoplanktonSim
from models.parameterizer import BoidParameterizer

app = FastAPI(title="Phytoplankton Digital Twin Backend")

DATA_PATH = Path("data/copernicus_data.nc")
sim_engine = None
predictor_model = None
parameterizer = BoidParameterizer()
dataset = None
current_frame_idx = 12


@app.on_event("startup")
async def startup_event():
    global sim_engine, predictor_model, dataset

    if not DATA_PATH.exists():
        ingestor = CopernicusIngestor(cache_dir="data")
        constraints = {
            "start": "2025-01-01", "end": "2025-03-31",
            "min_lon": -10.0, "max_lon": 10.0,
            "min_lat": 45.0, "max_lat": 55.0,
            "variables": ["chl", "thetao", "so"]
        }
        ingestor.fetch_subset("cmems_mod_glo_bgc_my_0.25_P1M-m", constraints)

    dataset = xr.open_dataset(DATA_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_model = SwinPredictor(img_size=64).to(device)
    predictor_model.eval()

    bounds = {"min_lon": -10.0, "max_lon": 10.0, "min_lat": 45.0, "max_lat": 55.0}
    sim_engine = PhytoplanktonSim(count=10000, bounds=bounds)


def bridge_ml_to_simulation(prediction_tensor: torch.Tensor) -> Dict:
    density_map = prediction_tensor.detach().cpu().numpy().squeeze()

    params = parameterizer.get_parameters(density_map)

    params.update({
        "temp_min": 12.0, "temp_max": 25.0,
        "sal_min": 30.0, "sal_max": 38.0,
        "max_speed": params.get("speed", 2.0),
        "perception": 0.5
    })
    return params


@app.post("/simulation/step")
async def run_step():
    global current_frame_idx, dataset, sim_engine, predictor_model

    if current_frame_idx >= len(dataset.time):
        raise HTTPException(status_code=400, detail="End of dataset reached")

    window = dataset["chl"].isel(time=slice(current_frame_idx - 12, current_frame_idx))
    input_tensor = torch.from_numpy(window.values).float().unsqueeze(0).unsqueeze(2)

    device = next(predictor_model.parameters()).device
    with torch.no_grad():
        prediction = predictor_model(input_tensor.to(device))

    current_params = bridge_ml_to_simulation(prediction)

    current_env = {
        'temp': dataset["thetao"].isel(time=current_frame_idx).values,
        'sal': dataset["so"].isel(time=current_frame_idx).values
    }

    sim_engine.step(current_env, current_params)
    current_frame_idx += 1

    return {
        "status": "success",
        "frame": current_frame_idx,
        "agent_count": len(sim_engine.positions),
        "params": current_params
    }


@app.get("/simulation/state")
async def get_state():
    if sim_engine:
        return {
            "positions": sim_engine.positions.tolist(),
            "count": len(sim_engine.positions)
        }
    return {"error": "Not initialized"}
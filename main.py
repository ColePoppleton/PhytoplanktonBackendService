import sys
import logging
from typing import Dict

import uvicorn
import torch
import xarray as xr
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path

from data.ingestion import CopernicusIngestor
from models.transformer import SwinPredictor
from engine.boids import PhytoplanktonSim
from models.parameterizer import BoidParameterizer

class DigitalTwinFormatter(logging.Formatter):
    def format(self, record):
        prefix = {
            logging.INFO: "  [SYSTEM] ",
            logging.ERROR: " [!] ERROR ",
            logging.WARNING: " [?] WARN  ",
            logging.DEBUG: " [>] DEBUG "
        }.get(record.levelno, "  ")
        return f"{self.formatTime(record, '%H:%M:%S')}{prefix}{record.getMessage()}"


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(DigitalTwinFormatter())
logger = logging.getLogger("digital_twin")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

DATA_PATH = Path("data/copernicus_data.nc")
sim_engine = None
predictor_model = None
parameterizer = BoidParameterizer()
dataset = None
current_frame_idx = 12


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_engine, predictor_model, dataset
    logger.info("=" * 50)
    logger.info("INITIALIZING PHYTOPLANKTON DIGITAL TWIN BACKEND")
    logger.info("=" * 50)

    if not DATA_PATH.exists():
        logger.info("DATA CHECK: Initializing fresh data ingestion pipeline.")
        ingestor = CopernicusIngestor(cache_dir="data")

        sync_config = {
            "cmems_mod_glo_bgc_my_0.25deg_P1M-m": ["chl"],
            "cmems_mod_glo_phy-all_my_0.25deg_P1M-m": ["thetao", "so"]
        }

        common_constraints = {
            "start": "2025-01-01", "end": "2025-03-31",
            "min_lon": -10.0, "max_lon": 10.0,
            "min_lat": 45.0, "max_lat": 55.0,
        }

        try:
            dataset = ingestor.fetch_and_merge(sync_config, common_constraints, DATA_PATH)
        except Exception as e:
            logger.error(f"FATAL: Pipeline failure during data sync. {e}")
            raise
    else:
        logger.info(f"DATA CHECK: Using existing merged dataset at {DATA_PATH}.")
        dataset = xr.open_dataset(DATA_PATH)

    logger.info("MODEL: Loading Swin Transformer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_model = SwinPredictor(img_size=64).to(device)
    predictor_model.eval()

    logger.info("SIMULATION: Spawning 10,000 Lagrangian agents...")
    bounds = {"min_lon": -10.0, "max_lon": 10.0, "min_lat": 45.0, "max_lat": 55.0}
    sim_engine = PhytoplanktonSim(count=10000, bounds=bounds)

    logger.info("STATUS: Digital Twin backend is ONLINE.")
    logger.info("-" * 50)
    yield
    if dataset: dataset.close()


app = FastAPI(title="Phytoplankton Digital Twin Backend", lifespan=lifespan)


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
        logger.warning("Simulation reached end of temporal dataset.")
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

    logger.debug(f"Step {current_frame_idx} complete. Active agents: {len(sim_engine.positions)}")

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
    return {"error": "Simulation engine not initialized"}

if __name__ == "__main__":
    logger.info("Starting local Uvicorn server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
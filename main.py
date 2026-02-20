import glob
import sys, logging, asyncio, uvicorn, torch, xarray as xr, json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from data.ingestion import CopernicusIngestor
from models.transformer import SwinPredictor
from engine.boids import PhytoplanktonSim
from models.parameterizer import BoidParameterizer
from engine.trainer import ModelTrainer
from data.processing import DataManager
from visualize_metrics import plot_training_results


class DigitalTwinFormatter(logging.Formatter):
    def format(self, record):
        prefix = {logging.INFO: "  [SYSTEM] ", logging.ERROR: " [!] ERROR ", logging.WARNING: " [?] WARN  "}.get(
            record.levelno, "  ")
        return f"{self.formatTime(record, '%H:%M:%S')}{prefix}{record.getMessage()}"


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(DigitalTwinFormatter())
logger = logging.getLogger("digital_twin")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


with open("config.json", "r") as f:
    cfg = json.load(f)

DATA_PATH = Path("data/copernicus_data.nc")
WEIGHTS_PATH = Path("models/weights/swin_latest.pt")
sim_engine = None
predictor_model = None
parameterizer = BoidParameterizer()
dataset = None
current_frame_idx = cfg["model"]["window_size"]
training_status = {"status": "idle", "epoch": 0, "total_epochs": 0}


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


async def update_training_status(epoch: int, total: int):
    global training_status
    training_status.update({"status": "training", "epoch": epoch, "total_epochs": total})


async def run_automated_pipeline():
    global predictor_model, DATA_PATH, training_status
    try:
        manager = DataManager(str(DATA_PATH))
        manager.extract_subset(spatial_ratio=cfg["model"]["spatial_ratio"])
        loaders = manager.get_splits(variable="chl")
        trainer = ModelTrainer(predictor_model, next(predictor_model.parameters()).device)

        log_path = await trainer.run_full_training(
            loaders, epochs=cfg["model"]["epochs"], save_path=WEIGHTS_PATH
        )
        training_status["status"] = "complete"
        predictor_model.eval()
        plot_training_results(log_path)
    except Exception as e:
        logger.error(f"PIPELINE ERROR: {e}")
        training_status["status"] = "failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_engine, predictor_model, dataset
    logger.info("=" * 50)
    logger.info("INITIALIZING PHYTOPLANKTON DIGITAL TWIN")
    logger.info("=" * 50)

    if not DATA_PATH.exists():
        ingestor = CopernicusIngestor()
        sync_config = {
            cfg["data"]["variable_mapping"]["bgc_product"]: ["chl"],
            cfg["data"]["variable_mapping"]["phy_product"]: [
                cfg["data"]["variable_mapping"]["temp_name"],
                cfg["data"]["variable_mapping"]["sal_name"]
            ]
        }
        dataset = ingestor.fetch_and_merge(sync_config, {
            "start": cfg["data"]["start_date"], "end": cfg["data"]["end_date"],
            "min_lon": cfg["data"]["min_lon"], "max_lon": cfg["data"]["max_lon"],
            "min_lat": cfg["data"]["min_lat"], "max_lat": cfg["data"]["max_lat"]
        }, DATA_PATH)
    else:
        dataset = xr.open_dataset(DATA_PATH)

    if cfg["data"]["variable_mapping"]["temp_name"] in dataset:
        dataset = dataset.rename({
            cfg["data"]["variable_mapping"]["temp_name"]: "thetao",
            cfg["data"]["variable_mapping"]["sal_name"]: "so"
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor_model = SwinPredictor(img_size=cfg["model"]["img_size"]).to(device)

    if WEIGHTS_PATH.exists():
        logger.info(f"MODEL: Weights found. Loading state from {WEIGHTS_PATH}")
        predictor_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        predictor_model.eval()

        log_files = glob.glob('logs/*.json')
        if log_files:
            latest_log = max(log_files, key=lambda x: Path(x).stat().st_mtime)
            logger.info(f"PIPELINE: Found existing logs. Generating fresh graphs...")
            plot_training_results(latest_log)
    else:
        asyncio.create_task(run_automated_pipeline())

    sim_engine = PhytoplanktonSim(
        count=cfg.get("simulation", {}).get("agent_count", 10000),
        bounds={
            "min_lon": cfg["data"]["min_lon"], "max_lon": cfg["data"]["max_lon"],
            "min_lat": cfg["data"]["min_lat"], "max_lat": cfg["data"]["max_lat"]
        }
    )
    logger.info("STATUS: Digital Twin backend is ONLINE.")
    yield
    if dataset: dataset.close()


app = FastAPI(title="Phytoplankton Digital Twin Backend", lifespan=lifespan)


@app.get("/model/status")
async def get_model_status():
    return training_status


@app.post("/simulation/step")
async def run_step():
    global current_frame_idx, dataset, sim_engine, predictor_model
    if current_frame_idx >= len(dataset.time):
        raise HTTPException(status_code=400, detail="End of dataset reached")

    window = dataset["chl"].isel(time=slice(current_frame_idx - cfg["model"]["window_size"], current_frame_idx))
    input_tensor = torch.from_numpy(window.values).float().unsqueeze(0).unsqueeze(2)

    with torch.no_grad():
        prediction = predictor_model(input_tensor.to(next(predictor_model.parameters()).device))

    current_params = bridge_ml_to_simulation(prediction)

    current_env = {
        'temp': dataset["thetao"].isel(time=current_frame_idx, depth=0).values if "depth" in dataset["thetao"].dims else
        dataset["thetao"].isel(time=current_frame_idx).values,
        'sal': dataset["so"].isel(time=current_frame_idx, depth=0).values if "depth" in dataset["so"].dims else dataset[
            "so"].isel(time=current_frame_idx).values
    }

    sim_engine.step(current_env, current_params)
    current_frame_idx += 1
    return {"status": "success", "frame": current_frame_idx, "agent_count": len(sim_engine.positions),
            "params": current_params}


@app.get("/simulation/state")
async def get_state():
    if sim_engine:
        return {"positions": sim_engine.positions.tolist(), "count": len(sim_engine.positions)}
    return {"error": "Not initialized"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
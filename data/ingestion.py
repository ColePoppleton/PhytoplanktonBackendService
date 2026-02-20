import copernicusmarine
import xarray as xr
from pathlib import Path
from typing import Dict, Optional

class CopernicusIngestor:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(exist_ok=True)

    def fetch_subset(self, dataset_id: str, constraints: Dict) -> xr.Dataset:
        filename = f"{dataset_id.replace('.', '_')}_subset.nc"
        file_path = self.cache_path / filename

        copernicusmarine.subset(
            dataset_id=dataset_id,
            start_datetime=constraints.get("start"),
            end_datetime=constraints.get("end"),
            minimum_longitude=constraints.get("min_lon"),
            maximum_longitude=constraints.get("max_lon"),
            minimum_latitude=constraints.get("min_lat"),
            maximum_latitude=constraints.get("max_lat"),
            variables=constraints.get("variables"),
            output_filename=str(file_path),
            overwrite=True
        )
        return xr.open_dataset(file_path)
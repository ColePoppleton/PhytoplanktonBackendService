import copernicusmarine
import xarray as xr
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("digital_twin")

class CopernicusIngestor:
    def __init__(self, cache_dir: str = "data"):
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
            overwrite=True,
            disable_progress_bar=False
        )
        return xr.open_dataset(file_path)

    def fetch_and_merge(self, dataset_configs: Dict[str, List[str]], common_constraints: Dict, output_path: Path) -> xr.Dataset:
        datasets = []
        for dataset_id, variables in dataset_configs.items():
            logger.info(f"COPERNICUS: Fetching {dataset_id} (variables: {variables})...")
            constraints = {**common_constraints, "variables": variables}
            ds = self.fetch_subset(dataset_id, constraints)
            datasets.append(ds)

        logger.info("Merging spatiotemporal data fields...")
        merged_ds = xr.merge(datasets)
        merged_ds.to_netcdf(output_path)
        logger.info(f"Data synchronised at {output_path}")
        return merged_ds
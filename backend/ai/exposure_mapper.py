from typing import Protocol, Any
from .schemas import Asset


class ExposureMapper(Protocol):
    """Modelling teammate implements conversion to their exposure contract."""
    def map_assets(self, assets: list[Asset]) -> list[dict[str, Any]]: ...


class CanonicalExposureMapper:
    """Lossless canonical payload; no geocoding or financial inference."""
    def map_assets(self, assets: list[Asset]) -> list[dict[str, Any]]:
        return [asset.model_dump(mode="json") for asset in assets]


def map_to_exposure_schema(data, **kwargs):
    from .exposure_mapping import map_to_exposure_schema as map_records
    return map_records(data, **kwargs)

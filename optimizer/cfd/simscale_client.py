"""SimScale API client factory.

Credentials come ONLY from the environment (or a local `.env` file that is
git-ignored):

    SIMSCALE_API_KEY   your personal API key  (never commit it, never paste it in chat)
    SIMSCALE_API_URL   optional, defaults to https://api.simscale.com

The key is stored with repr=False so it can never leak into logs or tracebacks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

API_KEY_HEADER = "X-API-KEY"
DEFAULT_URL = "https://api.simscale.com"


class SimScaleAuthError(RuntimeError):
    """Missing or rejected credentials: aborts the run instead of recording a failed evaluation."""


@dataclass
class SimScaleClient:
    api_client: Any
    api_key: str = field(repr=False)
    header: str = API_KEY_HEADER
    project_api: Any = None
    storage_api: Any = None
    geometry_import_api: Any = None
    geometry_api: Any = None
    mesh_operation_api: Any = None
    simulation_api: Any = None
    simulation_run_api: Any = None
    materials_api: Any = None

    def download(self, url: str) -> bytes:
        resp = self.api_client.rest_client.GET(url=url, headers={self.header: self.api_key}, _preload_content=False)
        return resp.data

    def upload(self, url: str, payload: bytes) -> None:
        self.api_client.rest_client.PUT(url=url, headers={"Content-Type": "application/octet-stream"}, body=payload)


def make_client() -> SimScaleClient:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("SIMSCALE_API_KEY")
    if not api_key:
        raise SimScaleAuthError(
            "SIMSCALE_API_KEY is not set. Copy .env.example to .env and put your key there "
            "(or export it in your shell). Do not share the key with anyone."
        )
    api_url = os.getenv("SIMSCALE_API_URL", DEFAULT_URL).rstrip("/")

    import urllib3
    from simscale_sdk import (ApiClient, Configuration, GeometriesApi, GeometryImportsApi, MaterialsApi,
                              MeshOperationsApi, ProjectsApi, SimulationRunsApi, SimulationsApi, StorageApi)

    configuration = Configuration()
    configuration.debug = False                       # debug logging would print request headers
    configuration.host = api_url + "/v0"
    configuration.api_key = {API_KEY_HEADER: api_key}
    api_client = ApiClient(configuration)
    api_client.rest_client.pool_manager.connection_pool_kw["retries"] = urllib3.Retry(
        connect=5, read=5, redirect=0, status=5, backoff_factor=0.2
    )
    return SimScaleClient(
        api_client=api_client,
        api_key=api_key,
        project_api=ProjectsApi(api_client),
        storage_api=StorageApi(api_client),
        geometry_import_api=GeometryImportsApi(api_client),
        geometry_api=GeometriesApi(api_client),
        mesh_operation_api=MeshOperationsApi(api_client),
        simulation_api=SimulationsApi(api_client),
        simulation_run_api=SimulationRunsApi(api_client),
        materials_api=MaterialsApi(api_client),
    )

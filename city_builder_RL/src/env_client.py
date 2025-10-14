import time
from typing import Any, Dict, Optional
import requests

from .config import Config

class EnvClient:
    """HTTP client for the City Builder environment."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.env_id: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.server_meta: Dict[str, Any] = {}

    # --- internal helpers with retries ---
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.BASE_URL + path
        last_err = None
        for attempt in range(self.cfg.RETRIES):
            try:
                r = self.session.post(url, json=payload, timeout=self.cfg.API_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.RETRY_BACKOFF_SEC * (2 ** attempt))
        raise RuntimeError(f"POST {path} failed after retries: {last_err}")

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.cfg.BASE_URL + path
        last_err = None
        for attempt in range(self.cfg.RETRIES):
            try:
                r = self.session.get(url, params=params or {}, timeout=self.cfg.API_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.RETRY_BACKOFF_SEC * (2 ** attempt))
        raise RuntimeError(f"GET {path} failed after retries: {last_err}")

    # --- API ---
    def init(self) -> Dict[str, Any]:
        resp = self._post("/init", {})
        self.env_id = resp["env_id"]
        self.server_meta = dict(resp.get("config", {}))
        return resp

    def seed(self, seed: int) -> Dict[str, Any]:
        assert self.env_id, "Call /init first"
        return self._post("/seed", {"env_id": self.env_id, "seed": seed})

    def reset(self) -> Dict[str, Any]:
        assert self.env_id, "Call /init first"
        resp = self._post("/reset", {"env_id": self.env_id})
        self.episode_id = resp["episode_id"]
        return resp

    def step(self, item_id: int) -> Dict[str, Any]:
        assert self.env_id and self.episode_id, "Call /reset first"
        return self._post("/step", {
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "action": {"item_id": item_id}
        })

    def summary(self) -> Dict[str, Any]:
        assert self.env_id and self.episode_id, "Call /reset first"
        return self._get(f"/episode/{self.episode_id}/summary", {"env_id": self.env_id})

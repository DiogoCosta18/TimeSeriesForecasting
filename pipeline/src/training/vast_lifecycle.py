from __future__ import annotations

import os


def vast_config_from_env() -> dict:
    return {
        "auto_destroy_requested": os.environ.get("VAST_AUTO_DESTROY", "0"),
        "output_sync_uri": os.environ.get("OUTPUT_SYNC_URI", ""),
        "vast_instance_id": os.environ.get("VAST_INSTANCE_ID"),
        "container_id": os.environ.get("CONTAINER_ID"),
    }


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def export_eff_parametric_metrics(
    *, out_path: str, results: Mapping[str, Any], run_config: Mapping[str, Any], per_solution: Mapping[str, Any] | None = None
) -> None:
    """Export evaluation results using `metrics-structures`.

    We store split-level metrics in `RunData.metadata` so we don't have to force-fit
    them into epoch-wise arrays.
    """

    from metrics_structures import RunData

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = _jsonify(dict(run_config))
    metadata.update(_jsonify(dict(results)))
    if per_solution is not None:
        metadata["per_solution"] = _jsonify(dict(per_solution))

    rd = RunData(metadata=metadata)
    # If caller provided these, also populate the dedicated fields.
    if "test_wall_time_sec" in metadata:
        try:
            rd.wall_time = float(metadata["test_wall_time_sec"])
        except Exception:
            pass
    if "test_batches_per_sec" in metadata:
        try:
            rd.it_per_sec = float(metadata["test_batches_per_sec"])
        except Exception:
            pass

    if out.suffix.lower() == ".json":
        rd.save_json(str(out))
    else:
        # default: pickle
        rd.save(str(out))



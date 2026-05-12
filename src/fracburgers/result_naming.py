"""Automated result organization and labeling based on parameter choices."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def sanitize_value(v: Any) -> str:
    """Convert a parameter value to a filesystem-safe string."""
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p")
    if isinstance(v, (list, tuple)):
        return "_".join(sanitize_value(x) for x in v)
    return str(v).replace("/", "_").replace(".", "_")


def build_result_folder(base_dir: Path, script_name: str, params: dict[str, Any]) -> Path:
    """
    Build an organized result folder path based on script name and key parameters.

    Args:
        base_dir: Root results directory (e.g., Path("results"))
        script_name: Name of the script (e.g., "solve", "train_pinn", "compare")
        params: Dictionary of parameters. Includes a special "__tags" key (list of parameter names)
                that controls which parameters appear in the folder name.

    Returns:
        A Path object pointing to the organized result folder.

    Example:
        params = {
            "ic": "sine",
            "alpha": 0.5,
            "nu": 0.1,
            "N": 512,
            "__tags": ["ic", "alpha", "nu", "N"]
        }
        path = build_result_folder(Path("results"), "solve", params)
        # Returns: results/solve/ic_sine/alpha_0p5/nu_0p1/N_512
    """
    base_dir = Path(base_dir)
    
    # Determine which parameters to include in the path
    tags = params.pop("__tags", None)
    if tags is None:
        # Default tags based on common parameters
        default_tags = {
            "solve": ["ic", "alpha", "nu", "N"],
            "train_pinn": ["ic", "alpha", "nu", "epochs"],
            "compare": ["ic", "alpha", "nu", "N"],
            "reference_convergence": ["k"],
            "plot_reference": ["a", "b", "nu"],
            "plot_diffusion_dispersion": ["nu"],
        }
        tags = default_tags.get(script_name, [])
    
    # Build the path incrementally: results/<script>/<key_param1>/<key_param2>/...
    path = base_dir / script_name
    
    for tag in tags:
        if tag in params:
            value = params[tag]
            safe_val = sanitize_value(value)
            path = path / f"{tag}_{safe_val}"
    
    return path


def get_output_dir(
    base_results_dir: Path,
    script_name: str,
    params: dict[str, Any],
) -> Path:
    """
    Convenience wrapper: builds the result folder and ensures it exists.

    Args:
        base_results_dir: Root results directory
        script_name: Name of the script
        params: Parameter dictionary (with optional __tags key)

    Returns:
        The result folder path (created if it doesn't exist).
    """
    path = build_result_folder(base_results_dir, script_name, params)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_params_for_label(**kwargs) -> str:
    """
    Format parameters as a human-readable label for figure titles or logging.

    Example:
        label = format_params_for_label(ic="sine", alpha=0.5, nu=0.1)
        # Returns: "ic=sine, alpha=0.5, nu=0.1"
    """
    items = [f"{k}={sanitize_value(v).replace('_', '.')}" for k, v in kwargs.items()]
    return ", ".join(items)

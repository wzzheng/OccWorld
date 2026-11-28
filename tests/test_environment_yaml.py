from pathlib import Path

import yaml


def test_setuptools_in_conda_dependencies_before_pip():
    """Ensure setuptools==60.2.0 is available in conda dependencies before pip runs."""
    env_path = Path(__file__).resolve().parent.parent / "environment.yaml"
    data = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    deps = data["dependencies"]

    pip_indices = [
        idx for idx, item in enumerate(deps) if isinstance(item, dict) and "pip" in item
    ]
    assert len(pip_indices) == 1
    pip_idx = pip_indices[0]

    conda_setuptools = [
        idx
        for idx, item in enumerate(deps)
        if isinstance(item, str)
        and (item == "setuptools=60.2.0" or item.startswith("setuptools=60.2.0="))
    ]
    assert conda_setuptools, "setuptools=60.2.0 must be listed in conda dependencies"
    assert conda_setuptools[0] < pip_idx

    pip_deps = deps[pip_idx]["pip"]
    pip_setuptools = [
        item
        for item in pip_deps
        if isinstance(item, str) and item.startswith("setuptools==")
    ]
    assert not pip_setuptools, "setuptools should not be duplicated only under pip"

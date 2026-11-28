import ast
from pathlib import Path


def test_get_meta_data_documents_rel_poses_and_gt_mode():
    """Ensure get_meta_data documents rel_poses coordinates and gt_mode semantics."""
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "dataset.py"
    tree = ast.parse(dataset_path.read_text(encoding="utf-8"))

    doc = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "nuScenesSceneDatasetLidar"
        ):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_meta_data":
                    doc = ast.get_docstring(item)
                    break

    assert doc is not None, "get_meta_data must have a docstring"
    assert "rel_poses" in doc
    assert "LiDAR" in doc
    assert "gt_mode" in doc
    assert "[1, 0, 0]" in doc
    assert "[0, 1, 0]" in doc
    assert "[0, 0, 1]" in doc

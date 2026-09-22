from pathlib import Path
import warnings

from analysis.som import SOMGridSearch
from analysis.dataloader import DataLoader
from analysis.analysis import DataConfig
import matplotlib.pyplot as plt


def main_grid_search(
    ghi_dir: str | Path,
    config: DataConfig | None = None,
    output_dir: str | Path | None = None,
    shapes=None
) -> None:
    """Run SOM grid search and save its visualisation and specs to a folder."""

    config = config or DataConfig()
    if output_dir is None or not Path(output_dir).is_absolute():
        warnings.warn("output_dir should be a full path; using the current working directory", UserWarning, stacklevel=2)
        output_path = Path.cwd()
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = DataLoader.from_file(
        ghi_dir,
        regrid_equal_area=config.regrid_equal_area,
    )
    grid_search = SOMGridSearch(data, config)

    figure, _results = grid_search.run_grid_search(
        output_path=output_path / "grid_search_specs.json",
        shapes=shapes
    )
    figure.savefig(output_path / "grid_search.png", dpi=150, bbox_inches="tight")

    plt.show()

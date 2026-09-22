import matplotlib
# W&B sweep jobs may run outside the main thread; never use the MacOS GUI
# backend for their diagnostic figures.
matplotlib.use("Agg", force=True)

import wandb
from .som import SOM, DEFAULT_TRAINING_ARGS
from .dataloader import DataLoader
from .analysis import DataConfig
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from functools import partial

NAMES = {"DJF":"summer", "MAM":"autumn", "JJA":"winter", "SON":"spring"}
GRID_SIZES = {"DJF":(3,3), "MAM":(3,4), "JJA":(3,3), "SON":(3,3)}
GHI_DIR = Path("aus_ghi") / "ghi_20200101_20201231.nc"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 0

def main(project_name: str, season: str, output_path: str | Path, sweep_name="hyperparam_sweep"):
    data = DataLoader.from_file(GHI_DIR, regrid_equal_area=True)
    data_config = DataConfig(season=season)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sweep_configuration = {
        "name": sweep_name,
        "method": "grid",
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {
            # "sigma": {"values":[2, 3]},
            # "learning_rate": {"values": [0.1]},
            "epochs": {"values":[200, 300, 400, 500]},
            "batch_size": {"value": 4}
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wandb.agent(sweep_id, 
                function=partial(
                    train,
                    project_name=project_name,
                    data_config=data_config,
                    size=GRID_SIZES[season],
                    output_path=output_path,
                    data=data,
                ),
                count=10)


def score(t, q):
    return t + q/300

def train(project_name, data_config, size, output_path, data):

    with wandb.init(project=project_name) as run:
        rows, cols = size
        # Keep every trial's local figures separate.
        output_path = Path(output_path) / run.id
        output_path.mkdir(parents=True, exist_ok=True)
        cfg = dict(run.config)
        cfg_training_args = {
                key: cfg[key]
                for key in DEFAULT_TRAINING_ARGS
                if key in cfg
            }
        

        som_training_args = dict(DEFAULT_TRAINING_ARGS)
        som_training_args.update(cfg_training_args)
        som_training_args["plot_errors"] = True

        season_name = NAMES[data_config.season]

        som = SOM(data, data_config).fit(
            rows,
            cols,
            output_dir=output_path,
            **som_training_args,
        )
        training_errors_path = output_path / "training_errors.png"
        if training_errors_path.exists():
            run.log({"error_curves": wandb.Image(str(training_errors_path))})

        nodes_figure, _ = som.plot_nodes(
            units_label="GHI",
            sample_label="day",
            title=f"{season_name} SOM nodes",
        )
        nodes_path = output_path / "nodes.png"
        nodes_figure.savefig(nodes_path, dpi=150, bbox_inches="tight")
        plt.close(nodes_figure)
        run.log({"nodes": wandb.Image(str(nodes_path))})


        correlation_figure, correlations = som.plot_bmu_correlations(figsize=(7, 5))
        corr_path = output_path / "bmu_correlations.png"
        correlation_figure.savefig(
            corr_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(correlation_figure)
        run.log({"correlation_matrix": wandb.Image(str(corr_path))})

        cdf_figure, _, _ = som.plot_bmu_correlation_cdf()
        cdf_path = output_path / "cdf.png"
        cdf_figure.savefig(cdf_path, dpi=150, bbox_inches="tight")
        plt.close(cdf_figure)

        run.log({"cdf": wandb.Image(str(cdf_path))})
        top_error = som.training_metrics["topographic_error"]
        quant_error = som.training_metrics["quantization_error"]
        run.log({"top_error": top_error, "quant_error": quant_error})
        run.log({"score": score(top_error, quant_error)})


if __name__ == "__main__":
    main("test_sweep", "DJF", "results/sweep", "epoch_sweep")

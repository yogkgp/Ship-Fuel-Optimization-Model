"""One command reproduces processing, experiments, figures and report summaries."""
import logging

from ship_fuel.train import run_training
from ship_fuel.visualization import make_figures

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data, splits, bundle = run_training()
    make_figures(data, splits, bundle)
    logging.info("Pipeline complete; selected model: %s", bundle["name"])

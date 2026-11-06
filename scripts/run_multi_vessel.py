"""Train all vessel-specific models and repeated temporal evaluations."""
import logging

from ship_fuel.multivessel import run_all_vessels
from ship_fuel.cross_vessel_visualization import make_cross_vessel_figures

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_all_vessels()
    make_cross_vessel_figures()
    logging.info("Completed vessels: %s", ", ".join(result["vessel"] for result in results))

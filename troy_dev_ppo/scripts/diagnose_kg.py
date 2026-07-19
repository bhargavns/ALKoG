#!/bin/python3
import sys, os
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.SymbolicKG import SymbolicKG
from lib import Diagnostics

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

required_arguments = []
optional_arguments = {
    "kg": "latest",   # path to a kg.pt / kg_trained.pt, or 'latest' for newest phase-1 run
    "out_dir": "",    # where to write plots + report; default: diag/ next to the kg file
    "device": "cpu",  # inspection needs no GPU
}

USAGE = """
diagnose_kg.py -- Verbose inspection of a saved symbolic knowledge graph.

Prints everything the KG stores (concept nodes, detection counts, visual
similarity structure, 4-dim symbols for nodes and relations, edges with
observation counts) and writes the same report plus diagnostic plots
(similarity heatmap, symbol tables, graph diagram, embedding PCA) to
`out_dir`. Works on both phase-1 (kg.pt) and phase-2 (kg_trained.pt) graphs.

    Required:

    Optional:
        kg=latest out_dir= device=cpu

    Example Usage:
        diagnose_kg.py kg=.../output/runs/ppo_20260717_120000/kg_trained.pt
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def diagnose():
    kg_path = g_ArgParse.get("kg")
    out_dir = g_ArgParse.get("out_dir")
    device = g_ArgParse.get("device")

    if kg_path == "latest":
        kg_path = Diagnostics.latest_kg_path(_OUTPUT)
    if not out_dir:
        out_dir = Diagnostics.make_run_dir(os.path.dirname(kg_path), "diag")
    else:
        os.makedirs(out_dir, exist_ok=True)

    kg = SymbolicKG.load(kg_path, device=device)
    print(f"Loaded {kg_path}")

    # exemplar crops live next to a phase-1 kg.pt; reference them if present
    exemplar_dir = os.path.join(os.path.dirname(kg_path), "exemplars")
    report = Diagnostics.kg_report(
        kg, exemplar_dir=exemplar_dir if os.path.isdir(exemplar_dir) else None
    )
    print("\n" + report)
    report_path = Diagnostics.unique_path(os.path.join(out_dir, "kg_report.txt"))
    with open(report_path, "w") as f:
        f.write(f"kg: {kg_path}\n\n" + report + "\n")
    print(f"Wrote report {report_path}")
    for p in Diagnostics.save_kg_plots(kg, out_dir):
        print(f"Wrote plot {p}")


def main(inputArguments):
    initialize(inputArguments)
    diagnose()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)

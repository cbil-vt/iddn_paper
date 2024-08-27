"""
Find the path of simulation input and output data
"""

from pathlib import Path


def get_work_folder():
    with open(Path.home() / "ddn_cfg.txt") as f:
        lines = f.readlines()
    top_folder = "./"
    for line in lines:
        cur_line = line.strip("\n")
        xx = cur_line.split("=")
        if xx[0] == "ddn":
            top_folder = xx[1]

    return top_folder

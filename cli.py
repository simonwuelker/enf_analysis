import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("input", type=str, help="the path to the input audio file")
parser.add_argument("--enf", help="the electric network frequency, defaults to 50Hz", default=50)

args = parser.parse_args()

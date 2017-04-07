import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node


def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	tsv = "../Question_C/data/map_0/grid_0.tsv"
	v = viterbi_matrix(load_path=tsv)

if __name__ == '__main__':
	main()

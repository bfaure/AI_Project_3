import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node


def main():

	data_dir = "../Question_C/data/"

	map_dir = data_dir+"map_0/"

	tsv = map_dir+"grid_0.tsv"
	v = viterbi_matrix(load_path=tsv)

	seq = map_dir+"traversal_0.txt"
	v.load_observations(seq,buffer_size=10)


if __name__ == '__main__':
	main()

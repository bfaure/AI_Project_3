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
	v.load_observations(seq,grid_buffer_size=0,path=True)

	'''
	num_grid_files = 10
	traversals_per_file = 10

	for grid_idx in range(num_grid_files):
		map_dir = data_dir+"map_"+str(grid_idx)+"/"
		tsv = map_dir+"grid_"+str(grid_idx)+".tsv"
		v = viterbi_matrix(load_path=tsv)

		for trav_idx in range(traversals_per_file):
			trav_file = map_dir+"traversal_"+str(trav_idx)+".txt"
			v.load_observations(trav_file,grid_buffer_size=1,path=False)
			if trav_idx!=traversals_per_file-1: v.reload_conditions_matrix() # reset weights
	'''

if __name__ == '__main__':
	main()

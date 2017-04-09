import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

def main():
	start_time = time.time()
	src_dir = "../Question_C/data/"
	runtime_code = str(int(time.time()))

	num_grid_files = 10
	traversals_per_file = 10
	buf_len = 20

	for grid_idx in range(num_grid_files):
		map_dir = src_dir+"map_"+str(grid_idx)+"/"
		tsv = map_dir+"grid_"+str(grid_idx)+".tsv"
		v = viterbi_matrix(load_path=tsv)
		total_score = 0

		for trav_idx in range(traversals_per_file):
			trav_file = map_dir+"traversal_"+str(trav_idx)+".txt"
			save_dir = "exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/traversal_"+str(trav_idx)
			total_score += v.load_observations(trav_file,grid_buffer_size=buf_len,path=True,save_dir=save_dir,print_nothing=True)
			if trav_idx!=traversals_per_file-1: v.reload_conditions_matrix() # reset weights

		data_file = open("exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/meta.txt","w")
		data_file.write("Total Score: "+str(total_score)+"\n")
		data_file.write("Buffer Size: "+str(buf_len)+"\n")
		data_file.close()

	end_time = time.time()
	print("\nDone. Total time: "+str(end_time-start_time)[:7]+" seconds")

if __name__ == '__main__':
	main()

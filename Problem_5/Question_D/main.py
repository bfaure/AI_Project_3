import sys
import time
import random
import os
from copy import deepcopy, copy

import imageio

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

# writes a gif in parent_folder made up of all it's sorted .png files
def make_gif(parent_folder):
	items = os.listdir(parent_folder)
	png_filenames = []
	for elem in items:
		if elem.find(".png")!=-1:
			png_filenames.append(elem)

	#sorted(png_filenames,key=int())
	sorted_png = []
	while True:
		lowest = 10000000
		lowest_idx = -1
		for p in png_filenames:
			val = int(p.split("-")[2].split(".")[0])
			if lowest_idx==-1 or val<lowest:
				lowest = val
				lowest_idx = png_filenames.index(p)
		sorted_png.append(png_filenames[lowest_idx])
		del png_filenames[lowest_idx]
		if len(png_filenames)==0: break
	png_filenames = sorted_png

	with imageio.get_writer(parent_folder+"/prediction-heatmap.gif", mode='I',duration=0.2) as writer:
		for filename in png_filenames:
			image = imageio.imread(parent_folder+"/"+filename)
			writer.append_data(image)

def main():
	start_time = time.time()
	src_dir = "../Question_C/data/"
	runtime_code = str(int(time.time()))

	num_grid_files = 1
	traversals_per_file = 1
	buf_len = 0
	overall_total_score = 0

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
			make_gif("exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/traversal_"+str(trav_idx))

		data_file = open("exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/meta.txt","w")
		data_file.write("Total Score: "+str(total_score)+"\n")
		data_file.write("Buffer Size: "+str(buf_len)+"\n")
		data_file.close()
		overall_total_score+=total_score

	end_time = time.time()
	print("\nDone. Total time: "+str(end_time-start_time)[:7]+" seconds")
	print("Overall total score: "+str(overall_total_score))
	dir_name = "exec_data-"+runtime_code
	os.rename(dir_name,dir_name+"-(complete)")

if __name__ == '__main__':
	main()

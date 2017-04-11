import sys
import time
import random
import os
from copy import deepcopy, copy

import imageio
import threading

import signal

#from helpers import viterbi_matrix, viterbi_node

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import mlab as ml
from matplotlib import colors

import numpy as np


import Cython, subprocess
import shutil, filecmp

if os.path.exists("helpers.pyx"):
	if filecmp.cmp("../helpers.py","helpers.pyx")==False: # if they are not the same already
		shutil.copyfile("../helpers.py","helpers.pyx")
else:
	shutil.copyfile("../helpers.py","helpers.pyx")

val = subprocess.Popen('python setup.py build_ext --inplace',shell=True).wait()


#import Cython
#import subprocess

#sys.path.insert(0,"..")
#val = subprocess.Popen('python ../setup.py build_ext --inplace',shell=True).wait()

from helpers import viterbi_matrix,viterbi_node

threads_open = 0
gif_manager_sleep_time = 0.1

# waits until there are at least num_pngs in parent_folder then calls make_gif()
def gif_creation_manager(parent_folder,num_pngs):
	global threads_open
	threads_open += 1 # increment number of open gif managers

	# wait until all png's are written
	while True:
		if exiting: return
		items = os.listdir(parent_folder)
		num_cur_png = 0
		for elem in items:
			if elem.find(".png")!=-1: num_cur_png+=1
		if num_cur_png==num_pngs:
			break
		else:
			time.sleep(2.0)

	# assemble the gif
	make_gif(parent_folder)

	threads_open-=1
	return

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

open_grids = ""

def exit_handler(signum,frame):
	global exiting
	exiting = True
	for a in open_grids:
		a.signal_exit()
	print("\nExiting...")
	sys.exit(0)

def create_png(src_tsv,targ_png,trav_so_far,dpi):
	#print(src_tsv+" "+targ_png+" ",trav_so_far)
	actual_location = trav_so_far[-1]
	zs = []
	smallest_z = 10

	f = open(src_tsv,"r")
	rows = f.read().split("\n")
	for y in range(len(rows)):
		row = []
		src_row = rows[y].split("\t")
		if len(src_row) in [0,1]: continue
		for item in src_row:
			val = float(item)
			row.append(val)
			if val<smallest_z and val!=0: smallest_z = val
		zs.append(row)

	smallest_scaled_z = 10
	for y in range(len(zs)):
		row = []
		for x in range(len(zs[y])):
			if zs[y][x]==0.0: val = smallest_z-(smallest_z/2.0)
			else: val = zs[y][x]
			zs[y][x] = val
			if val<smallest_scaled_z: smallest_scaled_z = val

	Z = np.array(zs)

	fig,ax = plt.subplots()

	png_title = targ_png.split("/")[1]+" | "+targ_png.split("/")[2]+" | "
	png_title += src_tsv.split("/")[-1].split(".")[0].split("-")[-1]

	fig.suptitle(png_title,fontsize=12,y=1.02)

	ax.set_xlabel("X Coordinate")
	ax.set_ylabel("Y Coordinate")

	ax.xaxis.set_label_position('top')
	ax.xaxis.tick_top()

	text_x = 1
	text_y = int(len(zs)/10)

	ax.annotate('Actual Location',xy=(actual_location[0],actual_location[1]),xytext=(text_x,text_y),
				arrowprops=dict(arrowstyle="-|>"), color='white') #ha="right",va="center")   #facecolor='white',shrink=0.01), color='white')

	#cax = ax.imshow(Z,cmap='plasma',norm=colors.LogNorm(vmin=smallest_scaled_z, vmax=1.0))
	cax = ax.imshow(Z,cmap='plasma')
	#cax = ax.imshow(Z,cmap='plasma',norm=colors.LogNorm(vmin=Z.min(),vmax=Z.max()))

	trav_xs = []
	trav_ys = []
	for s in trav_so_far:
		trav_xs.append(s[0])
		trav_ys.append(s[1])
	ax.plot(trav_xs,trav_ys,lw=0.5,c='black')

	ticks = [smallest_scaled_z,Z.max()]
	ylabels = ["%0.5f"%smallest_scaled_z,"%0.5f"%Z.max()]

	cbar = fig.colorbar(cax,ticks=ticks)
	cbar.ax.set_yticklabels(ylabels)

	#save_spot = save_base+"prediction-heatmap-"+str(iteration)+".png"
	fig.savefig(targ_png,bbox_inches='tight',dpi=dpi)
	plt.close()

def get_traversal_sequence(src_txt):
	f = open(src_txt,"r")
	lines = f.read().split("          0")[0].split("\n")
	seq = []
	for l in lines:
		if l not in [""," ","  "]:
			items = l.split(" ")
			#print items
			for i in items:
				if i in [""," ","  "]: continue
				x,y = i.split(",")
				x = int(x[1:])
				y = int(y[:-1])
				seq.append([x,y])
				if len(seq)==100:
					return seq
	return seq

def get_most_recent_data_dir():
	items = os.listdir(".")

	most_recent_name = None
	most_recent_secs = 0

	for item in items:
		if os.path.isdir(item):
			if item.find("exec_data")!=-1:
				secs = int(item.split("-")[1])
				if secs>most_recent_secs:
					most_recent_secs = secs
					most_recent_name = item

	if most_recent_name==None:
		print("ERROR: Must first generate data, none found.")
	return most_recent_name


def main():
	# generate execution data given data in Question_C folder
	regenerate_data = True
	if regenerate_data:
		print("--> Generating data...\n")
		start_time = time.time()
		src_dir = "../Question_C/data/"
		runtime_code = str(int(time.time()))

		num_grid_files = 1
		traversals_per_file = 1
		grid_width = 100
		grid_height = 100
		overall_total_score = 0

		for grid_idx in range(num_grid_files):
			map_dir = src_dir+"map_"+str(grid_idx)+"/"
			tsv = map_dir+"grid_"+str(grid_idx)+".tsv"
			v = viterbi_matrix(load_path=tsv)
			total_score = 0

			for trav_idx in range(traversals_per_file):
				trav_file = map_dir+"traversal_"+str(trav_idx)+".txt"
				save_dir = "exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/traversal_"+str(trav_idx)
				total_score += v.load_observations(trav_file,grid_width=grid_width,grid_height=grid_height,path=True,save_dir=save_dir,print_nothing=True)
				if trav_idx!=traversals_per_file-1: v.reload_conditions_matrix() # reset weights / bounds

			data_file = open("exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/meta.txt","w")
			data_file.write("Total Score: "+str(total_score)+"\n")
			data_file.write("Grid Width: "+str(grid_width)+"\n")
			data_file.write("Grid Height: "+str(grid_height)+"\n")
			data_file.close()
			overall_total_score+=total_score

		end_time = time.time()
		print("\nDone. Total time: "+str(end_time-start_time)[:7]+" seconds")
		print("Overall total score: "+str(overall_total_score)+"\n")


	generate_pngs_and_gifs = False
	# generate gifs and pngs for the data
	if generate_pngs_and_gifs:
		print("--> Generating images...\n")
		start_time = time.time()

		num_png = 0
		num_gif = 0
		dpi = 200

		sys.stdout.write("Generating .png and .gif files... ")
		sys.stdout.flush()
		src = get_most_recent_data_dir()+"/"
		map_dirs = os.listdir(src)
		for m in map_dirs:
			if os.path.isdir(src+m):
				trav_dirs = os.listdir(src+m)

				for t in trav_dirs:

					if os.path.isdir(src+m+"/"+t):
						sys.stdout.write("\r"+m+" - "+t+"                                                                     \n")
						sys.stdout.flush()

						data_files = os.listdir(src+m+"/"+t)

						# parse out the actual traversal sequence
						for d in data_files:
							if d.find("actual_traversal_sequence")!=-1:
								actual_traversal_sequence = get_traversal_sequence(src+m+"/"+t+"/"+d)
								break

						# create a new png for each prediction float matrix
						for d in data_files:
							if d.find("prediction-floats")!=-1:
								idx = d.split(".")[0].split("-")[2]
								#actual_loc = actual_traversal_sequence[int(idx)-1]
								trav_so_far = actual_traversal_sequence[:int(idx)]
								#create_png(src+m+"/"+t+"/"+d,src+m+"/"+t+"/"+"prediction-heatmap-"+idx+".png",actual_loc)
								create_png(src+m+"/"+t+"/"+d,src+m+"/"+t+"/"+"prediction-heatmap-"+idx+".png",trav_so_far,dpi)
								num_png+=1
								sys.stdout.write("\rGenerating .png and .gif files... GIF: "+str(num_gif)+", PNG: "+str(num_png)+"        ")
								sys.stdout.flush()

						# generate a gif from the newly created .png files
						make_gif(src+m+"/"+t)
						num_gif+=1

		sys.stdout.write("\nDone. Total time: "+str(time.time()-start_time)[:7]+" seconds\n")
		sys.stdout.flush()
		return

	dir_name = "exec_data-"+runtime_code
	os.rename(dir_name,dir_name+"-(complete)")

if __name__ == '__main__':
	main()

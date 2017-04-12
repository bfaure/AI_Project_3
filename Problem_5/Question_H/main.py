import sys
import time
import random
import os
from copy import deepcopy, copy
import imageio
import threading
import signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import mlab as ml
from matplotlib import colors
import numpy as np
import Cython, subprocess
import shutil, filecmp

use_cython = False


if use_cython:
	if os.path.exists("helpers.pyx"):
		if filecmp.cmp("../helpers.py","helpers.pyx")==False: # if they are not the same already
			shutil.copyfile("../helpers.py","helpers.pyx")
	else:
		shutil.copyfile("../helpers.py","helpers.pyx")

	val = subprocess.Popen('python setup.py build_ext --inplace',shell=True).wait()
else:
	sys.path.insert(0,"..")

from helpers import viterbi_matrix,viterbi_node, make_gif, create_png

def get_traversal_sequence(src_txt,first=False):
	if not first:	
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
						f.close()
						return seq
		f.close()
		return seq
	else:

		f = open(src_txt,"r")
		regions = f.read().split("~~~")[1:]
		seq = []
		for region in regions:
			items =  region.split("Sequence Probability: ")[1]
			#traj_probs.append(items.split("\n")[0][:6])
			lines = items.split("\n")[1:]
			for l in lines:
				if l.find("          0")!=-1: break
				if l in [""," ","  "]: continue
				elems = l.split(" ")
				for i in elems:
					if i in [""," ","  "]: continue
					x,y = i.split(",")
					x = int(x[1:])
					y = int(y[:-1])
					seq.append([x,y])
			break 
		f.close()
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

# parses a txt file to re-create the matrix in 2D list form
def resurrect_condition_matrix(src_txt):
	f = open(src_txt,"r")
	lines = f.read().split("          0")[1].split("\n")
	m = []
	for l in lines:
		if l.find("|")==-1: continue
		if l.split("|")[0]=="     ": continue
		l = l.replace(">"," ").replace("<"," ").replace("-"," ")
		row = []
		elems = l.split("|")[1].split(" ")
		for e in elems:
			if e not in ["N","H","T","B"]: continue
			#if len(e)

			row.append(e.strip())
		if len(row)>0: m.append(row)
	f.close()
	return m

def get_bounding_rect(x_set,y_set):
	x0,y0,x1,y1 = 1000,1000,0,0
	for xs,ys in zip(x_set,y_set):
		min_x = min(xs)
		max_x = max(xs)
		min_y = min(ys)
		max_y = max(ys)

		if min_x<x0: x0 = min_x 
		if max_x>x1: x1 = max_x 
		if min_y<y0: y0 = min_y  
		if max_y>y1: y1 = max_y 

	return x0,y0,x1,y1

def create_likely_trajectories_pic(src_txt,targ_png,conditions_matrix,dpi=750):
	f = open(src_txt)
	regions = f.read().split("~~~")[1:]

	traj_probs = [] # probability for each trajectory

	traj_x = [] # x coordinates
	traj_y = [] # y coordinates

	for region in regions:
		items =  region.split("Sequence Probability: ")[1]
		traj_probs.append(items.split("\n")[0][:6])
		seq_x = []
		seq_y = []
		lines = items.split("\n")[1:]
		for l in lines:
			if l.find("          0")!=-1: break
			if l in [""," ","  "]: continue
			elems = l.split(" ")
			for i in elems:
				if i in [""," ","  "]: continue
				x,y = i.split(",")
				x = int(x[1:])
				y = int(y[:-1])
				seq_x.append(x)
				seq_y.append(y)
		traj_x.append(seq_x)
		traj_y.append(seq_y)

	# variables used to store the bounding box of all sequences
	#x0,y0,x1,y1 = get_bounding_rect(traj_x,traj_y)

	iteration = targ_png.split("/")[-1].split("-")[2].split(".")[0]
	#iteration = src_txt.split("-")[2].split(".")[0]

	png_title = targ_png.split("/")[1]+" | "+targ_png.split("/")[2]+" | "
	png_title += "Iteration "+iteration

	fig,ax = plt.subplots()
	title = fig.suptitle(png_title,fontsize=10,y=0.99)
	#title.set_position()

	ax.set_xlabel("X Coordinate",fontsize=8)
	ax.set_ylabel("Y Coordinate",fontsize=8)

	for y in range(len(conditions_matrix)):
		for x in range(len(conditions_matrix[y])):
			ax.annotate(conditions_matrix[y][x],xy=(x-0.4,y-0.5),fontsize=3)

	line_handles = []

	# plot the 9 less likely sequences
	i=1
	for seq_x,seq_y in zip(reversed(traj_x[1:]),reversed(traj_y[1:])):
		line_label = str(traj_probs[i])
		line = ax.plot(seq_x,seq_y,lw=2,label=line_label)#,alpha=1.0-(float(i)/15))
		i+=1

	# plot the highest probability sequence in diff color
	line = ax.plot(traj_x[0],traj_y[0],lw=2.0,label=traj_probs[0])

	plt.legend(fontsize=6, bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

	#plt.xlim([x0-4,x1+4])
	#plt.ylim([y0-4,y1+4])

	#plt.xticks(range(x0,x1,2),fontsize=4)
	#plt.yticks(range(y0,y1,2),fontsize=4)

	plt.xlim([0,len(conditions_matrix[0])])
	plt.ylim([0,len(conditions_matrix)])

	plt.xticks(range(0,len(conditions_matrix[0]),5),fontsize=4)
	plt.yticks(range(0,len(conditions_matrix),5),fontsize=4)

	fig.savefig(targ_png,bbox_inches='tight',dpi=dpi)
	plt.close()

def get_sequence_score(actual,predicted):
	score = []
	for a,p in zip(actual,predicted):
		score.append(abs(a[0]-p[0])+abs(a[1]-p[1]))
	return score

def main():
	# generate execution data given data in Question_C folder
	regenerate_data = True
	if regenerate_data:
		print("--> Generating data...\n")

		start_time   = time.time()
		src_dir      = "../Question_C/data/"
		runtime_code = str(int(time.time()))

		num_grid_files      = 1
		traversals_per_file = 1
		grid_width          = 300
		grid_height         = 300
		overall_total_score = 0

		for grid_idx in range(num_grid_files):
			map_dir = src_dir+"map_"+str(grid_idx)+"/"
			tsv     = map_dir+"grid_"+str(grid_idx)+".tsv"
			v       = viterbi_matrix(load_path=tsv)
			total_score = 0
			for trav_idx in range(traversals_per_file):
				trav_file   = map_dir+"traversal_"+str(trav_idx)+".txt"
				save_dir    = "exec_data-"+runtime_code+"/map_"+str(grid_idx)+"/traversal_"+str(trav_idx)
				total_score += v.load_observations(trav_file,grid_width=grid_width,grid_height=grid_height,path=True,save_dir=save_dir,print_nothing=True)
				if trav_idx !=traversals_per_file-1: v.reload_conditions_matrix() # reset weights / bounds

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
	just_likely_traversals = True

	# generate gifs and pngs for the data
	if generate_pngs_and_gifs:
		print("--> Generating images...\n")
		start_time = time.time()

		num_png  = 0
		num_gif  = 0
		num_traj = 0
		dpi      = 200

		sys.stdout.write("Generating .png and .gif files... ")
		sys.stdout.flush()
		src = get_most_recent_data_dir()+"/"
		map_dirs = os.listdir(src)
		for m in map_dirs:
			if os.path.isdir(src+m):
				trav_dirs 	    = os.listdir(src+m)
				for t in trav_dirs:
					if os.path.isdir(src+m+"/"+t):
						sys.stdout.write("\r"+m+" - "+t+"                                                                        \n")
						sys.stdout.flush()
						data_files = os.listdir(src+m+"/"+t)

						# parse out the actual traversal sequence and the current condition matrix
						for d in data_files:
							if d.find("actual_traversal_sequence")!=-1:
								actual_traversal_sequence = get_traversal_sequence(src+m+"/"+t+"/"+d)
								cur_cond_matrix 		  = resurrect_condition_matrix(src+m+"/"+t+"/"+d)
								break

						# create pngs for the 10 most likely sequences (taken at 10, 50, 100 iterations)
						for d in data_files:
							if d.find("likely_trajectories")!=-1 and d.find(".txt")!=-1:
								traj_f = d.split(".")[0]+".png"
								create_likely_trajectories_pic( src+m+"/"+t+"/"+d , src+m+"/"+t+"/"+traj_f ,cur_cond_matrix )
								num_traj+=1


						# create a new png for each prediction float matrix
						for d in data_files:
							if d.find("prediction-floats")!=-1:
								idx = d.split(".")[0].split("-")[2]
								trav_so_far = actual_traversal_sequence[:int(idx)]
								if not just_likely_traversals: create_png(src+m+"/"+t+"/"+d,src+m+"/"+t+"/"+"prediction-heatmap-"+idx+".png",trav_so_far,dpi)
								num_png+=1
								sys.stdout.write("\rGenerating .png and .gif files... GIF: "+str(num_gif)+", PNG: "+str(num_png)+", Trajectories (PNG): "+str(num_traj)+"          ")
								sys.stdout.flush()

						# generate a gif from the newly created .png files
						if not just_likely_traversals: make_gif(src+m+"/"+t)
						num_gif+=1

		sys.stdout.write("\nDone. Total time: "+str(time.time()-start_time)[:7]+" seconds\n\n")
		sys.stdout.flush()

	# calculate scores for the final trajectories for all maps 
	score_trajectories = False 
	if score_trajectories:
		f = open("traj_100_scores.txt","w")

		scores 		= [0] * 100
		num_counted = 0

		sys.stdout.write("Calculating final trajectory scores... ")
		sys.stdout.flush()
		src = get_most_recent_data_dir()+"/"
		map_dirs = os.listdir(src)

		for m in map_dirs:
			if os.path.isdir(src+m):
				trav_dirs = os.listdir(src+m)
				for t in trav_dirs:
					if os.path.isdir(src+m+"/"+t):
						data_files = os.listdir(src+m+"/"+t)

						score = None 

						# parse out the actual traversal sequence and the current condition matrix
						for d in data_files:
							if d.find("actual_traversal_sequence")!=-1:
								actual_traversal_sequence = get_traversal_sequence(src+m+"/"+t+"/"+d)
								break


						for d in data_files:
							if d.find("likely_trajectories-100.txt")!=-1:
								predicted_traversal_sequence = get_traversal_sequence(src+m+"/"+t+"/"+d,first=True)
								score = get_sequence_score(actual_traversal_sequence,predicted_traversal_sequence)
								break 

						if score is not None:
							for i in range(100):
								scores[i] += score[i]
							num_counted+=1
						else:
							print("ERROR: here")

		for i in range(100):
			scores[i] = ( float(scores[i]) / float(num_counted))
			f.write("%0.5f\n"%scores[i])

		f.close()
		print("Done")
		print("Number counted: "+str(num_counted))

	#dir_name = "exec_data-"+runtime_code
	#os.rename(dir_name,dir_name+"-(complete)")

if __name__ == '__main__':
	main()



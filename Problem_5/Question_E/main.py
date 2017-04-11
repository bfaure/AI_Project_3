import sys
import time
import random
import os

from shutil import rmtree
from copy import deepcopy, copy

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import mlab as ml
from matplotlib import colors

import numpy as np

def get_overall_average_score(src_dir,targ_dir):
	sys.stdout.write("\nCalculating average scores... ")
	sys.stdout.flush()

	if src_dir[-1]!="/": src_dir+="/" 
	if targ_dir[-1]!="/": targ_dir+="/"

	# create directory if not already existant
	if not os.path.exists(targ_dir): os.makedirs(targ_dir)

	# check to ensure src_dir exists
	if not os.path.exists(src_dir):
		print("\nERROR: src_dir does not exist!")
		return

	# get all map_x directory names
	map_dirs = os.listdir(src_dir)

	# overall averages written to this file
	overall_avgs_file = open(targ_dir+"overall-avg_score.txt","w")

	# file to hold all scores
	all_scores_file = open(targ_dir+"all-scores.txt","w")

	# totals for all trials on all grids
	scores = [0] * 100

	# total number of trials read
	trials_read = 0

	for m in map_dirs:
		# write out individual averages to this file
		#cur_avgs_file = open(targ_dir+m+"-avg.txt","w")

		# current directory name
		cur_dir = src_dir+m+"/"

		# get items in map_x directory (mostly traversal_x dirs)
		cur_dir_items = os.listdir(cur_dir)

		for c in cur_dir_items:
			full_c = cur_dir+c

			# check if the item is a directory
			if os.path.isdir(full_c):
				trials_read+=1
				meta_loc = full_c+"/meta.txt"
				if not os.path.exists(meta_loc):
					print("\nERROR: Cannot locate "+meta_loc)
					return

				# open current meta.txt file for reading
				meta_f = open(meta_loc,"r")

				# split text into lines
				text = meta_f.read()
				lines = text.split("\n")

				# the index of the current score
				score_idx = 0

				total_score = 0

				# iterate over each line of meta.txt looking for path score
				for l in lines:
					if l.find("Predicted Path Score")!=-1:
						val = int(l.split(",")[0].split(": ")[1])
						total_score+= val
						scores[score_idx] += val
						score_idx += 1

				all_scores_file.write(m+" - "+c+" - score: \t"+str(total_score)+"\n")

				# close the meta.txt file
				meta_f.close()

	all_scores_file.close()

	sys.stdout.write("writing file... ")
	sys.stdout.flush()

	# get average for each step
	average_scores = []
	for s in scores:
		average_scores.append(float(s)/float(trials_read))

	# write out average scores
	for a in average_scores:
		overall_avgs_file.write(str(a))
		overall_avgs_file.write("\n")

	# close the overall_avgs_file
	overall_avgs_file.close()

	sys.stdout.write("done.\n")
	sys.stdout.flush()

	# create plot of average scores
	X = np.arange(1,len(average_scores)+1)
	Y = np.array(average_scores)

	fig,ax = plt.subplots()

	ax.bar(X,Y,width=0.9,color='blue')

	ax.set_xlabel("Iteration")
	ax.set_ylabel("Average Score")
	ax.set_title("Average Score, All 100 Experiments")

	title_fontsize = 20
	axis_label_fontsize = 20

	ax.title.set_fontsize(title_fontsize)
	ax.xaxis.label.set_fontsize(axis_label_fontsize)
	ax.yaxis.label.set_fontsize(axis_label_fontsize)

	plt.show()

	# return the list of averages
	return average_scores

def get_overall_correctness_probability(src_dir,targ_dir):
	sys.stdout.write("\nCalculating correctness probability... ")
	sys.stdout.flush()

	if src_dir[-1]!="/": src_dir+="/" 
	if targ_dir[-1]!="/": targ_dir+="/"

	# create directory if not already existant
	if not os.path.exists(targ_dir): os.makedirs(targ_dir)

	# check to ensure src_dir exists
	if not os.path.exists(src_dir):
		print("\nERROR: src_dir does not exist!")
		return

	# get all map_x directory names
	map_dirs = os.listdir(src_dir)

	# overall averages written to this file
	overall_avgs_file = open(targ_dir+"overall-correctness_probability.txt","w")

	# totals for all trials on all grids
	correct = [0] * 100

	# total number of trials read
	trials_read = 0

	for m in map_dirs:

		# current directory name
		cur_dir = src_dir+m+"/"

		# get items in map_x directory (mostly traversal_x dirs)
		cur_dir_items = os.listdir(cur_dir)

		for c in cur_dir_items:
			full_c = cur_dir+c

			# check if the item is a directory
			if os.path.isdir(full_c):
				trials_read+=1
				meta_loc = full_c+"/meta.txt"
				if not os.path.exists(meta_loc):
					print("\nERROR: Cannot locate "+meta_loc)
					return

				# open current meta.txt file for reading
				meta_f = open(meta_loc,"r")

				# split text into lines
				text = meta_f.read()
				lines = text.split("\n")

				# the index of the current score
				score_idx = 0

				# iterate over each line of meta.txt looking for path score
				for l in lines:
					if l.find("Predicted Path Score")!=-1:
						val = int(l.split(",")[0].split(": ")[1])
						if val==0: correct[score_idx]+=1
						score_idx += 1

				# close the meta.txt file
				meta_f.close()

	sys.stdout.write("writing file... ")
	sys.stdout.flush()

	# get average for each step
	average_scores = []
	for s in correct:
		average_scores.append(float(s)/float(trials_read))

	# write out average scores
	for a in average_scores:
		overall_avgs_file.write(str(a))
		overall_avgs_file.write("\n")

	# close the overall_avgs_file
	overall_avgs_file.close()

	sys.stdout.write("done.\n")
	sys.stdout.flush()

	# create plot of average scores
	X = np.arange(1,len(average_scores)+1)
	Y = np.array(average_scores)

	fig,ax = plt.subplots()

	ax.bar(X,Y,width=0.9,color='blue')

	ax.set_xlabel("Iteration")
	ax.set_ylabel("Correct Prediction Probability")
	ax.set_title("Average Correct Prediction Probability, All 100 Experiments")

	title_fontsize = 20
	axis_label_fontsize = 20

	ax.title.set_fontsize(title_fontsize)
	ax.xaxis.label.set_fontsize(axis_label_fontsize)
	ax.yaxis.label.set_fontsize(axis_label_fontsize)

	plt.show()

	# return the list of averages
	return average_scores


def get_most_recent_data_dir(parent_dir):
	items = os.listdir(parent_dir)
	most_recent_name = None
	most_recent_secs = 0
	for item in items:
		if os.path.isdir(parent_dir+"/"+item) and item.find("exec_data")!=-1:
			secs = int(item.split("-")[1])
			if secs>most_recent_secs:
				most_recent_secs = secs
				most_recent_name = item
	if most_recent_name==None:
		print("ERROR: Must first generate data, none found.")
	return most_recent_name

def main():
	parent_dir = "../Question_D"
	src_dir = parent_dir+"/"+get_most_recent_data_dir(parent_dir)
	targ_dir = "cleaned_data/"

	# get and plot overall average scores at each iteration (all maps/traversals)
	avgs = get_overall_average_score(src_dir,targ_dir)

	# get and plot overall average probability of correct prediction at each iteration (all maps/travs)
	probs = get_overall_correctness_probability(src_dir,targ_dir)

	# get the average score

if __name__ == '__main__':
	main()

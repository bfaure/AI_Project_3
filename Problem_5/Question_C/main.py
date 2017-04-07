import sys
import time
import random
import os

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

class grid:
	def __init__(self):
		self.init_map()

	def init_map(self):
		# 50% normal, 20% highway, 20% hard to traverse, 10% blocked
		self.num_cols = 1000
		self.num_rows = 1000

		self.data = []
		for _ in range(self.num_rows):
			row = []
			for _ in range(self.num_cols):
				cond_type = random.randint(1,100) # generate num from [1...100]
				if   cond_type <= 50: row.append("N")
				elif cond_type <= 70: row.append("H")
				elif cond_type <= 90: row.append("T")
				else: 				  row.append("B")
			self.data.append(row)

	def get_starting_point(self):
		while True:
			x = random.randint(1,self.num_cols-1)
			y = random.randint(1,self.num_rows-1)
			if self.data[y][x] is not "B": return [x,y]

	def add_column(self,side):
		for y in range(self.num_rows):
			cond_type = random.randint(1,100) # generate num from [1...100]
			if   cond_type <= 50: cond = "N"
			elif cond_type <= 70: cond = "H"
			elif cond_type <= 90: cond = "T"
			else: 				  cond = "B"

			if side=="left": self.data[y].insert(0,cond)
			if side=="right": self.data[y].append(cond)
		self.num_cols+=1

	def add_row(self,side):
		new_row = []
		for x in range(self.num_cols):
			cond_type = random.randint(1,100) # generate num from [1...100]
			if   cond_type <= 50: cond = "N"
			elif cond_type <= 70: cond = "H"
			elif cond_type <= 90: cond = "T"
			else: 				  cond = "B"
			new_row.append(cond)

		if side=="bottom": self.data.append(new_row)
		if side=="top":	   self.data.insert(0,new_row)

	def get_item(self,x,y):
		# if we need to dynamically add a new row or column
		if x<0: self.add_column("left")
		elif x>self.num_cols-1: self.add_column("right")

		if y<0: self.add_row("top")
		elif y>self.num_rows-1: self.add_row("bottom")

		return self.data[y][x]

def generate_data(num_maps=10,num_per_map=10,sequence_length=100):
	targ_dir = "data"
	if not os.path.exists(targ_dir): os.makedirs(targ_dir)

	directions = ["U","L","D","R"] # up, left, down, right
	conditions = ["N","H","T","B"] # normal, highway, hard to traverse, blocked

	for i in range(num_maps):
		map_dir = targ_dir+"/map"+str(i)
		if not os.path.exists(map_dir): os.makedirs(map_dir)

		# generate ground truth state for 100 locations
		ground_truth = [] # list of actual conditions in map
		for _ in range(sequence_length):
			cond_type = random.randint(1,100) # generate num from [1...100]

			if cond_type <= 50:   groud_truth.append("N")
			elif cond_type <= 70: ground_truth.append("H")
			elif cond_type <= 90: ground_truth.append("T")
			else: 				  ground_truth.append("B")






def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	generate_data()

if __name__ == '__main__':
	main()

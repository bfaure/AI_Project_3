import os
import sys

import random

class grid(object):
	def __init__(self,width=3,height=3):
		self.width = width
		self.height = height
		self.init_cells()
		self.init_current_location()

	# initializes the list of cells and cell values
	def init_cells(self):
		self.cells = []

		default = True 
		if default:
			self.cells = ["H","H","T","N","N","N","N","B","H"]
			return 

	# initialize the current location member variable, chooses among non-blocked cells
	def init_current_location(self):
		while True:
			# get a possible x starting location
			attempted_x = random.randint(1,self.width)
			# get a possible y starting location
			attempted_y = random.randint(1,self.height)
			# if the spot isnt blocked, break
			if self.get_cell_value(attempted_x,attempted_y) is not "B": break
		# set the current location
		self.current_location = [attempted_x,attempted_y]

	# if zero==False:
	# x: [1,2,3,...,self.width]
	# y: [1,2,3,...,self.height]
	# else:
	# x: [0,1,2,...,self.width-1]
	# y: [0,1,2,...,self.height-1]
	def get_cell_value(self,x,y,zero=False):
		# adjust the coords if not starting at zero
		if not zero:
			x += -1
			y += -1
		return self.cells[(self.width*y)+x]

	# if zero==False:
	# x: [1,2,3,...,self.width]
	# y: [1,2,3,...,self.height]
	# else:
	# x: [0,1,2,...,self.width-1]
	# y: [0,1,2,...,self.height-1]
	def get_cell_value_estimate(self,x,y,zero=False):
		# all possible cell values
		possible_values = ["N","H","T"]
		# get the actual cell value (from set 'possible_values')
		real_value = self.get_cell_value(x,y,zero)
		# remove the actual value from the list of possible fake values
		del possible_values[possible_values.index(real_value)]
		# get a random integer from 1 to 100
		action = random.randint(1,100)
		# return the real value
		if action<=90: return real_value
		# return a fake value
		return random.choice(possible_values)

	# if zero==False:
	# x: [1,2,3,...,self.width]
	# y: [1,2,3,...,self.height]
	# else:
	# x: [0,1,2,...,self.width-1]
	# y: [0,1,2,...,self.height-1]
	# 
	# checks if the input coordinates [x,y] are within the bounds stated above
	def is_within_bounds(self,x,y,zero=False):
		# adjust the coords if not starting at zero
		if not zero:
			x += -1
			y += -1

		# check if the x or y coord is out of bounds
		if x<0 or x>=(self.width): 	return False 
		if y<0 or y>=(self.height): return False 

		# if we get here, we know coords are within bounds
		return True

	# action in set ["Up","Left","Down","Right"]
	def move(self,action):

		# get the current location
		current_location = self.current_location

		# set the proposed next location
		if action=="Up": 	proposed_location = [current_location[0],current_location[1]-1]
		if action=="Down": 	proposed_location = [current_location[0],current_location[1]+1]
		if action=="Left": 	proposed_location = [current_location[0]-1,current_location[1]]
		if action=="Right": proposed_location = [current_location[0]+1,current_location[1]]

		# if the proposed move will bring us out of bounds
		if not self.is_within_bounds(proposed_location[0],proposed_location[1]): 
			return self.get_cell_value_estimate(current_location[0],current_location[1])

		# get a random integer from 1 to 10
		action = random.randint(1,10)

		# with 90% probability we should return the estimate of the new location conditions
		if action<=9: return self.get_cell_value_estimate(proposed_location[0],proposed_location[1])

		# with a 10% probability return estimate of current location conditions
		return self.get_cell_value_estimate(current_location[0],current_location[1])

def create_prediction_matrix(values=["H","H","T","N","N","N","N","B","H"]):
	matrix = []
	for y in range(3):
		row = []
		for x in range(3):
			value_idx = (3*y)+x 
			if values[value_idx] is not "B":
				row.append(float(1.0/8.0))
			else:
				row.append(0.0)
		matrix.append(row)
	return matrix

def print_prediction_matrix(matrix):
	print("\n\n_________________________________________________")
	for row in matrix:
		sys.stdout.write("| ")
		for item in row:
			desired_item_size = 10
			real_item_size = len(str(item))
			sys.stdout.write(str(item)[:10])
			if real_item_size<desired_item_size:
				for _ in range(desired_item_size-real_item_size):
					sys.stdout.write(" ")
					
			if row.index(item) is not len(row)-1:
				sys.stdout.write(" \t| ")
			else:
				sys.stdout.write("\n")
		if matrix.index(row) is not len(matrix)-1:
			sys.stdout.write("\n_________________________________________________\n")
		else:
			sys.stdout.write("\n_________________________________________________\n\n")

# compute the probability of where we are in grid world given inputs 'actions' and 
# subsequent sensor readings 'readings'
def predict_location(actions,readings):

	pred_matrix = create_prediction_matrix()
	print_prediction_matrix(pred_matrix)

	#for cur_action,cur_reading in zip(actions,readings):






def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]
	predict_location(actions,readings)

if __name__ == '__main__':
	main()
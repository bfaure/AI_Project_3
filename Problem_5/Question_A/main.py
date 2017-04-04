import os
import sys
import time

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

# creates a new 3x3 prediction matrix given the provided conditions
def create_prediction_matrix(values=["H","H","T","N","N","N","N","B","H"]):
	matrix = []
	for y in range(3):
		row = []
		for x in range(3):
			value_idx = (3*y)+x 
			if values[value_idx] is not "B":
				row.append(float(1.0/8.0)) # 1/8 constant probability
			else: # if the current location is a blocked cell
				row.append(0.0)
		matrix.append(row)
	return matrix

# desired_item_size: column width in characters
#
# prints out either a prediction or condition matrix
def print_matrix(matrix,desired_item_size=20):
	delim_line = ''.join("_" for _ in range(3*desired_item_size+10))
	sys.stdout.write("\n"+delim_line+"\n")
	for row in matrix:
		sys.stdout.write("| ")
		for item in row:
			real_item_size = len(str(item))
			sys.stdout.write(str(item)[:desired_item_size])
			if real_item_size<desired_item_size:
				for _ in range(desired_item_size-real_item_size):
					sys.stdout.write(" ")
					
			if row.index(item) is not len(row)-1:
				sys.stdout.write(" | ")
			else:
				sys.stdout.write(" |")
		if matrix.index(row) is not len(matrix)-1:
			sys.stdout.write("\n"+delim_line+"\n")
		else:
			sys.stdout.write("\n"+delim_line+"\n")

# creates a new 3x3 condition matrix given the provided conditions
def create_condition_matrix(values=["H","H","T","N","N","N","N","B","H"]):
	matrix = []
	for y in range(3):
		row = []
		for x in range(3):
			value_idx = (3*y)+x 
			row.append(values[value_idx])
		matrix.append(row)
	return matrix

# prints out information about the current step, i.e. the current
# condition matrix (doesn't change over steps), the current prediction
# matrix (adjusted on each step), the current reported action, and the
# current reported reading
def print_current_state(condition_matrix=None,pred_matrix=None,move_index=0,cur_action=None,cur_reading=None,desired_item_size=20):
	delim_line = ''.join("=" for _ in range(3*desired_item_size+10))
	if move_index==0:
		print("\n"+delim_line)
		print("Initial State")
	else:
		print(delim_line)
		print("\nMove Index:\t\t"+str(move_index))
		print("Reported Action:\t("+str(cur_action)+", "+str(cur_reading)+")")

	if condition_matrix is not None:
		sys.stdout.write("\nCondition Matrix:")
		print_matrix(condition_matrix,5)

	sys.stdout.write("\nPrediction Matrix")
	print_matrix(pred_matrix,desired_item_size)
	print("\n"+delim_line)

# returns the element-wise sum of the input 3x3 matrix
def get_matrix_sum(matrix):
	matrix_sum = 0
	for y in range(3):
		for x in range(3):
			matrix_sum += float(matrix[y][x])
	return matrix_sum

# divides each elements of the input 3x3 matrix by its matrix sum
def normalize_matrix(matrix):
	matrix_sum = float(get_matrix_sum(matrix))
	for y in range(3):
		for x in range(3):
			matrix[y][x] = float(matrix[y][x])/matrix_sum
	return matrix

# compute the probability of where we are in grid world given inputs 'actions' and 
# subsequent sensor readings 'readings'
def predict_location(actions,readings):

	condition_matrix = create_condition_matrix()
	pred_matrix = create_prediction_matrix()
	print_current_state(condition_matrix,pred_matrix)

	move_index = 1
	for cur_action,cur_reading in zip(actions,readings):

		# set probabilities given the reported reading compared to state values
		for y in range(3):
			for x in range(3):
				# never in this state
				if condition_matrix[y][x]=="B": pred_matrix[y][x] = 0.0

				# in this state with 0.9 confidence (same as reading)
				elif condition_matrix[y][x]==cur_reading: pred_matrix[y][x] *= 0.9
		
				# in this state only if there was a mis-reading of the cur_reading
				else: pred_matrix[y][x] *= 0.1
		
		# set probabilities given the reported movement (cur_action) compared to condition neighbors
		#
		# iterate over all possible current locations
		for y in range(3):
			for x in range(3):

				# if the current location is a blocked cell, it will have already been set to P = 0.0
				if condition_matrix[y][x]=="B": continue

				# if the reported action was a translation to the right
				if cur_action=="Right":
					
					# if the current location is in the middle or right columns
					if x==1 or x==2: pred_matrix[y][x] *= 0.9

					# if the current location is in the left column
					if x==0:
						# if a right translation could be prevented due to a blocked cell to the right
						if condition_matrix[y][x+1]=="B": pred_matrix[y][x] *= 0.9
						else: 							  pred_matrix[y][x] *= 0.1

				if cur_action=="Left":
					if x==0 or x==1: pred_matrix[y][x] *= 0.9
					if x==2:
						if condition_matrix[y][x-1]=="B": pred_matrix[y][x] *= 0.9
						else: 							  pred_matrix[y][x] *= 0.1

				if cur_action=="Up":
					if y==0 or y==1: pred_matrix[y][x] *= 0.9
					if y==2:
						if condition_matrix[y-1][x]=="B": pred_matrix[y][x] *= 0.9
						else: 							  pred_matrix[y][x] *= 0.1

				if cur_action=="Down":
					if y==1 or y==2: pred_matrix[y][x] *= 0.9
					if y==0:
						if condition_matrix[y+1][x]=="B": pred_matrix[y][x] *= 0.9
						else:							  pred_matrix[y][x] *= 0.1
				
		# now need to normalize all values by dividing by probability sum
		pred_matrix = normalize_matrix(pred_matrix)

		# print out current state information
		print_current_state(pred_matrix=pred_matrix,move_index=move_index,cur_action=cur_action,cur_reading=cur_reading)

		move_index+=1

def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	#actions = ["Right","Down","Down","Down","Down"]
	#readings = ["N","H","H","H","H"]

	predict_location(actions,readings)

if __name__ == '__main__':
	main()
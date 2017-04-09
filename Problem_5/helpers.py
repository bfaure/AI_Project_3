import sys
import time
import random
import os
from copy import deepcopy, copy

def get_sequence_bounds(sequence):
	y_max, x_max, x_min, y_min = None, None, None, None
	for x,y in sequence:
		if x_max==None: x_max = x
		if x_min==None: x_min = x
		if y_max==None: y_max = y
		if y_min==None: y_min = y

		if x<x_min: x_min = x
		if x>x_max: x_max = x
		if y<y_min: y_min = y
		if y>y_max: y_max = y
	return x_max,y_max,x_min,y_min

class viterbi_node:
	def __init__(self):
		self.value = ""
		self.parent = None

		self.best_path = None
		self.best_path_cost = None

class viterbi_matrix:
	def __init__(self,num_rows=3,num_cols=3,values=["H","H","T","N","N","N","N","B","H"],load_path=None):
		self.temp_anc_matrix = None # debugging

		# Questions A & B...
		if load_path is None:
			self.actual_traversal_path = None
			self.num_rows = num_rows
			self.num_cols = num_cols
			self.values = values
			self.init_conditions_matrix(values)
			self.init_prediction_matrix()

		# Question D...
		else:
			self.load_path = load_path
			self.load_conditions_matrix()

	# loads a grid world matrix from a .tsv file
	def load_conditions_matrix(self):
		# ensure the .tsv file exists
		if not os.path.exists(self.load_path):
			print("\nWARNING: Could not find .tsv file "+self.load_path+"\n")
			return

		sys.stdout.write("\nLoading \""+self.load_path+"\"... ")
		f = open(self.load_path,"r")
		text = f.read()
		lines = text.split("\n")

		self.conditions_matrix = []
		self.num_cols = None

		for line in lines:
			elems = line.split("\t")
			if len(elems)<2: break # break if line doesn't make sense to exist
			row = []
			for elem in elems:
				new_node = viterbi_node()
				new_node.value = elem
				row.append(new_node)
			if self.num_cols is None: self.num_cols = len(row)
			else:
				if len(row) != self.num_cols:
					sys.stdout.write("warning, ")
					row = row[:self.num_cols]
			self.conditions_matrix.append(row)
		self.num_rows = len(self.conditions_matrix)
		sys.stdout.write("success. ")
		sys.stdout.write("Rows: "+str(self.num_rows)+", Columns: "+str(self.num_cols)+"\n")
		f.close()

	# reloads the prior conditions_matrix from same file (resets current weights)
	def reload_conditions_matrix(self):
		self.load_conditions_matrix()

	# loads in an observation file (.txt)
	# if buffer_size=None then none of the original conditions_matrix will be trimmed, if
	# buffer_size is a positive integer then we will find the outer bounds of the actual
	# traversal path and add buffer_size to those bounds, translate the actual traversal path
	# accordingly and trim the conditions_matrix down to said bounds (the conditions_matrices are
	# originally 500x500 but the agent rarely makes if very far to justify that large of a search
	# space) - see self.adjust_environment_bounds()
	def load_observations(self,observation_path,grid_buffer_size=None,path=False,method="default"):

		# ensure the file exists
		if not os.path.exists(observation_path):
			print("\nWARNING: Could not find "+observation_path+"\n")
			return

		sys.stdout.write("Loading \""+observation_path+"\"... ")
		f = open(observation_path,"r")
		text = f.read()
		lines = text.split("\n")

		self.actual_traversal_path = [] # loaded from provided file

		self.observed_actions  = [] # to be filled as we encounter more actions
		self.observed_readings = [] # to be filled as we encounter more readings

		self.queued_actions  = [] # filled with all initial actions (loaded from file)
		self.queued_readings = [] # filled with all initial readings (loaded from file)

		self.start_location = None
		self.show_all       = False
		current_item_type   = None

		for line in lines:
			if line.find("start_location")!=-1:
				x,y = line.split(" - ")[1].split(",")
				self.start_location = [int(x.replace("(","")),int(y.replace(")",""))]

			elif line.find("~")!=-1:
				if current_item_type==None: current_item_type = "actual_traversal_path"
				elif current_item_type=="actual_traversal_path": current_item_type = "queued_actions"
				elif current_item_type=="queued_actions": current_item_type = "queued_readings"
				else: break
			else:
				if current_item_type=="actual_traversal_path":
					x,y = line.split(",")
					self.actual_traversal_path.append([int(x.replace("(","")),int(y.replace(")",""))])
				else: self.__dict__[current_item_type].append(line)

		if len(self.queued_actions)!=len(self.queued_readings):
			sys.stdout.write("failure: observations invalid\n")
			return

		sys.stdout.write("success. ")
		print("Path: "+str(len(self.actual_traversal_path))+", Observations: "+str(len(self.queued_readings)))

		self.transition_matrices = []

		# trim the environment down to a smaller area containing the real movement of the agent
		if grid_buffer_size is not None: self.adjust_environment_bounds(grid_buffer_size)

		self.init_prediction_matrix()

		self.print_transition       = True
		self.print_condition  	    = True
		self.print_actual_traversal = True

		self.move_index = 1
		self.current_predicted_length = 1

		print_size = 6
		self.print_current_state(print_size)

		max_limit = len(self.queued_readings)
		#max_limit = 10

		# if we are performing path approximations, print the full actual path on each iteration
		self.print_full_traversal = False if path else True

		for self.cur_action,self.cur_reading in zip(self.queued_actions[:max_limit],self.queued_readings[:max_limit]):

			# add the observation
			self.add_observation()

			# update weights given the new information
			self.update_weights()

			# print out current state information
			self.print_current_state(print_size)

			# if calculating the most likely sequences as well...
			if path:
				# get the current sequence
				pred_seq,pred_prob = self.get_predicted_sequence()

				self.current_predicted_length = len(pred_seq)

				# print out current sequence
				sys.stdout.write("\n Current PREDICTED Agent Traversal (Probability = ")
				sys.stdout.write("%0.5f" % pred_prob[-1])
				sys.stdout.write(")...\n")
				self.print_single_sequence(pred_seq)

		if path:
			final_loc = self.actual_traversal_path[-1]
			print("\nActual Final Location: ("+str(final_loc[0])+", "+str(final_loc[1])+")")
			print("Actual traversal path length: "+str(len(self.actual_traversal_path)))

			pred_final_loc = pred_seq[-1]
			print("\nPredicted Final Location: ("+str(pred_final_loc[0])+", "+str(pred_final_loc[1])+")")
			print("Predicted traversal path length: "+str(len(pred_seq)))
		print("\n")

	# various methods for ensuring data validity
	def check_validity(self):
		# check that all prediction_matrices are of the same size as the conditions_matrix
		for m in self.prediction_matrices:
			if len(m)!=len(self.conditions_matrix): print("WARNING: Invalid y bounds")
			for row_idx in range(len(m)):
				if len(m[row_idx])!=len(self.conditions_matrix[row_idx]): print("WARNING: Invalid x bounds")

	# given that the actual path taken by the agent rarely fills anywhere near the entire conditions_matrix
	# we can choose to trim the conditions_matrix down to fit the range of the path taken
	def adjust_environment_bounds(self,buffer_size=10):
		sys.stdout.write("\nAdjusting bounds... ")
		x_max,y_max,x_min,y_min = get_sequence_bounds(self.actual_traversal_path)

		preferred_x_max = x_max+buffer_size if (x_max+buffer_size)<self.num_cols else self.num_cols-1
		preferred_y_max = y_max+buffer_size if (y_max+buffer_size)<self.num_rows else self.num_rows-1
		preferred_x_min = x_min-buffer_size if (x_min-buffer_size)>=0 			 else 0
		preferred_y_min = y_min-buffer_size if (y_min-buffer_size)>=0 			 else 0

		sys.stdout.write("x_off: "+str(preferred_x_min)+", y_off: "+str(preferred_y_min)+" ")

		self.start_location = [self.start_location[0]-preferred_x_min,self.start_location[1]-preferred_y_min]

		self.translate_actual_path(-1*preferred_x_min,-1*preferred_y_min)
		self.trim_conditions_matrix(preferred_x_max,preferred_y_max,preferred_x_min,preferred_y_min)

		self.num_rows = preferred_y_max-preferred_y_min+1
		self.num_cols = preferred_x_max-preferred_x_min+1
		sys.stdout.write("rows: "+str(self.num_rows)+", cols: "+str(self.num_cols)+"\n")

	# translates the coordinates of the actual traversal path
	def translate_actual_path(self,x_offset,y_offset):
		for i in range(len(self.actual_traversal_path)):
			self.actual_traversal_path[i] = [self.actual_traversal_path[i][0]+x_offset,self.actual_traversal_path[i][1]+y_offset]

	# trims the conditions matrix so it fits in specified bounds
	def trim_conditions_matrix(self,x_max,y_max,x_min,y_min):
		self.conditions_matrix = self.conditions_matrix[y_min:y_max+1]
		for i in range(len(self.conditions_matrix)):
			self.conditions_matrix[i] = self.conditions_matrix[i][x_min:x_max+1]

	# returns the number of blocked cells in the self.conditions_matrix
	def get_num_blocked_cells(self):
		num_blocked = 0
		for y in range(self.num_rows):
			for x in range(self.num_cols):
				if self.conditions_matrix[y][x].value=="B": num_blocked+=1
		return num_blocked

	# creates a new condition matrix given the provided conditions
	def init_conditions_matrix(self,conditions=None):
		if conditions is not None: self.values = conditions
		self.conditions_matrix = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = self.values[y*self.num_cols+x]
				row.append(new_node)
			self.conditions_matrix.append(row)

	# clears the current prediction_matrices list and creates a new prediction matrix
	# to be inserted at the first location in the list, all initial probabilities are set to 1/8
	# besides the location containing "B" which has it's probability set to 0
	def init_prediction_matrix(self,start_location=None):
		if start_location==None:
			init_probability = 1.0/float(self.num_rows*self.num_cols-self.get_num_blocked_cells())
			self.prediction_matrices = []
			cells = []
			for y in range(self.num_rows):
				row = []
				for x in range(self.num_cols):
					new_node = viterbi_node()
					new_node.parent = None
					new_node.coords = [x,y]
					new_node.value = init_probability if self.conditions_matrix[y][x].value!="B" else 0.0
					#if self.conditions_matrix[y][x].value=="B": new_node.value = 0.0
					row.append(new_node)
				cells.append(row)
			self.prediction_matrices.append(cells)
		else:
			self.prediction_matrices = []
			cells = []
			for y in range(self.num_rows):
				row = []
				for x in range(self.num_cols):
					new_node = viterbi_node()
					new_node.parent = None
					new_node.coords = [x,y]
					new_node.value = 1.0 if x==start_location[0] and y==start_location[1] else 0.0
					row.append(new_node)
				cells.append(row)
			self.prediction_matrices.append(cells)

	# add a single observed action and observed reading while also incrementing the move index
	def add_observation(self):
		self.observed_actions.append(self.cur_action)
		self.observed_readings.append(self.cur_reading)
		self.move_index+=1

	# creates a new prediction matrix given the provided conditions
	def empty_prediction_matrix(self):
		cells = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = 0.0
				new_node.parent = None
				new_node.coords = [x,y]
				row.append(new_node)
			cells.append(row)
		return cells

	# cur_action: reported movement direction in this step
	# cur_reading: reported reading in this step
	# condition_matrix: condition matrix (strings)
	# pred_matrix: current prediction matrix (floats)
	#
	# return: updated prediction matrix (given a new move)
	def update_weights(self):

		# if we have already updated the weights for the most recent observation
		if len(self.observed_actions)+1==len(self.prediction_matrices):
			print("ERROR: update_weights()")
			return

		# set local variables to hold the last observed action and observed reading
		cur_action 	= self.cur_action
		cur_reading = self.cur_reading

		old_pred_matrix 	= self.prediction_matrices[-1] # get the last prediction matrix
		transition_matrix 	= self.empty_prediction_matrix() # create new matrix
		condition_matrix 	= self.conditions_matrix # matrix holding cell conditions

		# set probabilities given the reported reading compared to state values
		#
		# iterate over all possible current locations
		for y in range(self.num_rows):
			for x in range(self.num_cols):

				# can't ever be in a blocked cell, set probability to zero
				if condition_matrix[y][x].value=="B": transition_matrix[y][x].value = 0.0

				# if the reported cell type, likelyhood of cur_reading being correct is 90%
				elif condition_matrix[y][x].value==cur_reading: transition_matrix[y][x].value = 0.9

				# in this state only if there was a mis-reading of the cur_reading
				else: transition_matrix[y][x].value = 0.1

		# NEW: normalizing the transition_matrix (effectively the emission matrix for current step)
		#transition_matrix = self.normalize_matrix(transition_matrix)

		if len(self.transition_matrices)>0:
			prior_transition_matrix = self.transition_matrices[-1]
		else:
			prior_transition_matrix = None

		ancestors = []

		# set probabilities given the reported movement (cur_action) compared to condition neighbors
		#
		# iterate over all possible current locations
		for y in range(self.num_rows):
			ancestor_row = []

			for x in range(self.num_cols):

				#if prior_transition_matrix!=None:
				#	continue

				# REMOVE THIS
				# if the current location is a blocked cell, it will have already been set to P = 0.0
				#if condition_matrix[y][x].value=="B": continue

				# if the reported action was a translation to the right
				if cur_action in ["Right","R"]:
					# if in an inner column
					if x>0 and x<=self.num_cols-1:
						# if the cell to the left is blocked (i.e. we couldn't have come from left)
						if condition_matrix[y][x-1].value=="B":
							ancestor_row.append([x,y])
							#transition_matrix[y][x].value *= 0.1
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1
						# we could have legally made the reported move
						else:
							ancestor_row.append([x-1,y])
							#transition_matrix[y][x].value *= 0.9
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9
							#if prior_transition_matrix!=None:
							#	prior

					# if in the left column
					if x==0:
						# if a right translation could be prevented due to a blocked cell to the right
						if condition_matrix[y][x+1].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9
						# only could be here if the reported action was not actually performed
						else:
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1

				# if the reported action was a translation to the left
				if cur_action in ["Left","L"]:
					# if in an inner column
					if x>=0 and x<self.num_cols-1:
						# if the right cell is blocked (i.e. we couldn't have come from right)
						if condition_matrix[y][x+1].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1
						# we could have legally made the reported move
						else:
							ancestor_row.append([x+1,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9

					# if in the right column
					if x==self.num_cols-1:
						# if a left translation could be prevented due to a blocked cell to the left
						if condition_matrix[y][x-1].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9
						# only here if the reported action was not actually performed
						else:
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1

				# if the reported action was a translation up
				if cur_action in ["Up","U"]:
					# if in an inner row
					if y>=0 and y<self.num_rows-1:
						if condition_matrix[y+1][x].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1
						else:
							ancestor_row.append([x,y+1])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9

					# if in the bottom row
					if y==self.num_rows-1:
						if condition_matrix[y-1][x].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9
						else:
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1

				# if the reported action was a translation down
				if cur_action in ["Down","D"]:
					# if in an inner row
					if y>0 and y<=self.num_rows-1:
						if condition_matrix[y-1][x].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1
						else:
							ancestor_row.append([x,y-1])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9


					# if in the top row
					if y==0:
						if condition_matrix[y+1][x].value=="B":
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.9
							transition_matrix[y][x].value *= 0.9
						else:
							ancestor_row.append([x,y])
							#if prior_transition_matrix==None: transition_matrix[y][x].value *= 0.1
							transition_matrix[y][x].value *= 0.1

			ancestors.append(ancestor_row)

		#print("\n transition_matrix... (before normalization)")
		#self.print_matrix(transition_matrix,10)

		temp_anc_matrix = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = "("+str(ancestors[y][x][0])+","+str(ancestors[y][x][1])+")"
				row.append(new_node)
			temp_anc_matrix.append(row)
		#print("\nAncestors Matrix: ")
		self.temp_anc_matrix = temp_anc_matrix
		#self.print_matrix(temp_anc_matrix,6)

		# NEW: normalizing the transition_matrix (effectively the emission matrix for current step)
		transition_matrix = self.normalize_matrix(transition_matrix)

		# add new transition matrix to list of prior transition matrices
		self.transition_matrices.append(deepcopy(transition_matrix))

		# create new prediction matrix by multiplying each element of the newly created transition matrix
		# by the element in the same location of the prior prediction matrix
		new_pred_matrix = self.resolve_prediction_matrix(transition_matrix,old_pred_matrix,ancestors)

		# normalize the new prediction matrix
		new_pred_matrix = self.normalize_matrix(new_pred_matrix)
		#new_pred_matrix = deepcopy(new_pred_matrix)

		'''
		# point each element of the new prediction matrix to it's ancestor in the prior prediction matrix
		if len(self.prediction_matrices)==0:
			for y in range(self.num_rows):
				for x in range(self.num_cols):
					new_pred_matrix[y][x].parent = None
					new_pred_matrix[y][x].transition_prob = transition_matrix[y][x].value
		else:
			for y in range(self.num_rows):
				for x in range(self.num_cols):
					new_pred_matrix[y][x].parent = self.prediction_matrices[-1][y][x]
					new_pred_matrix[y][x].transition_prob = transition_matrix[y][x].value
		'''

		# add new prediction matrix to list
		self.prediction_matrices.append(new_pred_matrix)

	# creates a new matrix provided a prior prediction matrix and a transition matrix
	def resolve_prediction_matrix(self,transition_matrix,old_pred_matrix,ancestors=None):
		if ancestors==None:
			new_pred_matrix = []
			for y in range(self.num_rows):
				row = []
				for x in range(self.num_cols):
					new_node = viterbi_node()
					new_node.coords = [x,y]
					new_node.value = float(transition_matrix[y][x].value)*float(old_pred_matrix[y][x].value)
					new_node.parent = None
					row.append(new_node)
				new_pred_matrix.append(row)
			return new_pred_matrix
		else:
			new_pred_matrix = []
			for y in range(self.num_rows):
				row = []
				for x in range(self.num_cols):
					anc_x,anc_y = ancestors[y][x]

					new_node = viterbi_node()
					new_node.coords = [x,y]
					new_node.parent = deepcopy(old_pred_matrix[anc_y][anc_x])
					anc_prob = new_node.parent.value
					#anc_prob = float(old_pred_matrix[anc_y][anc_x].value)
					new_node.value = float(transition_matrix[y][x].value)*anc_prob
					row.append(new_node)
				new_pred_matrix.append(row)
			return new_pred_matrix

	# returns the sum of all elements in input matrix
	def get_matrix_sum(self,matrix):
		matrix_sum = 0
		for y in range(len(matrix)):
			for x in range(len(matrix[y])):
				matrix_sum += float(matrix[y][x].value)
		return matrix_sum

	# divides each elements of the input matrix by its matrix sum
	def normalize_matrix(self,matrix):
		matrix_sum = float(self.get_matrix_sum(matrix))
		for y in range(len(matrix)):
			for x in range(len(matrix[y])):
				matrix[y][x].value = float(matrix[y][x].value)/matrix_sum
		return matrix

	# initializes the prediction matrix, iterates over provided observations
	# and updates weights at each step. If path is set to True the viterbi algorithm
	# will be called to analyze each step and output the most likely path taken
	def init_observations(self,seen_actions,seen_readings,path=True,print_transition=True,print_condition=True):

		self.show_all = False
		self.print_transition = print_transition
		self.print_condition = print_condition

		self.transition_matrices = []
		self.init_conditions_matrix()

		self.observed_actions = []
		self.observed_readings = []

		if not path:
			sys.stdout.write("\nGrid Conditions...\n")
			self.print_matrix(self.conditions_matrix,3)

		self.move_index = 1
		self.print_current_state(6)

		for self.cur_action,self.cur_reading in zip(seen_actions,seen_readings):

			# add the observation
			self.add_observation()

			# update weights given the new information
			self.update_weights()

			# print out current state information
			self.print_current_state(6)

			if path:
				# get the current sequence
				pred_seq,pred_prob = self.get_predicted_sequence()

				# print out current sequence
				self.print_predicted_sequence(pred_seq,pred_prob)

		print("\nAll observations complete.\n")

	# pred_matrix: prediction matrix
	# current_location: [x,y] (x,y in [0,1,2]), current location
	#
	# return: [x,y] (x,y in [0,1,2]), coordinates of likely ancestor
	def get_ancestor(self,pred_matrix,current_location,last_action=None):
		highest_prob = 0
		ancestor = [-1,-1]
		possible_ancestors = self.get_neighbors(current_location)
		for x,y in possible_ancestors:
			val = pred_matrix[y][x].value
			if val>highest_prob:
				highest_prob = val
				ancestor = [x,y]
		return ancestor,highest_prob

	# pred_matrix: prediction matrix
	# current_location: [x,y] (x,y in [0,1,2]), current location
	#
	# return: [[x,y],...] list of neighbor indices
	def get_neighbors(self,current_location):
		possible_neighbors = []

		x_operations = [1,-1,0]
		y_operations = [1,-1,0]

		for y in y_operations:
			for x in x_operations:
				possible_neighbors.append([current_location[0]+x,current_location[1]+y])

		neighbors = []
		for x,y in possible_neighbors:
			if ((x>=0 and x<self.num_cols) and (y>=0 and y<self.num_rows)):
				if x!=current_location[0] and y!=current_location[1]: continue
				neighbors.append([x,y])
		return neighbors

	# pred_matrix: prediction matrix
	#
	# return: [x,y] (x,y in [0,1,2]), most likely current location
	def predict_location(self,pred_matrix):
		highest_prob = 0
		location 	 = [-1,-1]
		for y in range(len(pred_matrix)):
			for x in range(len(pred_matrix[y])):
				val = pred_matrix[y][x].value
				if val > highest_prob:
					highest_prob = val
					location = [x,y]
		return location, highest_prob

	# pred_matrices: list of prediction matrices (1 for each reported action)
	#
	# return: [[x,y],...] list of predicted locations back to starting spot
	def get_predicted_sequence(self):

		#print("Inside get_predicted_sequence")

		last_location, last_probability = self.predict_location(self.prediction_matrices[-1])

		best_path = []
		best_path.append(last_location)

		last_location = self.prediction_matrices[-1][last_location[1]][last_location[0]]

		path_prob = last_probability

		while True:
			last_location = last_location.parent
			if last_location == None: break
			new_coords = last_location.coords
			best_path.insert(0,new_coords)

		return [best_path[1:],[path_prob]]



		pred_matrices = self.prediction_matrices[1:]
		cur_action = self.cur_action
		cur_reading = self.cur_reading
		seen_actions = self.observed_actions

		'''
		if self.show_all:
			for p in pred_matrices:
				print_matrix(p)
		'''

		best_path = ""
		best_path_probabilities = ""
		highest_prob = 0.0

		for y in range(self.num_rows):
			for x in range(self.num_cols):

				#last_location = [x,y]
				#last_probability = pred_matrices[-1][y][x].value
				#if last_probability==0.0: continue
				last_location,last_probability = self.predict_location(self.prediction_matrices[-1])

				#last_location = None

				predicted_moves = []
				predicted_probabilities = []

				#last_probability = None

				#if len(pred_matrices)<=1:
				#	last_location,last_probability = predict_location(pred_matrices[0])

				for cur_pred_matrix,cur_transition_matrix in zip(reversed(pred_matrices),reversed(self.transition_matrices)):

					#possible_ancestors = self.get_neighbors(cur_pred_matrix,last_location)

					# if this is the first iteration (last prediction matrix)
					#if last_location is None:
					#	last_location,last_probability = self.predict_location(cur_pred_matrix)

					#else:
					last_location,next_probability = self.get_ancestor(cur_transition_matrix,last_location,cur_action)
					last_probability = next_probability*last_probability

					# add the predicted move to beginning of the list
					predicted_moves.insert(0,last_location)
					predicted_probabilities.insert(0,last_probability)

				#if len(pred_matrices)<=2:
				#	predicted_moves.insert(0,last_location)
				#	predicted_probabilities.insert(0,last_probability)

				if predicted_probabilities[-1] > highest_prob:
					best_path = copy(predicted_moves)
					best_path_probabilities = copy(predicted_probabilities)
					highest_prob = predicted_probabilities[-1]

		return best_path, best_path_probabilities

	# condition_matrix: condition matrix (strings)
	# predicted_seq: [[x,y],...] list of predicted locations
	#
	# prints out the path overlaid on the grid
	def print_predicted_sequence(self,predicted_seq,seq_probabilities):
		condition_matrix = self.conditions_matrix
		seen_actions     = self.observed_actions
		seen_readings    = self.observed_readings

		print "\n Predicted Sequence:",predicted_seq
		sys.stdout.write("\n")

		# 9 total rows in predicted sequence diagram
		rows = []
		for i in range(3*self.num_rows):
			rows.append("")

		# row above the sequence digram (see state_header_str below)
		actions = []

		# create string to represent the x axis of the plot (horizontal, top of plot)
		x_axis = "      "
		#for i in range(len(seq_probabilities)):
		if True:
			for i in range(self.num_cols):
				item = str(i)
				left = True
				while len(item)<5:
					if left:
						left = False
						item = " "+item
					else:
						left = True
						item += " "
				x_axis += item
			x_axis += "    "

		# print out state matrix with a single location ( ) denoting current spot, appended onto
		# whatever we have so far in the rows[] array (any previous state matrices, iterations)
		for y in range(self.num_rows):

			above_row = ""
			below_row = ""
			full_row = ""

			for x in range(self.num_cols):

				above_section = "     "
				below_section = "     "

				above_section = list(above_section)
				below_section = list(below_section)

				cur_cond = condition_matrix[y][x].value
				row = "  "+cur_cond+"  "

				row = list(row)

				if [x,y] in predicted_seq:
					i = predicted_seq.index([x,y])

					last_loc = None
					next_loc = None

					if i>0:          		   last_loc = predicted_seq[i-1]
					if len(predicted_seq)>i+1: next_loc = predicted_seq[i+1]

					if last_loc is not None:
						# if the prior location was in the same column
						if last_loc[0]==x:
							# if the prior location was in the above neighbor
							if last_loc[1]==y-1:
								above_section[2] = '|'
								#need_above = True
							# if the prior location was in the below neighbor
							elif last_loc[1]==y+1:
								below_section[2] = '^'
								#need_below = True

						# if the prior location was in the same row
						elif last_loc[1]==y:
							# if the prior location was in the left neighbor
							if last_loc[0]==x-1: row[0],row[1] = '-','>'
							# if the prior location was in the right neighbor
							if last_loc[0]==x+1: row[3],row[4] = '<','-'

					if next_loc is not None:
						# if the next location is in the same column
						if next_loc[0]==x:
							# if the next location is in the above neighbor
							if next_loc[1]==y-1:
								above_section[2] = '|'
							# if the next location is in the below neighbor
							elif next_loc[1]==y+1:
								below_section[2] = 'v'
								#need_below = True

						# if the next location is in the same row
						elif next_loc[1]==y:
							# if the next location is in the left neighbor
							if next_loc[0]==x-1: row[0],row[1] = '<','-'
							# if the next location is in the right neighbor
							if next_loc[0]==x+1: row[3],row[4] = '-','>'

					if predicted_seq[len(predicted_seq)-1] == [x,y]:
						row[1] = '('
						row[3] = ')'

					if predicted_seq[0] == [x,y]:
						row[1] = '['
						row[3] = ']'

					#del predicted_seq[i]

				above_section = "".join(above_section)
				below_section = "".join(below_section)
				row = "".join(row)

				above_row += above_section
				full_row  += row
				below_row += below_section

			rows[3*y]   += above_row
			rows[3*y+1] += full_row
			rows[3*y+2] += below_row

		horizontal_delim = ''.join("_" for _ in range(5*self.num_cols))
		sys.stdout.write(x_axis)
		sys.stdout.write("\n      "+horizontal_delim+"\n")

		# write out the path displays
		for i in range(1,3*self.num_rows-1):
			if (i+2)%3==0:
				item = " "+str(int(i/3))
				while len(item)<4:
					item+=" "
				#sys.stdout.write(item+"|")
			else:
				item = ""
			while len(item)<5:
				item = " "+item
			item += "|"
			sys.stdout.write(item)
			sys.stdout.write(rows[i]+"|"+"\n")

		sys.stdout.write("      "+horizontal_delim)
		sys.stdout.write("\n")

		sys.stdout.write("\n Predicted Sequence Probability: ")
		sys.stdout.write(str(seq_probabilities[-1])[:8])
		sys.stdout.write("\n")

	# prints a copy of the conditions_matrix with the provided sequence overlaid
	def print_single_sequence(self,sequence,print_seq=True):


		if print_seq:
			sys.stdout.write("  "+" ".join("["+str(a)+","+str(b)+"]" for [a,b] in sequence))
			sys.stdout.write("\n\n")


		sys.stdout.write("\n")

		# create x axis to print above plot
		x_axis = ""
		for i in range(self.num_cols):
			item = str(i)
			left = True
			while len(item)<5:
				if left:
					left = False
					item = " "+item
				else:
					left = True
					item += " "
			x_axis += item
		sys.stdout.write("        "+x_axis+"\n")

		x_axis_divider = "".join("_" for _ in range(5*self.num_cols))
		sys.stdout.write("      "+x_axis_divider+"\n")

		condition_matrix = self.conditions_matrix

		rows = []
		for i in range(3*self.num_rows):
			rows.append("")

		# make shallow copy of the predicted sequence (a list of [x,y] coordinates)
		seq = copy(sequence)

		# to know if we have covered each location on the sequence
		rendered_node = [0] * len(sequence)

		# if a row has no contents, don't write it out at end
		empty_rows = [False] * (3*self.num_rows)

		# print out state matrix with a single location ( ) denoting current spot, appended onto
		# whatever we have so far in the rows[] array (any previous state matrices, iterations)
		for y in range(self.num_rows):

			above_row = "" # row printed above current element row
			below_row = "" # row printed below current element row
			full_row  = "" # row holding current element row
			need_above = False # if we don't place anything in the above_row
			need_below = False # if we don't place anything in the below_row

			for x in range(self.num_cols): # iterate over each element in current row

				above_section = "     " # substring of above_row (to be appended)
				below_section = "     " # substring of below_row (to be appended)

				above_section = list(above_section)
				below_section = list(below_section)

				cur_cond = condition_matrix[y][x].value
				row = "  "+cur_cond+"  " # add the current element
				row = list(row)

				if [x,y] in seq: # if this spot is in the traversal sequence

					i = seq.index([x,y])
					skip = False

					if rendered_node[i]==1:
						i+=1
						while True:
							if i==len(seq):
								skip = True
								break
							if x==seq[i][0] and y==seq[i][1]:
								if rendered_node[i]==0:
									rendered_node[i] = 1
									skip = False
									break
							i+=1

					last_loc = None
					next_loc = None

					if not skip:
						if i>0:          last_loc = seq[i-1] # if there was an earlier element in sequence
						if i<len(seq)-1: next_loc = seq[i+1] # if there is another element in sequence

					if last_loc is not None:
						# if the prior location was in the same column
						if last_loc[0]==x:
							# if the prior location was in the above neighbor
							if last_loc[1]==y-1:
								above_section[2] = '|'
								need_above = True
							# if the prior location was in the below neighbor
							elif last_loc[1]==y+1:
								below_section[2] = '^'
								need_below = True

						# if the prior location was in the same row
						elif last_loc[1]==y:
							# if the prior location was in the left neighbor
							if last_loc[0]==x-1: row[0],row[1] = '-','>'
							# if the prior location was in the right neighbor
							if last_loc[0]==x+1: row[3],row[4] = '<','-'

					if next_loc is not None:
						# if the next location is in the same column
						if next_loc[0]==x:
							# if the next location is in the above neighbor
							if next_loc[1]==y-1:
								above_section[2] = '|'
								need_above = True
							# if the next location is in the below neighbor
							elif next_loc[1]==y+1:
								below_section[2] = 'v'
								need_below = True

						# if the next location is in the same row
						elif next_loc[1]==y:
							# if the next location is in the left neighbor
							if next_loc[0]==x-1: row[0],row[1] = '<','-'
							# if the next location is in the right neighbor
							if next_loc[0]==x+1: row[3],row[4] = '-','>'

					# if the final destination
					if seq[len(seq)-1] == [x,y]:
						row[1] = '('
						row[3] = ')'

					# if the starting location
					if seq[0] == [x,y]:
						row[1] = '['
						row[3] = ']'


				above_section = "".join(above_section) # turn list to string
				below_section = "".join(below_section) # turn list to string
				row = "".join(row) # turn list to string

				# add items to this rows' above, below, and full row attributes
				above_row += above_section
				full_row  += row
				below_row += below_section



			rows[3*y]   += above_row
			rows[3*y+1] += full_row
			rows[3*y+2] += below_row

			if need_above==False: empty_rows[3*y] = True # if we never put anything in above_row
			if need_below==False: empty_rows[3*y+2] = True # if we never put anything in below_row

		# print out all nonempty rows
		for i in range(3*self.num_rows):
			if not empty_rows[i]:
				if (i+2)%3 == 0:
					item =" "+str(int(i/3))
					while len(item)<5:
						item += " "
				else:
					item = "     "
				item += "|  "
				print_row = item+rows[i]
				print(print_row)

		sys.stdout.write("\n")

	# desired_item_size: column width in characters
	#
	# prints out either a prediction or condition matrix
	def print_matrix(self,matrix,desired_item_size=20):

		x_axis = ""
		for i in range(self.num_cols):
			item = str(i)
			left = True
			while len(item)<desired_item_size+3:
				if left:
					left = False
					item = " "+item
				else:
					left = True
					item += " "
			x_axis += item
		sys.stdout.write("      "+x_axis+"\n")

		delim_line = ''.join("_" for _ in range(self.num_cols*(desired_item_size+3)))
		#delim_line = delim_line
		sys.stdout.write("      "+delim_line[:len(delim_line)-1]+"\n")

		idx = -1
		for row in matrix:
			idx+=1
			before = "  "+str(idx)
			while len(before)<5:
				before+=" "
			sys.stdout.write(before+"| ")
			for item in row:

				# if just a string entry
				if str(item.value)==item.value:
					real_item_size = len(str(item.value))
					sys.stdout.write(str(item.value)[:desired_item_size])
				# if a float entry, write out formatted
				else:
					real_item_size = desired_item_size
					output_str = "%0."+str(desired_item_size-2)+"f"
					sys.stdout.write(output_str % item.value)

				if real_item_size<desired_item_size:
					for _ in range(desired_item_size-real_item_size):
						sys.stdout.write(" ")

				if row.index(item) is not len(row)-1:
					sys.stdout.write(" | ")
				else:
					sys.stdout.write(" |")
			if matrix.index(row) is not len(matrix)-1:
				sys.stdout.write("\n     |"+delim_line[:len(delim_line)-1]+"|\n")
			else:
				sys.stdout.write("\n     |"+delim_line[:len(delim_line)-1]+"|\n")

	# prints out information about the current step, i.e. the current
	# condition matrix (doesn't change over steps), the current prediction
	# matrix (adjusted on each step), the current reported action, and the
	# current reported reading
	def print_current_state(self,desired_item_size=20):

		pred_matrix 	 = self.prediction_matrices[-1] # get the last prediction matrix
		condition_matrix = self.conditions_matrix # get the condition matrix

		delim_line = ''.join("=" for _ in range((self.num_cols+5)*(desired_item_size)))
		if self.move_index==1:
			print("\n"+delim_line+"\n"+delim_line)
			print(" Initial State")
		else:
			print("\n"+delim_line+"\n"+delim_line)
			print(" Move Index:\t\t"+str(self.move_index-1))
			print(" Reported Action:\t("+str(self.cur_action)+", "+str(self.cur_reading)+")")

		if condition_matrix is not None and self.print_condition:
			sys.stdout.write("\n Condition Matrix:\n")
			self.print_matrix(condition_matrix,desired_item_size)

		if len(self.transition_matrices)!=0 and self.print_transition:
			sys.stdout.write("\n Transition Matrix:\n")
			self.print_matrix(self.transition_matrices[-1],desired_item_size)

		if self.temp_anc_matrix is not None:
			print("\n Ancestors Matrix: ")
			self.print_matrix(self.temp_anc_matrix,desired_item_size)

		if pred_matrix is not None and not self.show_all:
			sys.stdout.write("\n\n Current Probability Matrix:\n")
			self.print_matrix(pred_matrix,desired_item_size)

		if self.actual_traversal_path!=None and self.print_actual_traversal:
			if self.move_index!=1 and not self.print_full_traversal:
				sys.stdout.write("\n Current ACTUAL Agent Traversal (trimd. to length of predicted)...\n")
				if len(self.actual_traversal_path)<=self.current_predicted_length+1:
					print_seq = self.actual_traversal_path
				else:
					print_seq = self.actual_traversal_path[:self.current_predicted_length+1]
				self.print_single_sequence(print_seq)
			else:
				sys.stdout.write("\n Full ACTUAL Agent Traversal Path { end=(), start=[] }...\n")
				self.print_single_sequence(self.actual_traversal_path)

import sys
import time

import random

from copy import deepcopy, copy

class viterbi_node:
	def __init__(self):
		self.value = ""
		self.parent = None

class viterbi_matrix:
	def __init__(self,num_rows=3,num_cols=3,values=["H","H","T","N","N","N","N","B","H"]):
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.values = values
		self.init_conditions_matrix(values)
		self.init_prediction_matrix(values)

	# creates a new 3x3 condition matrix given the provided conditions
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

	def init_prediction_matrix(self,conditions=None):
		self.prediction_matrices = []
		cells = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = float(0.125)
				if self.values[y*self.num_cols+x] is "B":
					new_node.value = 0.0
				row.append(new_node)
			cells.append(row)
		self.prediction_matrices.append(cells)

	def add_observation(self):
		self.observed_actions.append(self.cur_action)
		self.observed_readings.append(self.cur_reading)
		self.move_index+=1

	# creates a new 3x3 prediction matrix given the provided conditions
	def empty_prediction_matrix(self):
		cells = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = float(1/8)
				if new_node.value is "B":
					new_node.value = 1.0
				row.append(new_node)
			cells.append(row)
		return cells

	def init_emission_matrix(self):
		self.emission_matrix = emission_matrix(self.values)

	# cur_action: reported movement direction in this step
	# cur_reading: reported reading in this step
	# condition_matrix: 3x3 condition matrix (strings)
	# pred_matrix: current 3x3 prediction matrix (floats)
	#
	# return: updated 3x3 prediction matrix (given a new move)
	def update_weights(self):

		#self.init_emission_matrix()

		if len(self.observed_actions)+1==len(self.prediction_matrices):
			print("ERROR: update_weights()")
			return

		cur_action = self.cur_action
		cur_reading = self.cur_reading

		old_pred_matrix 	= self.prediction_matrices[-1] # get the last prediction matrix
		transition_matrix 	= self.empty_prediction_matrix()

		condition_matrix = self.conditions_matrix

		# set probabilities given the reported reading compared to state values
		for y in range(3):
			for x in range(3):

				# never in this state
				if condition_matrix[y][x].value=="B": transition_matrix[y][x].value = 0.0

				# in this state with 0.9 confidence (same as reading)
				elif condition_matrix[y][x].value==cur_reading: transition_matrix[y][x].value = 0.9
		
				# in this state only if there was a mis-reading of the cur_reading
				else: transition_matrix[y][x].value = 0.1

		# set probabilities given the reported movement (cur_action) compared to condition neighbors
		#
		# iterate over all possible current locations
		for y in range(3):
			for x in range(3):

				# if the current location is a blocked cell, it will have already been set to P = 0.0
				if condition_matrix[y][x].value=="B": continue

				# if the reported action was a translation to the right
				if cur_action=="Right":
					# if the current location is in the middle or right columns
					if x==1:
						if condition_matrix[y][x-1].value=="B": transition_matrix[y][x].value *= 0.1
						else: 							  		transition_matrix[y][x].value *= 0.9

					#if x==2: transition_matrix[y][x].value *= 0.9
					
					# if the current location is in the left column
					if x==0:
						# if a right translation could be prevented due to a blocked cell to the right
						if condition_matrix[y][x+1].value=="B": transition_matrix[y][x].value *= 0.9
						else: 							  		transition_matrix[y][x].value *= 0.1

				if cur_action=="Left":
					if x==1: 
						if condition_matrix[y][x+1].value=="B": transition_matrix[y][x].value *= 0.1
						else: 							  		transition_matrix[y][x].value *= 0.9

					#if x==0: transition_matrix[y][x].value *= 0.9

					if x==2:
						if condition_matrix[y][x-1].value=="B": transition_matrix[y][x].value *= 0.9
						else: 							  		transition_matrix[y][x].value *= 0.1

				if cur_action=="Up":
					if y==1: 
						if condition_matrix[y+1][x].value=="B": transition_matrix[y][x].value *= 0.1
						else: 									transition_matrix[y][x].value *= 0.9

					#if y==0: transition_matrix[y][x].value *= 0.9

					if y==2:
						if condition_matrix[y-1][x].value=="B": transition_matrix[y][x].value *= 0.9
						else: 							  		transition_matrix[y][x].value *= 0.1

				if cur_action=="Down":
					if y==1: 
						if condition_matrix[y-1][x].value=="B": transition_matrix[y][x].value *= 0.1
						else: 									transition_matrix[y][x].value *= 0.9

					#if y==2: transition_matrix[y][x].value *= 0.9

					if y==0:
						if condition_matrix[y+1][x].value=="B": transition_matrix[y][x].value *= 0.9
						else:							  		transition_matrix[y][x].value *= 0.1
		
		# now need to normalize all values by dividing by probability sum
		#transition_matrix = self.normalize_matrix(transition_matrix)

		# add new transition matrix to list of prior transition matrices
		self.transition_matrices.append(transition_matrix)

		# create new prediction matrix
		new_pred_matrix = self.resolve_prediction_matrix(transition_matrix,old_pred_matrix)

		# normalize the new prediction matrix
		new_pred_matrix = deepcopy(self.normalize_matrix(new_pred_matrix))

		if len(self.prediction_matrices)==0:
			for y in range(3):
				for x in range(3):
					new_pred_matrix[y][x].parent = None
					new_pred_matrix[y][x].transition_prob = transition_matrix[y][x].value
		else:
			for y in range(3):
				for x in range(3):
					new_pred_matrix[y][x].parent = self.prediction_matrices[-1][y][x]
					new_pred_matrix[y][x].transition_prob = transition_matrix[y][x].value

		# add new prediction matrix to list
		self.prediction_matrices.append(new_pred_matrix)

	def resolve_prediction_matrix(self,transition_matrix,old_pred_matrix):
		new_pred_matrix = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = float(transition_matrix[y][x].value)*float(old_pred_matrix[y][x].value)
				row.append(new_node)
			new_pred_matrix.append(row)
		return new_pred_matrix

	def get_matrix_sum(self,matrix):
		matrix_sum = 0
		for y in range(3):
			for x in range(3):
				matrix_sum += float(matrix[y][x].value)
		return matrix_sum

	# divides each elements of the input 3x3 matrix by its matrix sum
	def normalize_matrix(self,matrix):
		matrix_sum = float(self.get_matrix_sum(matrix))
		for y in range(3):
			for x in range(3):
				matrix[y][x].value = float(matrix[y][x].value)/matrix_sum
		return matrix

	def init_observations(self,seen_actions,seen_readings,path=True):

		self.show_all = False

		self.transition_matrices = []
		self.init_conditions_matrix()

		self.observed_actions = []
		self.observed_readings = []

		self.move_index = 1
		self.print_current_state()

		for self.cur_action,self.cur_reading in zip(seen_actions,seen_readings):

			# add the observation
			self.add_observation()
			
			# update weights given the new information
			self.update_weights()

			# print out current state information
			self.print_current_state()

			if path:
				# get the current sequence
				pred_seq,pred_prob = self.get_predicted_sequence()

				# print out current sequence
				self.print_predicted_sequence(pred_seq,pred_prob)

	# pred_matrix: 3x3 prediction matrix
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

	# pred_matrix: 3x3 prediction matrix
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
			if ((x>=0 and x<3) and (y>=0 and y<3)): 
				if x!=current_location[0] and y!=current_location[1]: continue
				neighbors.append([x,y])
		return neighbors

	# pred_matrix: 3x3 prediction matrix
	#
	# return: [x,y] (x,y in [0,1,2]), most likely current location
	def predict_location(self,pred_matrix):
		highest_prob = 0
		location 	 = [-1,-1]
		for y in range(3):
			for x in range(3):
				val = pred_matrix[y][x].value
				if val > highest_prob:
					highest_prob = val 
					location = [x,y]
		return location, highest_prob

	# pred_matrices: list of 3x3 prediction matrices (1 for each reported action)
	#
	# return: [[x,y],...] list of predicted locations back to starting spot
	def get_predicted_sequence(self):

		#print("Inside get_predicted_sequence")

		pred_matrices = self.prediction_matrices[1:]
		cur_action = self.cur_action 
		cur_reading = self.cur_reading 
		show_all = self.show_all
		seen_actions = self.observed_actions

		if show_all:
			for p in pred_matrices:
				print_matrix(p)

		best_path = ""
		best_path_probabilities = ""
		highest_prob = 0.0

		#print("here")

		'''
		recent_location,recent_probability = self.predict_location(self.predicted_matrices[-1])
		initial_probability = recent_probability

		while True:

			depth = 0
			possible_ancestors = self.get_neighbors(recent_location
		'''

		for y in range(3):
			for x in range(3):

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

	# condition_matrix: 3x3 condition matrix (strings)
	# predicted_seq: [[x,y],...] list of predicted locations
	#
	# prints out the path overlaid on the grid
	def print_predicted_sequence(self,predicted_seq,seq_probabilities):
		condition_matrix = self.conditions_matrix
		seen_actions = self.observed_actions
		seen_readings = self.observed_readings

		print "\nPredicted Sequence:",predicted_seq
		print "..."

		# 9 total rows in predicted sequence diagram
		rows = []
		for i in range(9):
			rows.append("")

		# make shallow copy of the predicted sequence (a list of [x,y] coordinates)
		overall_predicted_seq = copy(predicted_seq)

		# incremented before each while loop (starts at zero)
		cur_step = -1

		# row above the sequence digram (see state_header_str below)
		actions = []

		# each iteration covers a single state in the sequence diagram
		while True:
			cur_step += 1
			if cur_step==len(overall_predicted_seq):
				break

			predicted_seq = overall_predicted_seq[:cur_step+1]

			# make delimeter between states in the sequence diagram
			if cur_step!=0:
				for i in range(1,9):
					rows[i] += " ## "

			# create headers, e.g. '(Left, N)', seen above the predicted sequence diagram
			state_header_str = "("+seen_actions[cur_step]+", "+seen_readings[cur_step]+")"
			if len(state_header_str)<10:
				for i in range(len(state_header_str),10):
					state_header_str+=" "
			actions.append(state_header_str)

			# print out state matrix with a single location ( ) denoting current spot, appended onto
			# whatever we have so far in the rows[] array (any previous state matrices, iterations)
			for y in range(3):
				
				above_row = ""
				below_row = ""
				full_row = ""

				for x in range(3):

					above_section = "     "
					below_section = "     "

					above_section = list(above_section)
					below_section = list(below_section)

					cur_cond = condition_matrix[y][x].value
					row = "  "+cur_cond+"  "

					row = list(row)

					if [x,y] in predicted_seq:

						predicted_seq = copy(overall_predicted_seq[:cur_step+1])

						i = predicted_seq.index([x,y])
						
						if predicted_seq[len(predicted_seq)-1] == [x,y]:
							row[1] = '('
							row[3] = ')'

						last_loc = None 
						next_loc = None 

						if i>0: last_loc = predicted_seq[i-1]
						if i<len(predicted_seq)-1: next_loc = predicted_seq[i+1]

						del predicted_seq[i]

						if last_loc is not None:

							# if the prior location was in the same column
							if last_loc[0]==x:

								# if the prior location was in the above neighbor
								if last_loc[1]==y-1:
									above_section[2] = '|'

								# if the prior location was in the below neighbor
								elif last_loc[1]==y+1:
									below_section[2] = '|'

							# if the prior location was in the same row
							elif last_loc[1]==y:

								# if the prior location was in the left neighbor
								if last_loc[0]==x-1:
									row[0] = '-'

								# if the prior location was in the right neighbor
								if last_loc[0]==x+1:
									row[4] = '-'

						if next_loc is not None:

							# if the prior location was in the same column
							if next_loc[0]==x:

								# if the prior location was in the above neighbor
								if next_loc[1]==y-1:
									above_section[2] = '|'

								# if the prior location was in the below neighbor
								elif next_loc[1]==y+1:
									below_section[2] = '|'

							# if the prior location was in the same row
							elif next_loc[1]==y:

								# if the prior location was in the left neighbor
								if next_loc[0]==x-1:
									row[0] = '-'

								# if the prior location was in the right neighbor
								if next_loc[0]==x+1:
									row[4] = '-'

					above_section = "".join(above_section)
					below_section = "".join(below_section)
					row = "".join(row)

					above_row += above_section
					full_row  += row 
					below_row += below_section

				rows[3*y]   += above_row
				rows[3*y+1] += full_row
				rows[3*y+2] += below_row 

		actions_str = ""
		for a in actions:
			actions_str+=a
		horizontal_delim = ''.join("=" for _ in range(len(actions_str)+(9*len(actions))))
		#sys.stdout.write(horizontal_delim+"\n")

		sys.stdout.write("   ")
		for a in actions:
			sys.stdout.write(a+"         ")

		for i in range(9):
			print(rows[i])

		sys.stdout.write("P:")
		sys.stdout.write("  ")
		for p in seq_probabilities:
			sys.stdout.write(str(p)[:8]+"           ")
		#sys.stdout.write("\n"+horizontal_delim+"\n")
		sys.stdout.write("\n...\n")

	# desired_item_size: column width in characters
	#
	# prints out either a prediction or condition matrix
	def print_matrix(self,matrix,desired_item_size=20):
		delim_line = ''.join("_" for _ in range(3*desired_item_size+10))
		sys.stdout.write("\n"+delim_line+"\n")
		for row in matrix:
			sys.stdout.write("| ")
			for item in row:
				real_item_size = len(str(item.value))
				sys.stdout.write(str(item.value)[:desired_item_size])
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

	# prints out information about the current step, i.e. the current
	# condition matrix (doesn't change over steps), the current prediction
	# matrix (adjusted on each step), the current reported action, and the
	# current reported reading
	def print_current_state(self,desired_item_size=20):
		pred_matrix = self.prediction_matrices[-1]
		condition_matrix = self.conditions_matrix

		delim_line = ''.join("=" for _ in range(3*desired_item_size+10))
		if self.move_index==1:
			print("\n"+delim_line)
			print("Initial State")
		else:
			print(delim_line)
			print("\nMove Index:\t\t"+str(self.move_index-1))
			print("Reported Action:\t("+str(self.cur_action)+", "+str(self.cur_reading)+")")

		'''
		if condition_matrix is not None:
			sys.stdout.write("\nCondition Matrix:")
			self.print_matrix(condition_matrix,5)
		'''
		'''
		if len(self.transition_matrices)!=0:
			sys.stdout.write("\nTransition Matrix:")
			self.print_matrix(self.transition_matrices[-1],desired_item_size)
		'''
		if pred_matrix is not None and not self.show_all:
			sys.stdout.write("\nPrediction Matrix:")
			self.print_matrix(pred_matrix,desired_item_size)

# compute the probability of where we are in grid world given inputs 'actions' and 
# subsequent sensor readings 'readings'
def predict_location(actions,readings):

	v_matrix = viterbi_matrix()
	v_matrix.init_observations(actions,readings,path=False)

def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	predict_location(actions,readings)

if __name__ == '__main__':
	main()
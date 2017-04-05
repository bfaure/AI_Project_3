import sys
import time

import random

from copy import deepcopy, copy

class viterbi_matrix:

	class viterbi_node:
		def __init__(self):
			self.value = ""
			self.parent = None

	def __init__(self,num_rows=3,num_cols=3,values=["H","H","T","N","N","N","N","B","H"]):
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.values = values
		self.init_conditions_matrix(values)
		self.init_matrix(num_rows,num_cols,values)

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
				new_node.value = self.values[y*self.num_cols+x]
				if new_node.value is "B":
					new_node.value = 0.0
				row.append(new_node)
			cells.append(row)
		self.prediction_matrices.append(cells)

	def add_observation(self,observed_action,observed_reading):
		self.observed_actions.append(observed_action)
		self.observed_readings.append(observed_reading)

	def create_prediction_matrix(self):
		cells = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = self.values[y*self.num_cols+x]
				if new_node.value is "B":
					new_node.value = 0.0
				row.append(new_node)
			cells.append(row)
		return cells

	def update_weights(self):

		if len(self.observed_actions)==len(self.prediction_matrices):
			print("ERROR: update_weights()")
			return

		cur_action = self.observed_actions[-1]
		cur_reading = self.observed_readings[-1]

		old_pred_matrix = self.prediction_matrices[-1] # get the last prediction matrix
		transition_matrix = deepcopy(old_pred_matrix)

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
					if x==1 or x==2: transition_matrix[y][x].value *= 0.9
					# if the current location is in the left column
					if x==0:
						# if a right translation could be prevented due to a blocked cell to the right
						if condition_matrix[y][x+1].value=="B": transition_matrix[y][x].value *= 0.9
						else: 							  		transition_matrix[y][x].value *= 0.1

				if cur_action=="Left":
					if x==0 or x==1: transition_matrix[y][x].value *= 0.9
					if x==2:
						if condition_matrix[y][x-1].value=="B": transition_matrix[y][x].value *= 0.9
						else: 							  		transition_matrix[y][x].value *= 0.1

				if cur_action=="Up":
					if y==0 or y==1: transition_matrix[y][x].value *= 0.9
					if y==2:
						if condition_matrix[y-1][x].value=="B": transition_matrix[y][x].value *= 0.9
						else: 							  		transition_matrix[y][x].value *= 0.1

				if cur_action=="Down":
					if y==1 or y==2: transition_matrix[y][x].value *= 0.9
					if y==0:
						if condition_matrix[y+1][x].value=="B": transition_matrix[y][x].value *= 0.9
						else:							  		transition_matrix[y][x].value *= 0.1
		
		# now need to normalize all values by dividing by probability sum
		transition_matrix = normalize_matrix(transition_matrix)

		# add new transition matrix to list of prior transition matrices
		self.transition_matrices.append(transition_matrix)

		# create new prediction matrix
		new_pred_matrix = self.create_prediction_matrix(transition_matrix,old_pred_matrix)

		# add new prediction matrix to list
		self.prediction_matrices.append(new_pred_matrix)

	def create_prediction_matrix(self,transition_matrix,old_pred_matrix):
		new_pred_matrix = []
		for y in range(self.num_rows):
			row = []
			for x in range(self.num_cols):
				new_node = viterbi_node()
				new_node.value = transition_matrix[y][x].value*old_pred_matrix[y][x].value
				row.append(new_node)
			new_pred_matrix.append(row)
		return new_pred_matrix

	def get_matrix_sum(self,matrix):
		matrix_sum = 0
		for y in range(3):
			for x in range(3):
				matrix_sum += float(matrix[y][x]value)
		return matrix_sum

	def normalize_pred_matrix(self,matrix):
		matrix_sum = float(self.get_matrix_sum(matrix))
		for y in range(3):
			for x in range(3):
				matrix[y][x].value = float(matrix[y][x].value)/matrix_sum
		return matrix

	def print_current_state(self):


	def init_observations(self,seen_actions,seen_readings):

		self.show_all = False

		self.init_conditions_matrix()
		self.init_observations_matrix()

		self.observed_actions = []
		self.observed_readings = []

		condition_matrix = create_condition_matrix()
		pred_matrix = create_prediction_matrix()
		print_current_state(condition_matrix,pred_matrix)

		move_index = 1

		seen_actions = []
		seen_readings = []

		#pred_matrices.append(copy(pred_matrix))

		for cur_action,cur_reading in zip(actions,readings):

			# add the observation
			self.add_observation(cur_action,cur_reading)
			
			# update weights given the new information
			self.update_weights()

			# print out current state information
			self.print_current_state()
			
			# get the most likely traversal sequence
			predicted_seq,probabilities = get_predicted_sequence(pred_matrices,cur_action,cur_reading,show_all,seen_actions)

			if predicted_seq!=-1:

				# print out the most likely traversal sequence
				print_predicted_sequence(condition_matrix,predicted_seq,probabilities,seen_actions,seen_readings)

			move_index+=1

	def print_matrix(self,matrix,desired_item_size=20):
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

	def print_current_state(self,desired_item_size=20):
		pred_matrix = self.prediction_matrices[-1]
		condition_matrix = self.condition_matrix

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
			self.print_matrix(condition_matrix,5)
		if pred_matrix is not None not show_all:
			sys.stdout.write("\nPrediction Matrix")
			self.print_matrix(pred_matrix,desired_item_size)

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
def print_current_state(condition_matrix=None,pred_matrix=None,move_index=0,cur_action=None,cur_reading=None,desired_item_size=20,print_pred=False):
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
	if pred_matrix is not None and print_pred:
		sys.stdout.write("\nPrediction Matrix")
		print_matrix(pred_matrix,desired_item_size)
	#print("\n"+delim_line)


# divides each elements of the input 3x3 matrix by its matrix sum
def normalize_matrix(matrix):
	matrix_sum = float(get_matrix_sum(matrix))
	for y in range(3):
		for x in range(3):
			matrix[y][x] = float(matrix[y][x])/matrix_sum
	return matrix

# cur_action: reported movement direction in this step
# cur_reading: reported reading in this step
# condition_matrix: 3x3 condition matrix (strings)
# pred_matrix: current 3x3 prediction matrix (floats)
#
# return: updated 3x3 prediction matrix (given a new move)
def update_predictions(cur_action,cur_reading,condition_matrix,old_pred_matrix):
	pred_matrix = deepcopy(old_pred_matrix)

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
	return pred_matrix

# pred_matrix: 3x3 prediction matrix
#
# return: [x,y] (x,y in [0,1,2]), most likely current location
def predict_location(pred_matrix):
	highest_prob = 0
	location 	 = [-1,-1]
	for y in range(3):
		for x in range(3):
			val = pred_matrix[y][x]
			if val > highest_prob:
				highest_prob = val 
				location = [x,y]
	return location, highest_prob

# pred_matrix: 3x3 prediction matrix
# current_location: [x,y] (x,y in [0,1,2]), current location
#
# return: [[x,y],...] list of neighbor indices
def get_neighbors(pred_matrix,current_location):
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
# current_location: [x,y] (x,y in [0,1,2]), current location
#
# return: [x,y] (x,y in [0,1,2]), coordinates of likely ancestor
def get_ancestor(pred_matrix,current_location,last_action=None):
	highest_prob = 0
	ancestor = [-1,-1]
	possible_ancestors = get_neighbors(pred_matrix,current_location)
	for x,y in possible_ancestors:
		val = pred_matrix[y][x]
		if val>highest_prob:
			highest_prob = val 
			ancestor = [x,y]
	return ancestor,highest_prob

def get_neighbor_weights(pred_matrix,last_location,neighbors):
	neighbor_weights = []
	for n in neighbors:
		neighbor_weights.append(pred_matrix[n[1],n[0]])
	return neighbor_weights

# pred_matrices: list of 3x3 prediction matrices (1 for each reported action)
#
# return: [[x,y],...] list of predicted locations back to starting spot
def get_predicted_sequence(pred_matrices,cur_action,cur_reading,show_all,seen_actions):

	if show_all:
		for p in pred_matrices:
			print_matrix(p)

	best_path = ""
	best_path_probabilities = ""
	highest_prob = 0.0

	for y in range(3):
		for x in range(3):

			last_location = [x,y]
			last_probability = pred_matrices[-1][y][x]
			if last_probability==0.0: continue

			#last_location = None

			predicted_moves = []
			predicted_probabilities = []

			#last_probability = None

			#if len(pred_matrices)<=1:
			#	last_location,last_probability = predict_location(pred_matrices[0])

			for cur_pred_matrix in reversed(pred_matrices[:-2]):

				# if this is the first iteration (last prediction matrix)
				if last_location is None: 
					last_location,last_probability = predict_location(cur_pred_matrix)

				else:
					last_location,next_probability = get_ancestor(cur_pred_matrix,last_location,cur_action)
					last_probability = next_probability*last_probability

				# add the predicted move to beginning of the list
				predicted_moves.insert(0,last_location)
				predicted_probabilities.insert(0,last_probability)

			if len(pred_matrices)<=2:
				predicted_moves.insert(0,last_location)
				predicted_probabilities.insert(0,last_probability)

			if predicted_probabilities[-1] > highest_prob:
				best_path = copy(predicted_moves)
				best_path_probabilities = copy(predicted_probabilities)
				highest_prob = predicted_probabilities[-1]

	#print("Best path: ",best_path)
	#print("Best path prob: ",best_path_probabilities)
	return best_path, best_path_probabilities

#def get_above_below_strings()

# condition_matrix: 3x3 condition matrix (strings)
# predicted_seq: [[x,y],...] list of predicted locations
#
# prints out the path overlaid on the grid
def print_predicted_sequence(condition_matrix,predicted_seq,seq_probabilities,seen_actions,seen_readings):
	print "\nPredicted Sequence:",predicted_seq
	print "Sequence Trace: \n"

	rows = []
	for i in range(9):
		rows.append("")

	overall_predicted_seq = copy(predicted_seq)
	cur_step = -1

	actions = []

	while True:
		cur_step += 1
		if cur_step==len(overall_predicted_seq):
			break

		predicted_seq = overall_predicted_seq[:cur_step+1]

		if cur_step!=0:
			for i in range(1,9):
				rows[i] += " || "

		#if cur_step!=0:
		actions.append("("+seen_actions[cur_step]+", "+seen_readings[cur_step]+")")
		#else:
		#	actions.append("        ")

		for y in range(3):
			
			above_row = ""
			below_row = ""
			full_row = ""

			for x in range(3):

				above_section = "     "
				below_section = "     "

				above_section = list(above_section)
				below_section = list(below_section)

				cur_cond = condition_matrix[y][x]
				row = "  "+cur_cond+"  "

				row = list(row)

				if [x,y] in predicted_seq:

					i = predicted_seq.index([x,y])
					
					if predicted_seq[len(predicted_seq)-1] == [x,y]:
						row[1] = '('
						row[3] = ')'

					last_loc = None 
					next_loc = None 

					if i>0: last_loc = predicted_seq[i-1]
					if i<len(predicted_seq)-1: next_loc = predicted_seq[i+1]

					#del predicted_seq[i]

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

	sys.stdout.write("   ")
	for a in actions:
		sys.stdout.write(a+"         ")

	for i in range(9):
		print(rows[i])

	sys.stdout.write("P:")
	sys.stdout.write("  ")
	for p in seq_probabilities:
		sys.stdout.write(str(p)[:8]+"           ")
	sys.stdout.write("\n")

def viterbi(actions,readings):

	show_all = False

	pred_matrices = []

	condition_matrix = create_condition_matrix()
	pred_matrix = create_prediction_matrix()
	print_current_state(condition_matrix,pred_matrix)

	move_index = 1

	seen_actions = []
	seen_readings = []

	#pred_matrices.append(copy(pred_matrix))

	for cur_action,cur_reading in zip(actions,readings):

		seen_actions.append(cur_action)
		seen_readings.append(cur_reading)

		# update prediction values given new information
		pred_matrix = update_predictions(cur_action,cur_reading,condition_matrix,pred_matrix)


		# add to list of states
		pred_matrices.append(deepcopy(pred_matrix)) 

		# print out current state information
		print_current_state(pred_matrix=pred_matrix,move_index=move_index,cur_action=cur_action,cur_reading=cur_reading,print_pred= not show_all)
		
		# get the most likely traversal sequence
		predicted_seq,probabilities = get_predicted_sequence(pred_matrices,cur_action,cur_reading,show_all,seen_actions)

		if predicted_seq!=-1:

			# print out the most likely traversal sequence
			print_predicted_sequence(condition_matrix,predicted_seq,probabilities,seen_actions,seen_readings)

		move_index+=1

def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	actions = ["Right","Down","Down","Down","Down"]
	readings = ["N","H","H","H","H"]

	actions = ["Left","Left","Up","Right","Right","Down","Down"]
	readings = ["N","N","H","H","T","N","H"]

	viterbi(actions,readings)

if __name__ == '__main__':
	main()
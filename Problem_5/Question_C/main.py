import sys
import time
import random
import os

from shutil import rmtree

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

# used to create the base grid which the random traversal will venture through
class grid:
	def __init__(self):
		self.init_map()

	# create new randomized 500x500 grid
	def init_map(self):
		# 50% normal, 20% highway, 20% hard to traverse, 10% blocked
		self.num_cols = 500
		self.num_rows = 500

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

	# get a random starting point [x,y] in the grid towards the center
	def get_starting_point(self):
		while True:
			x = random.randint(150,350)
			y = random.randint(150,350)
			if self.data[y][x] is not "B": return [x,y]

	# gets the value at requested [x,y] coordinate
	def get_item(self,x,y):
		return self.data[y][x]

	# saves the grid as a .tsv file at specified location
	def save(self,path):
		f = open(path,"w")
		for y in range(self.num_rows):
			for x in range(self.num_cols):
				f.write(self.data[y][x])
				if x != self.num_cols-1:
					f.write("\t")
				else:
					f.write("\n")
		f.close()

def generate_data(num_maps=10,num_per_map=10,sequence_length=100):
	sys.stdout.write("\nGenerating map data...\n")

	# delete prior data directory
	targ_dir = "data"
	if os.path.exists(targ_dir): rmtree(targ_dir)
	os.makedirs(targ_dir)

	directions = ["U","L","D","R"] # up, left, down, right
	conditions = ["N","H","T","B"] # normal, highway, hard to traverse, blocked

	# create all of the num_maps randomized grid worlds
	for map_idx in range(num_maps):

		# create directory for current grid world
		map_dir = targ_dir+"/map_"+str(map_idx)
		os.makedirs(map_dir)

		cur_grid = grid() # create new dynamically size grid world

		# create all of the num_per_map sequences on the same grid world
		for seq_idx in range(num_per_map):

			# get new random starting location [x,y]
			initial_point = cur_grid.get_starting_point()

			observed_actions = [] # movements (U,L,D,R)
			observed_readings = [] # readings (N,H,T,B)
			agent_real_locations = [] # [x,y] for all agent real locations

			current_point = initial_point

			for step in range(sequence_length):

				sys.stdout.write("                                                     \r")
				sys.stdout.write("Map: "+str(map_idx)+", Sequence: "+str(seq_idx)+", Step: "+str(step)+"\r")
				sys.stdout.flush()

				new_action = random.choice(directions) # get random direction
				new_point = copy(current_point) # copy the current location

				should_actually_move = random.randint(1,10)

				# will actually make the reported movement
				if should_actually_move<=9:

					# alter the current location to take into account the new direction
					if new_action=="U": new_point[1]+=-1
					if new_action=="L": new_point[0]+=-1
					if new_action=="D": new_point[1]+= 1
					if new_action=="R": new_point[0]+= 1

					# get the condition of the new location
					actual_reading = cur_grid.get_item(new_point[0],new_point[1])

					# if the new movement puts us at a blocked cell
					if actual_reading=="B":
						# get the condition of the prior location
						actual_reading = cur_grid.get_item(current_point[0],current_point[1])
						# we cannot move into the blocked cell
						new_point = copy(current_point)

				# will report the movement but not actually move
				else:
					# get the condition of the prior location
					actual_reading = cur_grid.get_item(current_point[0],current_point[1])

				# now need to add the probability that the sensor reading was incorrect
				reported_reading = copy(actual_reading)

				should_corrupt = random.randint(1,100)

				# we will corrupt the reported reading
				if should_corrupt>90:
					possible_reported_conditions = ["N","H","T"]
					del possible_reported_conditions[possible_reported_conditions.index(actual_reading)]

					# randomly select from the two other possible conditions
					reported_reading = random.choice(possible_reported_conditions)

				observed_actions.append(new_action)
				observed_readings.append(reported_reading)
				agent_real_locations.append(new_point)
				current_point = new_point

			save_file = open(map_dir+"/traversal_"+str(seq_idx)+".txt","w")

			save_file.write("start_location - ("+str(initial_point[0])+","+str(initial_point[1])+")\n")

			save_file.write("~\n")

			# write out coordinates of consecutive points that the agent actually goes through
			for x,y in agent_real_locations:
				save_file.write("("+str(x)+","+str(y)+")\n")

			save_file.write("~\n")

			# write out the type of actions executed
			for action in observed_actions:
				save_file.write(action+"\n")

			save_file.write("~\n")

			# write out the sensor readings reported
			for reading in observed_readings:
				save_file.write(reading+"\n")

			save_file.write("~\n")

			save_file.close()

		# save the actual grid world (debugging)
		cur_grid.save(map_dir+"/grid_"+str(map_idx)+".tsv")
	sys.stdout.write("\n")

def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	generate_data()

if __name__ == '__main__':
	main()

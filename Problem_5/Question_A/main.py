import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

def main():
	# set of input commands
	actions = ["Right","Right","Down","Down"]
	# set of observed readings
	readings = ["N","N","H","H"]

	v_matrix = viterbi_matrix() # create object from helpers.py file
	v_matrix.init_observations(actions,readings,path=False,print_transition=False,print_condition=False) # execute

if __name__ == '__main__':
	main()

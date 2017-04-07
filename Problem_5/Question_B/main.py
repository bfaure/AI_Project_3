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

	# harder set i was testing on
	#actions = ["Left","Up","Up","Right","Down","Up","Right","Down","Down"]
	#readings = ["N","N","H","H","N","H","T","N","H"]

	v_matrix = viterbi_matrix() # construct object from helpers.py file
	v_matrix.init_observations(actions,readings,path=True,print_condition=False,print_transition=False) # execute operation

if __name__ == '__main__':
	main()

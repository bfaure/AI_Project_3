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

	#actions=["Left","Up","Right","Down"]
	#readings=["N","H","H","N"]

	#actions = ["Left","Left","Up","Right","Right","Down","Left"]
	#readings = ["N","N","H","H","T","N","N"]

	v_matrix = viterbi_matrix() # construct object from helpers.py file
	v_matrix.init_observations(actions,readings,path=True,print_condition=False,print_ancestors=False) # execute operation

if __name__ == '__main__':
	main()

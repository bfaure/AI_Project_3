import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

def viterbi(actions,readings):
	print("Constructing viterbi matrix...")
	v_matrix = viterbi_matrix()
	print("Constructed viterbi matrix.")
	v_matrix.init_observations(actions,readings)

def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	#actions = ["Right","Down","Down","Down","Down"]
	#readings = ["N","H","H","H","H"]

	#actions = ["Left","Up","Right","Right","Down","Down"]
	#readings = ["N","H","H","T","N","H"]

	#actions = ["Up","Right","Right"]
	#readings = ["H","H","T"]

	#actions = ["Left","Up","Up","Right","Down","Up","Right","Down","Down"]
	#readings = ["N","N","H","H","N","H","T","N","H"]

	#print(str(len(actions))+" actions, "+str(len(readings))+" readings")
	viterbi(actions,readings)

if __name__ == '__main__':
	main()
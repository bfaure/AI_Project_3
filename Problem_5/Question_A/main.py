import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node

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
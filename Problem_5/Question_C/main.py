import sys
import time
import random

from copy import deepcopy, copy

sys.path.append("..")
from helpers import viterbi_matrix, viterbi_node


def main():
	actions = ["Right","Right","Down","Down"]
	readings = ["N","N","H","H"]

	#actions = ["Right","Down","Down","Down","Down"]
	#readings = ["N","H","H","H","H"]

	#predict_location(actions,readings)

if __name__ == '__main__':
	main()
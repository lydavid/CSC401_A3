from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import scipy

def main():

	a = 1

	a = np.exp(a)  # e^1
	print(a)

	a = np.log(a)  # ln(e^1) = 1
	print(a)

	a_arr = np.ones(3)
	print(a_arr)

	# these two are equivalent
	b = scipy.misc.logsumexp(a_arr)
	print(b)
	b = np.log(np.exp(a) + np.exp(a) + np.exp(a))
	print("b=%f" % b)

	b = np.exp(b)
	print("b=%f" % b)

	c = np.exp(a) + np.exp(a) + np.exp(a)
	print("c=%f" % c)
	c = np.exp(np.log(a) + np.log(a) + np.log(a))
	print("c=%f" % c)

	b_arr = np.zeros(3)
	b_arr[0] = 1
	b_arr[1] = 2
	b_arr[2] = 3

	print(a_arr - b_arr)

	c_arr = np.ones((3, 3))
	print(c_arr / b_arr)

	print(np.exp(c_arr))

if __name__ == "__main__":
	main()
import sys, os
import numpy as np
from scipy import stats

# Taken from the textbook
# Encodes inputs of length 4 to transmit 7 bits
G = np.array([[1,0,0,0,1,0,1],
			  [0,1,0,0,1,1,0],
			  [0,0,1,0,1,1,1],
			  [0,0,0,1,0,1,1]])
H = np.array([[1,1,1,0,1,0,0],
			  [0,1,1,1,0,1,0],
			  [1,0,1,1,0,0,1]])

# Taken from suggested solutions to HW1
# Encodes inputs of length 5 to transmit 9 bits
suggestedG = np.array([[1,0,0,0,0,1,1,0,0],
					   [0,1,0,0,0,1,0,1,0],
					   [0,0,1,0,0,1,1,1,1],
					   [0,0,0,1,0,0,1,0,1],
					   [0,0,0,0,1,0,0,1,1]])
suggestedH = np.array([[1,1,1,0,0,1,0,0,0],
					   [1,0,1,1,0,0,1,0,0],
					   [0,1,1,0,1,0,0,1,0],
					   [0,0,1,1,1,0,0,0,1]])

# Generator matrix for R(1,3). i.e. Reed-Muller with r = 1, m = 3.
# Encodes inputs of length 4 to transmit 8 bits
R = np.array([[1,1,1,1,1,1,1,1],
			  [1,1,1,1,0,0,0,0],
			  [1,1,0,0,1,1,0,0],
			  [1,0,1,0,1,0,1,0]])

def readInputs(encoding_length):
	global original_length

	# Read in from file into a bit numpy array
	arr = np.fromfile(fin, np.uint8)
	bits = np.unpackbits(arr)

	# Pad zeros to back of bit array
	original_length = bits.shape[0]
	pad_length = encoding_length - (bits.shape[0] % encoding_length);
	bits.resize(bits.shape[0] + pad_length)

	# Convert bit numpy array to appropriate sized numpy matrix
	padded_size = bits.shape[0]
	reshaped = np.reshape(bits, (padded_size/encoding_length, encoding_length)).T
	# Each column of reshaped is a input to be encoded

	return reshaped

def writeOutput(decoded):
	global original_length
	bits = np.reshape(decoded.T, (decoded.shape[0]*decoded.shape[1],))
	bits = bits[:original_length]
	packed = np.packbits(bits)
	packed.tofile(fout)

def make_channel(flip_probability):
	def channel(block):
		transmit_length = block.shape[0]
		input_size = block.shape[1]
		noise = np.random.binomial(1,p=[flip_probability]*input_size,size=(transmit_length, input_size))
		output = block^noise
		return output
	return channel

def passBSC(channel, encoded):
	return channel(encoded)

def hamming_encode(blocks):
	return np.dot(G.T,blocks) % 2

def hamming_decode(blocks):
	z = np.dot(H,blocks)
	z = z % 2

	#z = 000, do nothing
	#z = 001, flip 7th bit
	mask = (z[:,0] == 0) * (z[:,1] == 0) * (z[:,2] == 1)
	blocks[6,mask] = (blocks[6,mask] + 1) % 2

	#z = 010, flip 6th bit
	mask = (z[:,0] == 0) * (z[:,1] == 1) * (z[:,2] == 0)
	blocks[5,mask] = (blocks[5,mask] + 1) % 2

	#z = 011, flip 4th bit
	mask = (z[:,0] == 0) * (z[:,1] == 1) * (z[:,2] == 1)
	blocks[3,mask] = (blocks[3,mask] + 1) % 2

	#z = 100, flip 5th bit
	mask = (z[:,0] == 1) * (z[:,1] == 0) * (z[:,2] == 0)
	blocks[4,mask] = (blocks[4,mask] + 1) % 2

	#z = 101, flip 1st bit
	mask = (z[:,1] == 0) * (z[:,1] == 0) * (z[:,2] == 1)
	blocks[0,mask] = (blocks[0,mask] + 1) % 2

	#z = 110, flip 2nd bit
	mask = (z[:,0] == 1) * (z[:,1] == 1) * (z[:,2] == 0)
	blocks[1,mask] = (blocks[1,mask] + 1) % 2

	#z = 111, flip 3rd bit
	mask = (z[:,0] == 1) * (z[:,1] == 1) * (z[:,2] == 1)
	blocks[2,mask] = (blocks[2,mask] + 1) % 2

	output = blocks[:4]
	return output

def suggested_encode(blocks):
	return np.dot(suggestedG.T, blocks) % 2

def suggested_decode(blocks):
	z = np.dot(suggestedH,blocks)
	z = z % 2

	#z = 0000, do nothing
	#z = 1100, flip 1st bit
	mask = (z[:,0] == 1) * (z[:,1] == 1) * (z[:,2] == 0) * (z[:,3] == 0)
	blocks[0,mask] = (blocks[0,mask] + 1) % 2

	#z = 1010, flip 2nd bit
	mask = (z[:,0] == 1) * (z[:,1] == 0) * (z[:,2] == 1) * (z[:,3] == 0)
	blocks[1,mask] = (blocks[1,mask] + 1) % 2

	#z = 1111, flip 3rd bit
	mask = (z[:,0] == 1) * (z[:,1] == 1) * (z[:,2] == 1) * (z[:,3] == 1)
	blocks[2,mask] = (blocks[2,mask] + 1) % 2

	#z = 0101, flip 4th bit
	mask = (z[:,0] == 0) * (z[:,1] == 1) * (z[:,2] == 0) * (z[:,3] == 1)
	blocks[3,mask] = (blocks[3,mask] + 1) % 2

	#z = 0011, flip 5th bit
	mask = (z[:,0] == 0) * (z[:,1] == 0) * (z[:,2] == 1) * (z[:,3] == 1)
	blocks[4,mask] = (blocks[4,mask] + 1) % 2

	#z = 1000, flip 6th bit
	mask = (z[:,0] == 1) * (z[:,1] == 0) * (z[:,2] == 0) * (z[:,3] == 0)
	blocks[5,mask] = (blocks[5,mask] + 1) % 2

	#z = 0100, flip 7th bit
	mask = (z[:,0] == 0) * (z[:,1] == 1) * (z[:,2] == 0) * (z[:,3] == 0)
	blocks[6,mask] = (blocks[6,mask] + 1) % 2

	#z = 0010, flip 8th bit
	mask = (z[:,0] == 0) * (z[:,1] == 0) * (z[:,2] == 1) * (z[:,3] == 0)
	blocks[7,mask] = (blocks[7,mask] + 1) % 2

	#z = 0001, flip 9th bit
	mask = (z[:,0] == 0) * (z[:,1] == 0) * (z[:,2] == 0) * (z[:,3] == 1)
	blocks[8,mask] = (blocks[8,mask] + 1) % 2

	# Else, don't bother doing anything, it's wrong.

	output = blocks[:5]
	return output

def reed_muller_encode(blocks):
	return np.dot(R.T,blocks) %2

def reed_muller_decode(blocks):
	x1 = R[1]
	x2 = R[2]
	x3 = R[3]
	notx1 = (~x1 % 2)
	notx2 = (~x2 % 2)
	notx3 = (~x3 % 2)

	# Characteristic vectors and coefficients of x3
	x3_1 = np.logical_and(x1,x2) % 2
	x3_2 = np.logical_and(x1,notx2) % 2
	x3_3 = np.logical_and(notx1,x2) % 2
	x3_4 = np.logical_and(notx1,notx2) % 2
	v3 = np.array([x3_1,x3_2,x3_3,x3_4])
	all3 = np.dot(v3, blocks) % 2
	# c3 = stats.mode(all3)[0]
	(mode3, count3) = stats.mode(all3)
	c3 = ((mode3 == 1) * (count3 > all3.shape[0]/2)) % 2

	# Characteristic vectors and coefficients of x2
	x2_1 = np.logical_and(x1,x3) % 2
	x2_2 = np.logical_and(x1,notx3) % 2
	x2_3 = np.logical_and(notx1,x3) % 2
	x2_4 = np.logical_and(notx1,notx3) % 2
	v2 = np.array([x2_1,x2_2,x2_3,x2_4])
	all2 = np.dot(v2, blocks) % 2
	# c2 = stats.mode(all2)[0]
	(mode2, count2) = stats.mode(all2)
	c2 = ((mode2 == 1) * (count2 > all2.shape[0]/2)) % 2

	# Characteristic vectors and coefficients of x1
	x1_1 = np.logical_and(x2,x3) % 2
	x1_2 = np.logical_and(x2,notx3) % 2
	x1_3 = np.logical_and(notx2,x3) % 2
	x1_4 = np.logical_and(notx2,notx3) % 2
	v1 = np.array([x1_1,x1_2,x1_3,x1_4])
	all1 = np.dot(v1, blocks) % 2
	# c1 = stats.mode(all1)[0]
	(mode1, count1) = stats.mode(all1)
	c1 = ((mode1 == 1) * (count1 > all1.shape[0]/2)) % 2

	# Calculate coefficient of 0th row
	coefficients = np.concatenate((c1, c2, c3),axis=0)
	dotted = np.dot(coefficients.T, np.array([x1, x2, x3])).T % 2
	all0 = (dotted + blocks) % 2

	# If more 1's, then 1. Otherwise, 0.
	# This also handles case when equal number of 1's and 0's -> Set to 0
	(mode0, count0) = stats.mode(all0)
	c0 = ((mode0 == 1) * (count0 > all0.shape[0]/2)) % 2

	decoded = np.concatenate([c0, c1, c2, c3],axis=0)
	decoded = decoded.astype(int)
	return decoded

def test_hamming():
	encode_length = 4
	fromFile = readInputs(encode_length)
	encoded = hamming_encode(fromFile)
	chn = make_channel(error_probability)
	transmitted = passBSC(chn, encoded)
	decoded = hamming_decode(transmitted)
	writeOutput(decoded)

def test_suggested():
	encode_length = 5
	fromFile = readInputs(encode_length)
	encoded = suggested_encode(fromFile)
	chn = make_channel(error_probability)
	transmitted = passBSC(chn, encoded)
	decoded = suggested_decode(transmitted)
	writeOutput(decoded)

def test_reed_muller():
	encode_length = 4
	fromFile = readInputs(encode_length)
	encoded = reed_muller_encode(fromFile)
	chn = make_channel(error_probability)
	transmitted = passBSC(chn, encoded)
	decoded = reed_muller_decode(transmitted)
	writeOutput(decoded)

def run():
	global original_length, padded_length, fin, fout, error_probability
	# Globals
	original_length = -1
	padded_length = -1
	fin = sys.stdin
	fout = sys.stdout
	error_probability = eval(sys.argv[1])
	run_type = sys.argv[2]
	
	if run_type == "hamming":
		test_hamming()
	elif run_type == "suggested":
		test_suggested()
	elif run_type == "reedmuller":
		test_reed_muller()
	else:
		print "Error"

run()
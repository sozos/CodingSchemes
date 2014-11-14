import math, os, time
import numpy as np

def nCr(n,r):
	f = math.factorial
	return f(n) / f(r) / f(n-r)

def hamming_block_errors(f_list):
	output = []
	for f in f_list:
		error = 1 - (1-f)**7 - nCr(7,1)*f*(1-f)**6
		output.append(round(error,3))
	return output

def suggested_block_errors(f_list):
	output = []
	for f in f_list:
		error = 1 - (1-f)**9 - nCr(9,1)*f*(1-f)**8
		output.append(round(error,3))
	return output

def reedmuller_block_errors(f_list):
	output = []
	for f in f_list:
		error = 1 - (1-f)**8 - nCr(8,1)*f*(1-f)**7
		output.append(round(error,3))
	return output

f_list = [1.0/2, 1.0/3, 1.0/4, 1.0/5, 1.0/6, 1.0/7, 1.0/8, 1.0/9, 1.0/10]
print "f: " + str(f_list)
print "Hamming: " + str(hamming_block_errors(f_list))
print "Suggested: " + str(suggested_block_errors(f_list))
print "Reed-Muller: " + str(reedmuller_block_errors(f_list))

def file_to_bit_matrix(filename, block_size):
	# Read in from file into a bit numpy array
	arr = np.fromfile(filename, np.uint8)
	bits = np.unpackbits(arr)

	# Pad zeros to back of bit array
	original_length = bits.shape[0]
	pad_length = block_size - (bits.shape[0] % block_size);
	bits.resize(bits.shape[0] + pad_length)

	# Convert bit numpy array to appropriate sized numpy matrix
	padded_size = bits.shape[0]
	reshaped = np.reshape(bits, (padded_size/block_size, block_size)).T
	# Each column of reshaped is a input to be encoded

	return reshaped

def file_to_int_matrix(filename, block_size):
	if block_size == 4:
		T = np.array([[1,2,4,8]])
	if block_size == 5:
		T = np.array([[1,2,4,8,16]])

	matrix = file_to_bit_matrix(filename, block_size)
	return np.dot(T,matrix)

def grab_files():
	path = "/Volumes/SOZOS HDD/cs3236/"
	infolder = "gen_inputs/"
	outfolder = "gen_outputs/"
	inpath = path + infolder
	outpath = path + outfolder

	input_paths = []
	output_paths = []

	for filename in os.listdir(inpath):
		input_paths += [inpath + filename]

	for filename in os.listdir(outpath):
		output_paths += [outpath + filename]

	input_dict = {}
	ham_dict = [{},{},{},{},{},{},{},{},{}]
	sug_dict = [{},{},{},{},{},{},{},{},{}]
	rm_dict = [{},{},{},{},{},{},{},{},{}]

	for fpath in input_paths:
		file_index = int(fpath.split('/')[5].split('.')[0])
		input_dict[file_index] = fpath

	for fpath in output_paths:
		# print fpath
		# print fpath.split('/')[5]
		file_index = int(fpath.split('_')[4].split('.')[0])
		f_index = int(fpath.split('_')[3])-2
		if fpath.split('/')[5][0:2] == 'ha':
			ham_dict[f_index][file_index] = fpath
		if fpath.split('/')[5][0:2] == 'su':
			sug_dict[f_index][file_index] = fpath
		if fpath.split('/')[5][0:2] == 'rm':
			rm_dict[f_index][file_index] = fpath

	return [input_dict, ham_dict, sug_dict, rm_dict]

def compare_blocks(block1, block2):
	assert block1.shape == block2.shape

	total = block1.shape[1]
	wrong = sum(sum(block1 == block2))
	return float(wrong)/total

def compute_block_error():
	[input_dict, ham_dict, sug_dict, rm_dict] = grab_files()

	output = []
	for f in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
		print 'f =', 1.0/f
		ham_mean = 0
		sug_mean = 0
		rm_mean = 0

		for i, in_name in input_dict.items():
			print 'i =', i
			# start = time.time()
			in_block4 = file_to_int_matrix(in_name, 4)
			in_block5 = file_to_int_matrix(in_name, 5)
			ham_block = file_to_int_matrix(ham_dict[f-2][i], 4)
			sug_block = file_to_int_matrix(sug_dict[f-2][i], 5)
			rm_block = file_to_int_matrix(rm_dict[f-2][i], 4)
			# print 'Time to convert: ', time.time() - start

			# start = time.time()
			ham_mean += compare_blocks(in_block4, ham_block)
			sug_mean += compare_blocks(in_block5, sug_block)
			rm_mean += compare_blocks(in_block4, rm_block)
			# print 'Time to compare: ', time.time() - start

		ham_mean /= len(input_dict)
		sug_mean /= len(input_dict)
		rm_mean /= len(input_dict)

		print [1.0/f, len(input_dict), ham_mean, sug_mean, rm_mean]
		output += [[1.0/f, len(input_dict), ham_mean, sug_mean, rm_mean]]

	return output

print(compute_block_error())

	# for filename in os.listdir(inpath):
		# print "\nFile: " + inpath + filename
		# input_name = inpath + filename
		# blocks1 = readInputs(file1, block_size)
		# blocks2 = readInputs(file2, block_size)



# [file_size_in_bytes, error_rate_str, len(os.listdir(inpath)), np.mean(hamming_output), np.mean(suggested_output), np.mean(reedmuller_output)]
















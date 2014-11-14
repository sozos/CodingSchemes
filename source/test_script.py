import sys, os, subprocess, re
import numpy as np

def generate_random_files(file_size_in_bytes, filename):
	with open(filename, 'wb') as fout:
		fout.write(os.urandom(file_size_in_bytes))

def run_unix_command(command):
	return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()

def count_bytes(filename):
	command = 'cat ' + filename + ' | wc -c'
	diff_bytes = run_unix_command(command)[0]
	return int(diff_bytes[:-1].strip())
	# return run_unix_command(command)

def compare_files(file1, file2):
	command = 'cmp -l ' + file1 + ' ' + file2 + ' | wc -l'
	diff_bytes = run_unix_command(command)[0]
	return int(diff_bytes[:-1].strip())

def run_code(error_rate_str, run_type, file_in, file_out):
	command = 'python code.py ' + error_rate_str + ' ' + run_type + ' < ' + file_in + ' > ' + file_out
	run_unix_command(command)

def fix_ppm_header(original_file, output_file):
	print "Using " + original_file + " to fix .ppm header for " + output_file
	command = 'dd if=' + original_file + ' of=' + output_file + ' bs=15 count=1 conv=notrunc'
	run_unix_command(command)

def gen_files(file_size_in_bytes, N):
	path = 'gen_inputs/'
	for i in range(0,N):
		input_name = path + str(i+1) + '.in'
		print "Generating File #" + str(i+1)
		generate_random_files(file_size_in_bytes, input_name)
	print "Done"

def gen_stats(error_rate_str):
	inpath = 'gen_inputs/'
	outpath = 'gen_outputs/'
	hamming_output = []
	suggested_output = []
	reedmuller_output = []
	
	print "\n==========="
	print "Error rate: " + str(eval(error_rate_str))
	print "==========="
	for filename in os.listdir(inpath):
		print "\nFile: " + inpath + filename
		input_name = inpath + filename
		file_size_in_bytes = count_bytes(input_name)

		ham_output_name = outpath + 'ham_' + re.sub('/','_',error_rate_str) + '_' + filename[:-2] + 'out'
		sug_output_name = outpath + 'sug_' + re.sub('/','_',error_rate_str) + '_' + filename[:-2] + 'out'
		rm_output_name = outpath + 'rm_' + re.sub('/','_',error_rate_str) + '_' + filename[:-2] + 'out'

		print "Running Hamming Code simulation..."
		run_code(error_rate_str, "hamming", input_name, ham_output_name)
		print "Running Suggested Code simulation..."
		run_code(error_rate_str, "suggested", input_name, sug_output_name)
		print "Running Reed-Muller Code simulation..."
		run_code(error_rate_str, "reedmuller", input_name, rm_output_name)
		
		hamming_output.append(compare_files(input_name, ham_output_name))
		suggested_output.append(compare_files(input_name, sug_output_name))
		reedmuller_output.append(compare_files(input_name, rm_output_name))
	return [file_size_in_bytes, error_rate_str, len(os.listdir(inpath)), np.mean(hamming_output), np.mean(suggested_output), np.mean(reedmuller_output)]

def image_test(error_rate_str):
	inpath = 'image_inputs/'
	outpath = 'image_outputs/'
	
	print "\n==========="
	print "Error rate: " + str(eval(error_rate_str))
	print "==========="
	for filename in os.listdir(inpath):
		print "\nFile: " + inpath + filename
		input_name = inpath + filename

		ham_output_name = outpath + 'ham_' + re.sub('/','_',error_rate_str) + '_' + filename
		sug_output_name = outpath + 'sug_' + re.sub('/','_',error_rate_str) + '_' + filename
		rm_output_name = outpath + 'rm_' + re.sub('/','_',error_rate_str) + '_' + filename

		print "Running Hamming Code simulation..."
		run_code(error_rate_str, "hamming", input_name, ham_output_name)
		fix_ppm_header(input_name, ham_output_name)
		print "Running Suggested Code simulation..."
		run_code(error_rate_str, "suggested", input_name, sug_output_name)
		fix_ppm_header(input_name, sug_output_name)
		print "Running Reed-Muller Code simulation..."
		run_code(error_rate_str, "reedmuller", input_name, rm_output_name)
		fix_ppm_header(input_name, rm_output_name)

	print "Done"

if (int(sys.argv[1]) == 1):
	size = int(sys.argv[2])
	N = int(sys.argv[3])
	gen_files(size, N)
elif (int(sys.argv[1]) == 2):
	statistics = []
	for error_rate_str in sys.argv[2:]:
		statistics.append(gen_stats(error_rate_str))

	for stats in statistics:
		print stats
else:
	for error_rate_str in sys.argv[2:]:
		image_test(error_rate_str)
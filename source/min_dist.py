import numpy as np

G = np.array([[1,0,0,0,1,0,1],
			  [0,1,0,0,1,1,0],
			  [0,0,1,0,1,1,1],
			  [0,0,0,1,0,1,1]])

suggestedG = np.array([[1,0,0,0,0,1,1,0,0],
					   [0,1,0,0,0,1,0,1,0],
					   [0,0,1,0,0,1,1,1,1],
					   [0,0,0,1,0,0,1,0,1],
					   [0,0,0,0,1,0,0,1,1]])

R = np.array([[1,1,1,1,1,1,1,1],
			  [1,1,1,1,0,0,0,0],
			  [1,1,0,0,1,1,0,0],
			  [1,0,1,0,1,0,1,0]])

def add_spaces(s):
	o = s[0]
	for i in range(1, len(s)):
		o = o + ' '
		o = o + s[i]
	return o

def int_to_vec(i, m):
	vec = np.array([np.fromstring(add_spaces(np.binary_repr(i, width=m)), dtype=int, sep=' ')]).T
	return vec

def calc_min_dist(A):
	(m, n) = np.shape(A)
	x = []

	for i in range (0, 2 ** n):
		vec = int_to_vec(i, n)
		if x == []:
			x = vec
		else:
			x = np.concatenate((x, vec), axis=1)

	d = np.dot(A,x) % 2

	dT = d.T
	(a,b) = np.shape(dT)
	ones = []
	for i in range(1,a-1):
		ones.append(np.count_nonzero(dT[i]))
	return min(ones)

print 'Minimum distance of G: ' + str(calc_min_dist(G.T))
print 'Minimum distance of suggestedG: ' + str(calc_min_dist(suggestedG.T))
print 'Minimum distance of R: ' + str(calc_min_dist(R.T))
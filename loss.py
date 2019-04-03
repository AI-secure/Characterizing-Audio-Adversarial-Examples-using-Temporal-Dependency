import numpy as np

def lcp(x, y):
	x = x.split()
	y = y.split()
	n = len(x)
	m = len(y)

	k = min(n, m)
	idx = 0
	for i in range(k):
		if (x[i] != y[i]):
			break
		idx = i + 1
	
	return idx * 1.0 / max(n, m)

def newWER(x, y):
	x = x.split()
	y = y.split()
	
	n = len(x)
	m = len(y)
	k = min(n, m)
	d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)

	for i in range(k + 1):
		for j in range(k + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	for i in range(1, k + 1):
		for j in range(1, k + 1):
			if (x[i - 1] == y[j - 1]):
				d[i][j] = d[i - 1][j - 1]
			else:
				S = d[i - 1][j - 1] + 1
				I = d[i][j - 1] + 1
				D = d[i - 1][j] + 1
				d[i][j] = min(S, I, D)
	
	print (d[k][k])
	return d[k][k] * 1.0 / k

def WER(x, y):
	x = x.split()
	y = y.split()
	
	n = len(x)
	m = len(y)
	print (n)
	print (m)
	d = np.zeros((n + 1) * (m + 1), dtype = np.uint8).reshape(n + 1, m + 1)

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			if (x[i - 1] == y[j - 1]):
				d[i][j] = d[i - 1][j - 1]
			else:
				S = d[i - 1][j - 1] + 1
				I = d[i][j - 1] + 1
				D = d[i - 1][j] + 1
				d[i][j] = min(S, I, D)
	
	print (d[n][m])
	return d[n][m] * 1.0 / n

def newCER(x, y):
	x = x.replace(" ", "")
	y = y.replace(" ", "")
	n = len(x)
	m = len(y)
	k = min(n, m)
	d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)

	for i in range(k + 1):
		for j in range(k + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	for i in range(1, k + 1):
		for j in range(1, k + 1):
			if (x[i - 1] == y[j - 1]):
				d[i][j] = d[i - 1][j - 1]
			else:
				S = d[i - 1][j - 1] + 1
				I = d[i][j - 1] + 1
				D = d[i - 1][j] + 1
				d[i][j] = min(S, I, D)
	
	return d[k][k] * 1.0 / k

def CER(x, y):
	x = x.replace(" ", "")
	y = y.replace(" ", "")
	n = len(x)
	m = len(y)
	d = np.zeros((n + 1) * (m + 1), dtype = np.uint8).reshape(n + 1, m + 1)

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			if (x[i - 1] == y[j - 1]):
				d[i][j] = d[i - 1][j - 1]
			else:
				S = d[i - 1][j - 1] + 1
				I = d[i][j - 1] + 1
				D = d[i - 1][j] + 1
				d[i][j] = min(S, I, D)
	
	return d[n][m] * 1.0 / n



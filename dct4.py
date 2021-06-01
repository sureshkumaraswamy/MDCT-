def dct4(N):
#Author K. Suresh sureshk@ieee.org
#Generate N x N DCT IV Matrix 
#Returns DCT4m = N x N DCT IV matrix
	import numpy 
	DCT4m=numpy.zeros((N,N))
	for n in range(N):
		for k in range(N):
			DCT4m[n,k]=numpy.cos(numpy.pi*(k+0.5)*(n+0.5)/N)
	DCT4m=(1/numpy.sqrt(N/2))*DCT4m
	return DCT4m
	

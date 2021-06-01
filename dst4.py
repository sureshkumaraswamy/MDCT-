def dst4(N):
#Author K. Suresh sureshk@ieee.org
#Generate N x N DST IV Matrix 
#Returns DCT4m = N x N DST IV matrix
	import numpy 
	DST4m=numpy.zeros((N,N))
	for n in range(N):
		for k in range(N):
			DST4m[n,k]=numpy.sin(numpy.pi*(k+0.5)*(n+0.5)/N)
	DST4m=(1/numpy.sqrt(N/2))*DST4m
	return DST4m
	

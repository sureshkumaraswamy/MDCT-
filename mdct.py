def mdct(x,f_size):
# Author : K. Suresh sureshk@ieee.org
# Generates MDCT of x, using frame size 2*f_size
# Uses sine window for overlap addition 
# Returns x = f_size x number of frame sized matrix containing MDCT of x

	import scipy as sp
	import numpy as np 
	from scipy.linalg import hankel
	from dct4 import dct4 
	blks = np.floor(x.size/f_size).astype(int)
	zpad=np.zeros((1,f_size)).astype(float)
	y=np.zeros((blks+2)*f_size)
	x=x[0:blks*f_size]
	y[0:f_size]=zpad
	y[f_size:x.size+f_size]=x
	y[x.size+f_size:y.size]=zpad
	x=y
	blks=blks+1
	X=np.zeros((2*f_size,blks)).astype(float)
	for k in range(blks):
		X[:,k]=x[k*f_size:k*f_size+2*f_size]
	X=np.mat(X).astype(float)
	window=np.zeros((2*f_size,1)).astype(float)
	for k in range(2*f_size):
		window[k] = np.sin((np.pi/2)*np.sin((np.pi/(2*f_size))*(0.5+k))**2);
	cpy=np.ones(blks).astype(float)
	WIN=np.mat(window)*np.mat(cpy)
	X=np.multiply(X,WIN)
	eyemat=np.identity(int(f_size//2)).astype(float)
	zeero=np.zeros((int(f_size/2),int(f_size/2)))
	hvec=np.zeros(f_size)
	hvec[f_size-1]=-1
	hkel=hankel(hvec)
	idx1=np.append(zeero,eyemat,0)
	idx2=np.append(-eyemat,zeero,0)
	idx3=np.append(idx1,hkel,-1)
	ALCAN=np.append(idx3,idx2,-1)
	ALCAN=np.mat(ALCAN).astype(float)
	DC=dct4(f_size)
	x=np.mat(DC)*np.mat(ALCAN)*np.mat(X)
	return x
	

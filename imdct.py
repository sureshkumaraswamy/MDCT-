def imdct(X):
# Author : K. Suresh 
# Computes inverse MDCT of columns of X
# where X is generated by mdct 
# Returns RECON=inverese MDCT 
# Uses Sine window for overlap addition 


	import scipy as sp
	import numpy as np 
	from scipy.linalg import hankel
	from dct4 import dct4 
	shp=X.shape
	f_size=shp[0]
	
	X=np.mat(X).astype(float)

	eyemat=np.eye(int(f_size/2))
	zeero=np.zeros((int(f_size/2),int(f_size/2)))
	hvec=np.zeros(f_size)
	hvec[f_size-1]=-1
	hkel=hankel(hvec)
	idx1=np.append(zeero,eyemat,0)
	idx2=np.append(-eyemat,zeero,0)
	idx3=np.append(idx1,hkel,-1)
	ALCAN=np.append(idx3,idx2,-1)
	ALCAN=np.mat(ALCAN).astype(float)
	ALCAN=np.transpose(ALCAN)
#	print ALCAN.shape

	window=np.zeros((2*f_size,1)).astype(float)
	DC=dct4(f_size)
	X=np.mat(ALCAN)*np.mat(DC)*np.mat(X)
	for k in range(2*f_size):
		window[k] = np.sin((np.pi/2)*np.sin((np.pi/(2*f_size))*(0.5+k))**2);
	cpy=np.ones(shp[1]).astype(float)
	WIN=np.mat(window)*np.mat(cpy)
	X=np.multiply(X,WIN)
	RECON=np.zeros((f_size,shp[1])).astype(float)
	RECON[:,1:shp[1]]=X[0:f_size,1:shp[1]]+X[f_size:2*f_size,0:shp[1]-1]
	RECON=np.reshape(RECON[:,1:shp[1]],((shp[1]-1)*f_size),'F')
	return RECON
	

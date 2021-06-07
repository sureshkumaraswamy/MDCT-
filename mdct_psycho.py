def mdct_psycho(f_size,samp_freq,x):
# mdct_psycho(f_size,samp_freq,x)
# Author Suresh K sureshk@ieee.org
# Psychoacoustic threshold estimation in MDCT domain
# f_size - MDCT frame size, samp_freq - Sampling frequency, x - input signal 
# returns [masking_threshold,mdct_power_spectrum,quantized_mdct_coefficients]
# Reference : - K. Suresh and T. V. Sreenivas, "Direct MDCT Domain Psychoacoustic Modeling," 2007 # IEEE International Symposium on Signal Processing and Information Technology, 2007, pp. 742-747, # doi: 10.1109/ISSPIT.2007.4458108.

	import numpy as np
	import math
	from mdct import mdct
	
	
############ To Compute Bark Band Edge Value ###############################	
	
	def bark_edge(f_size,samp_freq):
	#
	# Returns Bark edges 
	#
	
		fcmax=(samp_freq/2.0)
		center=np.zeros((1)).astype(float)
		center[0]=24.7/1.892
		band_edge=np.zeros((1)).astype(float)
		freq_reso=fcmax/f_size
		i=0
		a=1
		while(center[i]<=fcmax):
			i=i+1
			tmp_edge=center[i-1]+a*(12.35+0.054*center[i-1])
			band_edge_tmp=np.append(band_edge,tmp_edge)
			band_edge=band_edge_tmp
			tmp_center=(band_edge[i-1]+12.35)/0.946
			center_tmp=np.append(center,tmp_center)
			center=center_tmp
		band_edge=np.round(band_edge/freq_reso)
		band_edge_slice = band_edge[0:len(band_edge):2]
		band_edge_slice[0] = 0
		band_edge_slice[len(band_edge_slice)-1] = f_size
		return band_edge_slice
############## ERB Band Computation ###################
	def erb(a,samp_freq):
		fcmax=(samp_freq/2.0)
		center=np.zeros((1)).astype(float)
		center[0] = 24.7/1.892
		band_edge = 0
		i = 0
		while(center[i]<=fcmax):
			i=i+1
			band_edge = center[i-1] + a*(12.35+0.054*center[i-1])
			tmp_center = (band_edge + 12.35)/0.946
			center_tmp=np.append(center,tmp_center)
			center=center_tmp	
		return center
	def gammatone(f_size,samp_freq,k):
		center = erb(k,samp_freq)
		freq_reso = samp_freq/(2*f_size)
		fqz = np.arange(freq_reso,(samp_freq/2+freq_reso),freq_reso)
		g = np.zeros((len(fqz),len(center))).astype(float)
		pi = math.pi
		for i in range(0,len(center),1):
			fc = center[i]
			bw = 24.7 + 0.108*fc
			g[:,i] = np.power(1+np.power((fqz-fc)*3*pi/(4*bw),2),-1.5)
		return g
	def abs_thresh(f_size,samp_freq):
		freq_reso = samp_freq/(2*f_size)
		fqz = np.arange(freq_reso,(samp_freq/2+freq_reso),freq_reso)
		fqz = np.reshape(fqz,(f_size,1))
		th = np.zeros((f_size,1)).astype(float)
		min_thresh = 80*np.ones((f_size,1))
		th = np.minimum(3.64*np.power(fqz/1000,-0.8)-6.5*np.exp(-0.6*np.power(((fqz/1000)-3.3),2))+0.001*np.power(fqz/1000,4),min_thresh)
		#th =(3.64*np.power(fqz/1000,-0.8)-6.5*np.exp(-0.6*np.power((fqz/1000-3.3),2))+0.001*np.power(fqz/1000,4))
		return th
	
		

	def mdct_masker(x,samp_freq,gamma,Ca):

		f_size = len(x)				#length of frame
		Leff = min( (2*f_size)/(0.3*samp_freq), 1) #Effective length - detection detectability and duration
		px = np.zeros((f_size,1)).astype(float)
		px = 10*np.log10(np.multiply(x,np.conjugate(x)))                 # power spectrum of the signal
		freq_reso = samp_freq/(2*f_size) 	       # frequency resolution
		h_om = -abs_thresh(f_size,samp_freq)	       # outer ear model - inverse of threshold in quiet function
		sm = np.zeros((f_size,1)).astype(float)
		sm = px + h_om                                 #  outer ear fltered inputrsignal
		y=sm.shape;
		h_om = 10**(h_om/10)                           #  inverse log - converted back to power 
		[dum,n_filt] = gamma.shape                     #  number of filters in n_filt
		gamma_power =  10*np.log10(np.multiply(gamma,np.conjugate(gamma)))         # power spectrum of the filter
		masker_power= np.mean(np.power(10,(np.matmul(sm,np.ones((1,n_filt))) +gamma_power)/10),axis=0)
		c = erb(0,samp_freq)
		b_edge = bark_edge(f_size,samp_freq).astype(int)
		nb = len(b_edge)-1
		center_freq = np.round((b_edge[0:len(b_edge)-1]+b_edge[1:len(b_edge)])/2).astype(int)
		C_A = np.zeros((f_size,1)).astype(float)
		C_S = np.zeros((f_size,1)).astype(float)
###### Spectral Flatness Measure #####################
		for band in range (nb):
			alpha = np.ones((nb,1))
			gf = gamma[center_freq[band],:]
			numer = np.multiply(gf,np.conjugate(gf))*Leff*h_om[round(1000/freq_reso)]*0.0001589
			px_band = px[b_edge[band]:b_edge[band+1]]
			x_band = x[b_edge[band]:b_edge[band+1]]
			gm = 10**(0.1*np.mean(px_band))
			am = np.mean(np.multiply(x_band,x_band))
			sfm = 1
			if am != 0:
				sfm = gm/am
			if sfm != 0:
				alpha[band] = np.minimum(-10*np.log10(sfm)/60,1)
			if f_size <= 256:
				jnd_ref = np.maximum(np.minimum(2*alpha[band]*0.001,0.001),0.0005)
			else :
				jnd_ref = np.maximum(np.minimum(2*alpha[band]*0.01,0.01),0.002)
			denom = np.multiply(gf, np.conjugate(gf))*jnd_ref*h_om[round(1000/freq_reso)] + np.multiply(gf, np.conjugate(gf))*Ca[band]
			cs = 1/np.sum(np.divide(numer,denom))
			C_A[b_edge[band]:b_edge[band+1]] = Ca[band]
			C_S[b_edge[band]:b_edge[band+1]] = cs
		H = np.multiply(h_om,np.ones((1,n_filt)))
		Deno = f_size*np.ones((f_size,1))*masker_power + np.multiply(np.multiply(C_A, np.ones((1,len(masker_power)))),np.multiply(gamma,np.conjugate(gamma)))
		Numo = np.multiply(np.multiply(gamma,np.conjugate(gamma)),H)
		W1 = np.divide(Numo,Deno)
		w1 = np.sum(W1,axis=1)
		z = np.multiply(C_S,w1)
		b_width = b_edge[1:len(b_edge)] -  b_edge[0:len(b_edge)-1]
		scale = np.zeros((f_size,1))
		for k in range(len(b_width)):
			scale[b_edge[k]:b_edge[k+1]] = b_width[k]
		z = np.multiply(z,scale)
		step_size = np.sqrt(np.divide(np.full((f_size,1),12),z))
		qtzd_coeffs = np.round(np.divide(x,step_size)).astype(float)
		qtzd_coeffs= np.multiply(qtzd_coeffs,step_size)
		mask_thresh = -10*np.log10(z)
		return mask_thresh,px,qtzd_coeffs
	
	
	b=bark_edge(f_size,samp_freq)
	length_b = len(b)
	ki = np.zeros((length_b-1))
	ki=b[1:length_b]-b[0:length_b-1]
	md_coeffs = mdct(x,f_size)
	[fs,nframes] = md_coeffs.shape
	gamma=gammatone(f_size,samp_freq,0.5)
	[dummi,n_filt]=gamma.shape
	Ca = np.zeros((n_filt)).astype(float)
	Ca = (1/f_size)*np.sum(np.multiply(gamma,np.conjugate(gamma)),axis=0)
	mask_thresh = np.zeros(((f_size,1))).astype(float) 
	sig_power = np.zeros(((f_size,1))).astype(float)
	qtzd_coeffs = np.zeros(((f_size,1))).astype(float)
	for k in range(nframes): 
		[mask_thresh_tmp,sig_power_tmp,qtzd_coeffs_tmp] = mdct_masker(md_coeffs[:,k],samp_freq,gamma,Ca)
		mask_thresh=np.append(mask_thresh,mask_thresh_tmp,axis=1)
		sig_power=np.append(sig_power,sig_power_tmp,axis=1)
		qtzd_coeffs=np.append(qtzd_coeffs,qtzd_coeffs_tmp,axis=1)
	mask_thresh = mask_thresh[:,1:nframes+1]
	sig_power = sig_power[:,1:nframes+1] 
	qtzd_coeffs = qtzd_coeffs[:,1:nframes+1]
	return mask_thresh,sig_power,qtzd_coeffs



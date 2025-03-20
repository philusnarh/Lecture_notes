#!/usr/bin/env python

"""Created by ANSAH-NARH, T. (t.narh@gaecgh.org) on 24 June 2019"""

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import astropy.io.fits as fits, time

class Singular_Value_Decompostion:
    def __init__(self, data):
        self.data = data
        # self.thresh = thresh

    def svdecom(self):
	    t1 = time.time()
	    data1 = np.nan_to_num(self.data)
	    nchan = data1.shape[0]
	    U = np.zeros((2,2,nchan,nchan), dtype='complex')
	    S = np.zeros((2,2,nchan), dtype='complex')
	    npix = data1.shape[3]*data1.shape[4]
	    V = np.zeros((2,2,nchan,npix), dtype='complex')
	    for i in range(2):
	        for j in range(2):
	            print(i,j) 
	            data2d = np.zeros((nchan,npix), dtype='complex')
	            for k in range(nchan):
	                data2d[k,:] = data1[k,i,j,:,:].ravel()
	            (u,s,v) = svd(data2d, full_matrices=False)
	            U[i,j,:,:], S[i,j,:], V[i,j,:,:] = u,s,v
	    t2 = time.time()
	    print ("-- Time taken = %.2f minutes"%((t2-t1)/60.))
	    return U,S,V


    def svdrecon(self, thresh):
	    data1 = np.nan_to_num(self.data)
	    nchan = data1.shape[0]
	    model = np.zeros(data1.shape, dtype='complex')
	    U, S, V =  self.svdecom()
	    W = np.zeros((2,2,nchan,nchan), dtype='complex')
	    mse = np.zeros((2,2))
	    for i in range(2):
	        for j in range(2):
	            w = np.dot(U[i,j,:,:],np.diag(S[i,j,:]))
	            W[i,j,:,:] = w
	            recon = np.dot(w[:,:thresh],V[i,j,:thresh,:])
	            dat = data1[:,i,j,:,:]
	            mod = recon.reshape(dat.shape)
	            model[:,i,j,:,:] = mod
	            mse[i,j] = abs(np.sqrt(np.mean((dat-mod)**2))/np.mean(dat))
	    return model, mse, W
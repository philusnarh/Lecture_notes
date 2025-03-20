#!/usr/bin/env python
import astropy.io.fits as af, numpy as np, matplotlib.pyplot as pl
#http://localhost:8896/notebooks/EL/notebooks/meerkat/notebooks/meerkat_beams.ipynb
# http://localhost:8896/notebooks/EL/notebooks/meerkat/notebooks/meerkat_beam.ipynb
# 
import os, sys, glob, time, h5py
import mpl_toolkits.axes_grid1.axes_grid as axes_grid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
from matplotlib.pyplot import *
from sys import argv
import matplotlib.pyplot as plt #, healpy as hp
#import katdal, katpoint, katholog

import warnings
warnings.filterwarnings('ignore')

def J2M(f1,f2):
    # Convert 2x2xNxN Jones matrix into 4x4xNxN Mueller matrix.
    # If a=b, compute the autocorrelation version.
    a, b = np.load(f1), np.load(f2)
    M = np.zeros((a.shape[0],4,4,a.shape[3],a.shape[4]), dtype='c16')
    S = 0.5*np.matrix('1 1 0 0; 0 0 1 1j; 0 0 1 -1j; 1 -1 0 0')
    for f in range(a.shape[0]):
        print( f)
        for i in range(a.shape[3]):
            if i in [50,100,200,300,400,500]: print( i)
            for j in range(a.shape[4]):
                ab = np.kron(a[f,:,:,i,j],b[f,:,:,i,j].conj())
                M[f,:,:,i,j] = np.dot( np.dot(np.linalg.inv(S), ab), S )
    np.save(f1[:-4]+'_Mueller.npy', M)
    return M

# ==================
# P L O T S
# These have been moved to meerkat/utilities.py
# ==================

def plot1d(x,y,xscale=None,yscale='log',xlabel=None,ylabel=None,figsize=None,cls=False):
    fig = pl.figure(1, figsize=figsize)
    pl.plot(x,y)
    pl.yscale(yscale)
    pl.grid()
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    if cls==True:
        pl.show()
        pl.close()

def plot2d(d,xlabel=None,ylabel=None,clabel=None,figsize=None,norm=None,extent=None,aspect='auto',\
    cmap=None):
    """
    Plot 2d data as an image using matplotlib.pyplot.imshow
    """
    fig = pl.figure(1, figsize=figsize)
    plt.imshow(d, aspect='auto', norm=norm,extent=extent, cmap="gist_earth")
    cb = plt.colorbar()
    cb.ax.set_ylabel(clabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()

def plot2d_multi(d,h=None,nrc=None,name='', sh=True, vmin=None, vmax=None, cticks=None,fov=None,fs=6,title=None):
    """
    Plot multiple 2d images using Imagegrid and imshow
    Input: 4d array, first two dimensions specify number of 2d images
    """
    if h!=None: fov = h['CDELT2'] * h['NAXIS1']
    else: fov=fov
    try: N = d.shape[2]
    except: N = h['NAXIS1']
    ticks = list(map(int,np.linspace(0,N, 5)))
    if fov/2%2==0: labels = list(map(int,np.linspace(-fov/2., fov/2., 5))); print('lop')
    else: labels = np.round(np.linspace(-fov/2., fov/2., 5),1)
    if nrc==None: nrc = (d.shape[0], d.shape[1])
    fig = pl.figure(1, figsize=(fs,fs))
    g = axes_grid.ImageGrid(fig, 111, nrows_ncols=nrc, axes_pad=0., add_all=True,\
                            share_all=True, aspect=True, label_mode='1', cbar_mode='single', \
                            cbar_size='5%', cbar_pad=0.05)
    c=0
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            im = g[c].imshow(d[i,j,:,:], origin='lower', cmap=pl.cm.cubehelix,vmin=vmin,vmax=vmax)
            if (title is not '') & (c in range(d.shape[1])):
                g[c].set_title('%s' %title[c], fontsize=15)
            g[c].set_xticks(ticks)
            g[c].set_xticklabels(labels)
            g[c].set_yticks(ticks)
            g[c].set_yticklabels(labels)
            g[c].set_xlabel('$\\Delta\\theta$ [deg]', fontsize=12)
            g[c].set_ylabel('$\\Delta\\theta$ [deg]', fontsize=12)
            ax = plt.gca()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
            c+=1
        #g[i].text(5,40,str(round(ha[i],1)), fontsize=13)
    cb = g[0].cax.colorbar(im)
    #if cticks==None: cb.ax.set_yticks([0,-10,-20,-30])
    #else: cb.ax.set_yticks(cticks)
    cb.ax.set_ylabel('Power [decibel]', fontsize=14)
    pl.title(title, fontsize=18)
    if sh==True: pl.show()
    else: pl.savefig(name, dpi=80, bbox_inches="tight")
    pl.close()

def plot2d_sphere(map,figsize=None, title=None, cmap="gist_earth", coord=["G"],\
    dpar=30., dmer=30.):
    """
    Project 2d map on a sphere using healpy
    Input: .hdf files
    """
    fig = pl.figure(1, figsize=figsize)
    hp.visufunc.mollview(map, cmap=cmap, coord=["C"],title=title) # healpy
    hp.visufunc.graticule(dpar,dmer,coord) # plot graticule grid
    plt.show()
    plt.close()

# ================
# G R I D D I N G
# ================

def radial_profile(data, c=None):
    y, x = np.indices(data.shape)
    if not c: c = list(map(int, [x.max()/2., y.max()/2.]))
    r = np.hypot(x - c[0], y - c[1])
    r = r.astype(np.int)
    ind = np.where(r<=data.shape[0]/2.)
    tbin = np.bincount(r[ind], data[ind])
    nr = np.bincount(r[ind])
    rp = tbin / nr
    return rp

def binning(d,n,c=None):
    x = np.indices(d.shape)
    if not c: c = int(x.max()/2.)
    r = abs(x-c)
    s,e = np.histogram(r,weights=d.reshape(1,len(d)),bins=n)
    c,e = np.histogram(r,bins=n)
    return s/c

def spherical_profile(data, nbins, c=None, lcut=0, ucut=0):
    z, y, x = np.indices(data.shape)
    if not c: c = list(map(int, [x.max()/2., y.max()/2., z.max()/2.]))
    r = np.hypot(x-c[0], y-c[1], z-c[2])
    sum, e = np.histogram(r, weights=data, bins=nbins)
    count, edges = np.histogram(r, bins=nbins)
    PS = sum/count
    l = len(e[e <= lcut])
    if ucut==0: ucut=data.shape[0]/2.
    u = len(e[e <= ucut])
    print( len(PS), len(PS[l:u]))
    return PS[l:u], e[l:u]

if __name__=='__main__':
    if argv[1]=='j2m': J2M(argv[2], argv[3])

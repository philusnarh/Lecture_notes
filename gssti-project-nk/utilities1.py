#!/usr/bin/env python
import numpy as np, scipy.constants as sc, matplotlib.pyplot as plt, sys, matplotlib as mpl, os, math
#import katholog
from matplotlib.backends.backend_pdf import PdfPages
import mpl_toolkits.axes_grid1.axes_grid as axes_grid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
from sys import argv
from matplotlib.artist import setp, getp
import astropy.io.fits as fits, time
import datetime
from matplotlib import ticker
from numpy import pi, exp, sin, cos, tan

def circular_mask(d):
	ny, nx = (d.shape[-2],d.shape[-1])
	c = ny/2-1
	ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
	distance = np.sqrt((ix - c)**2 + (iy - c)**2)
	d[...,distance>c] = np.nan
	return d

def radial_profile(data, nbins, c=None, lcut=0, ucut=0):
	y, x = np.indices(data.shape)
	if not c: c = map(int, [x.max()/2., y.max()/2.])
	r = np.hypot(x - c[0], y - c[1])
	sum, e = np.histogram(r, weights=data, bins=nbins)
	count, edges = np.histogram(r, bins=nbins)
	PS = sum/count
	l = len(e[e <= lcut])
	if ucut==0: ucut=data.shape[0]/2.
	u = len(e[e <= ucut])
	return PS[l:u]

def binning(d,n,c=None):
	x = np.indices(d.shape)
	if not c: c = int(x.max()/2.)
	r = abs(x-c)
	s,e = histogram(r,weights=d.reshape(1,len(d)),bins=n)
	c,e = histogram(r,bins=n)
	return s/c, e

def radial_profile_x(data, c=None):
	y, x = np.indices(data.shape)
	if not c: c = list(map(int, [x.max()/2., y.max()/2.]))
	r = np.hypot(x - c[0], y - c[1])
	r = r.astype(np.int)
	ind = np.where(r<=data.shape[0]/2.)
	tbin = np.bincount(r[ind], data[ind])
	nr = np.bincount(r[ind])
	rp = tbin / nr
	return rp

def beamcube_to_npy(beams, filename, type='data'):
	"""
	Write katholog beamcube dataset to .npy
	"""
	nchan = np.size(beams)
	n = beams[0].Gx.shape[1]
	d = np.zeros((nchan,2,2,n,n), dtype='c16')
	for f in range(nchan):
		print ('Saving for channel %i'%f)
		if type=='data':
			d[f,0,0,:,:] = beams[f].Gx[0,...]
			d[f,0,1,:,:] = beams[f].Dx[0,...]
			d[f,1,0,:,:] = beams[f].Dy[0,...]
			d[f,1,1,:,:] = beams[f].Gy[0,...]
			np.save(filename+'%s_J.npy'%type, d)
		elif type=='model':
			d[f,0,0,:,:] = beams[f].mGx[0,...]
			d[f,0,1,:,:] = beams[f].mDx[0,...]
			d[f,1,0,:,:] = beams[f].mDy[0,...]
			d[f,1,1,:,:] = beams[f].mGy[0,...]
			np.save(filename+'%s_J.npy'%type, d)
	return d

def jones_to_mueller(f1,f2=None, sv=True):
	"""
	Convert 2x2xNxN Jones matrix into 4x4xNxN Mueller matrix.
	If f1=f2, compute the autocorrelation version for single dishes
	"""
	if f2==None: f2 = f1
	a, b = np.load(f1), np.load(f2)
	M = np.zeros((a.shape[0],4,4,a.shape[3],a.shape[4]), dtype='c16')
	S = 0.5*np.matrix('1 1 0 0; 0 0 1 1j; 0 0 1 -1j; 1 -1 0 0')
	for f in range(a.shape[0]):
#        print f
		for i in range(a.shape[3]):
			if i in [50,100,200,300,400,500]: print(i)
			for j in range(a.shape[4]):
				ab = np.kron(a[f,:,:,i,j],b[f,:,:,i,j].conj())
				M[f,:,:,i,j] = np.dot( np.dot(np.linalg.inv(S), ab), S )
	if sv==True: np.save(f1[:-4]+'_M.npy', M)
	return M

# ==================
# P L O T S
# ==================

def plot1d(x,y,xscale=None,yscale='log',xlabel=None,ylabel=None,figsize=None,cls=False):
	fig = plt.figure(1, figsize=figsize)
	plt.plot(x,y)
	plt.yscale(yscale)
	plt.grid()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()
	plt.close()

def plot2d(d,xlabel=None,ylabel=None,clabel=None,figsize=None,norm=None,extent=None, cmap=None, vrange=[None,None]):
	"""
	Plot 2d data as an image using matplotlib.pyplot.imshow
	"""
	fig = plt.figure(1, figsize=figsize)
	plt.imshow(d, origin='lower', norm=norm, extent=extent, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
	fig = pl.figure(1, figsize=figsize)
	pl.plot(x,y)
	pl.yscale(yscale)
	pl.grid()
	pl.xlabel(xlabel)
	pl.ylabel(ylabel)
	if cls==True:
		pl.show()
		pl.close()

def make_colorbar(g, fs, bba=(0, 0, 1, 1.07),w='100%',h='5%'):
	ax = g
	cax = inset_axes(ax,
					 width=w, # width = 10% of parent_bbox width
					 height=h, # height : 50%
					 loc=5,
					 bbox_to_anchor=bba,
					 bbox_transform=ax.transAxes,
					 borderpad=0
					 )
	#setp(getp(gca(),'xticklabels'), 'color', 'black', fontsize=fs)
	#for i in ['top','bottom','left','right']: cax.spines[i].set_linewidth(0.01)
	return cax

def make_colorbar_right(g, fs, bba=(0, 0, 1, 1.07)):
	ax = g
	cax = inset_axes(ax,
					 width="100%", # width = 10% of parent_bbox width
					 height="5%", # height : 50%
					 loc=5,
					 bbox_to_anchor=bba,
					 bbox_transform=ax.transAxes,
					 borderpad=0
					 )
	#setp(getp(gca(),'xticklabels'), 'color', 'black', fontsize=fs)
	#for i in ['top','bottom','left','right']: cax.spines[i].set_linewidth(0.01)
	return cax

def plot2d_multi(d, sh=True, vrange=None, diameter=6, cbticks=None, cblabel='Power [dB]', title='', cmap=plt.cm.nipy_spectral, norm=None):
	fig = plt.figure(1, figsize=(8,8))
	nrc = (d.shape[0], d.shape[1])
	ngrids = d.shape[0] * d.shape[1]
	extent = [-diameter/2, diameter/2, -diameter/2, diameter/2]
	g = axes_grid.ImageGrid(fig, 111, nrows_ncols=nrc, axes_pad=0.08, add_all=True,\
							share_all=False, aspect=True, label_mode='1', cbar_mode='none')

	c = 0
	ims = []
	for i in range(d.shape[0]):
		for j in range(d.shape[1]):
			# Fix vmin, vmax and color bar ticks
			if vrange!=None:
				if i!=j: vmin, vmax = vrange[2], vrange[3]
				elif i==j: vmin, vmax = vrange[0], vrange[1]
			elif vrange==None: vmin, vmax, cticks = None, None, None

			im = g[c].imshow(d[i,j,:,:], extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
			ims.append(im)

			# Color bar
			if ngrids==4: rights=[1,3]
			elif ngrids==16: rights=[3,7,11,15]
			if c in rights:
				cax = inset_axes(g[c], loc=5, width='5%', height='96%', bbox_to_anchor=(0.1,0,1,1), \
					bbox_transform=g[c].transAxes)
				cb = plt.colorbar(im, cax=cax, orientation='vertical')
				if vrange != None: cb.set_ticks(range(vmin, vmax+1,5))
				cb.ax.xaxis.set_ticks_position('top')
				cb.ax.set_ylabel('Power [decibel]')

			# Ticks and circles
			g[c].set_ylabel('Angular distance [deg]')
			g[c].set_xlabel('Angular distance [deg]')
			cn, o = (extent[0]+extent[1])/2, abs(extent[0])/3.
			g[c].add_artist(plt.Circle((cn,cn), o*1, color='black', linestyle='dashed', fill=False))
			g[c].add_artist(plt.Circle((cn,cn), o*2, color='black', linestyle='dashed', fill=False))
			g[c].add_artist(plt.Circle((cn,cn), o*3, color='black', linestyle='dashed', fill=False))
			c += 1
	g[0].text(0,d.shape[3]+15, title, fontsize=13)

	fig.subplots_adjust(left=0.05,right=.87,bottom=-.13,top=1.1)
	if sh==True: plt.show(); plt.close()

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


# ==================
# I / O
# ==================

def write_fits(dataset, beamcube, filename, freqs, diameter, scanner=None, tracker=None, elevation=61., azimuth=0.):
	# Convert nans to num
	# beamcube = np.nan_to_num(beamcube)

	# Calculate parameters from the dataset
	try:
		ID = dataset.filename.split('/')[-1][:-3]
		hdr['ID'] = ID
		el = dataset.env_el[0]
		az = np.mean(dataset.scanaz*(180./np.pi))
	except:
		el = elevation 
		az = azimuth 
	
	# Create header
	hdr = fits.Header()
	fMHz = np.array(freqs)*1e6
	diam = float(diameter)
	ctypes = ['AZIMUTH', 'ELEVATION', 'FREQ', 'X', 'Y']
	crvals = [az+diam/2, el-diam/2, fMHz[0], 0, 0]
	cdelts = [diam/beamcube.shape[-2], diam/beamcube.shape[-1], fMHz[1]-fMHz[0], 1, 1]
	cunits = ['deg', 'deg', 'Hz', '', '']
	for i in range(len(beamcube.shape)):
		ii = str(i+1)
		hdr['CTYPE'+ii] = ctypes[i]
		hdr['CRVAL'+ii] = crvals[i]
		hdr['CDELT'+ii] = cdelts[i]
		hdr['CUNIT'+ii] = cunits[i]
	try:
		hdr['TELESCOP'] = 'MeerKAT'
		hdr['OBSSTART'] = dataset.rawtime[0]
		hdr['OBSEND'] = dataset.rawtime[-1]
		hdr['DURATION'] = (dataset.rawtime[-1]-dataset.rawtime[0])/3600.
		hdr['SCANNER'] = scanner
		hdr['TRACKER'] = tracker
		hdr['TARGET'] = dataset.target.name
		hdr['TARGRA'] = dataset.target.radec()[0]*(180./np.pi)
		hdr['TARGDEC'] = dataset.target.radec()[1]*(180./np.pi)
	except: None
	hdr['DATE'] = str(datetime.datetime.now())
	
	# Write real and imag parts of data
	hdu = fits.PrimaryHDU(beamcube, header=hdr)
	hdu.writeto(filename+'.fits', clobber=True)

def split_into_eight(filename):
	d = fits.getdata(filename)
	d = np.nan_to_num(d)
	h = fits.getheader(filename)
	del h['C*4']
	del h['C*5']
	del h['C*6']
	P, C = ['R', 'I'], ['X', 'Y']
	for p in range(2):
		for i in range(2):
			for j in range(2):
				fits.writeto(filename[:-5]+'_%s%s_%s.fits'%(C[i],C[j],P[p]), d[p,i,j,...], h, clobber=True)

def psnr(img1, img2):
	# Peak -2 - signal - noise -ratio to measure the quality of reconstruction
    mse = np.mean( (img1 - img2) ** 2 )
    # if mse == 0:
    #     return 100
    PIXEL_MAX = img1.max()
    return 20.0 * math.log10(PIXEL_MAX / math.sqrt(mse))


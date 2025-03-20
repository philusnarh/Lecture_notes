import mpl_toolkits.axes_grid1.axes_grid as axes_grid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import astLib.astWCS as astWCS
import astLib.astCoords as astCoords
from scipy import ndimage

def circular_mask(d):
    ny, nx = (d.shape[-2],d.shape[-1])
    c = ny/2-1
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    distance = np.sqrt((ix - c)**2 + (iy - c)**2)
    d[...,distance>c] = np.nan
    return d


def plot2d_multi(d, sh=True, vrange=None, diameter=6, cbticks=None, cblabel='', title='', figsize=(10,10),mask_outsidegrid=False, cmap=plt.cm.jet):
    
#    cm = plt.get_cmap('%s' %cmp) #'jet')# plt.cm.cubehelix
    yt = ['XX', 'XY', 'YY']; tl = 0
    fig = plt.figure(1, figsize=figsize)
    nrc = (d.shape[0], d.shape[1])
    ngrids = d.shape[0] * d.shape[1]
    extent = [-diameter/2, diameter/2, -diameter/2, diameter/2]
    g = axes_grid.ImageGrid(fig, 111, nrows_ncols=nrc, axes_pad=0.00, add_all=True,\
                            share_all=False, aspect=True, label_mode='1', cbar_mode='none')

    c = 0
    ims = []
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            # Fix vmin, vmax and color bar ticks
            if vrange!=None:
                if i!=j: vmin, vmax = vrange[2], vrange[3]; print vmax
                elif i==j: vmin, vmax = vrange[0], vrange[1]
            elif vrange==None: vmin, vmax, cticks = None, None, None
            
	    if mask_outsidegrid is True:
	    	d[i,j,:,:] = circular_mask(d[i,j,:,:])
            im = g[c].imshow(d[i,j,:,:], extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            if (title is not '') & (c in range(d.shape[1])):
            	g[c].set_title('%s' %title[c], fontsize=15)
#            	print c
#            if i in [0, 5, 10]:
#            	 g[c].set_ylabel('%s' %yt[tl], fontsize=14)
#            	 tl +=1
#             print i
#            cmap=plt.cm.nipy_spectral
#	    print 'Yeh'
            ims.append(im)

            # Color bar
            if ngrids==4: rights=[1,3]
            elif ngrids==16: rights=[3,7,11,15]
            elif ngrids==20: rights=[4,9,14,19]
            elif ngrids==24: rights=[5,11,17,23]
            elif ngrids==12: rights=[3,7,11]
            #
            if c in rights:
                cax = inset_axes(g[c], loc=5, width='5%', height='96%', bbox_to_anchor=(0.1,0,1,1), \
                    bbox_transform=g[c].transAxes)
                cb = plt.colorbar(im, cax=cax, format = '%.0e', orientation='vertical')
                from matplotlib import ticker
                tick_locator = ticker.MaxNLocator(nbins=5)
                cb.locator = tick_locator
                cb.update_ticks()
                if vrange != None: cb.set_ticks(range(vmin, vmax+1,5))
                cb.ax.xaxis.set_ticks_position('top')
#                 if cblabel is not None:
                cb.ax.set_ylabel('%s' %cblabel)   #%Power [decibel] Power [dB]

            # Ticks and circles
            g[c].set_ylabel('Angular distance [deg]')
            g[c].set_xlabel('Angular distance [deg]')
            cn, o = (extent[0]+extent[1])/2, abs(extent[0])/3.
            g[c].add_artist(plt.Circle((cn,cn), o*1, color='black', linestyle='dashed', fill=False))
            g[c].add_artist(plt.Circle((cn,cn), o*2, color='black', linestyle='dashed', fill=False))
            g[c].add_artist(plt.Circle((cn,cn), o*3, color='black', linestyle='dashed', fill=False))
            c += 1
#    g[0].text(0,d.shape[3]+15, title, fontsize=13)

    fig.subplots_adjust(wspace=0,hspace=0,left=0.05,right=.87,bottom=-.13,top=1.1)
#     plt.tight_layout()
    if sh==True: plt.show(); plt.close()



def jones2mueller(xx_real, xx_imag, xy_real, xy_imag, yx_real, yx_imag, yy_real, yy_imag):
        #
        '''
        Extracts the Jones components into XX, XY, YX, YY
        to produce Mueller components'''
        
        #
        #   Generating Jones Terms
        #
        xx = xx_real + 1j*xx_imag
        xy = xy_real + 1j*xy_imag
        yx = yx_real + 1j*yx_imag
        yy = yy_real + 1j*yy_imag
        
        M = []
        #    
        #  Generating Mueller Terms
        #   
        m_ii = 0.5*(xx*np.conjugate(xx) + xy*np.conjugate(xy) + yx*np.conjugate(yx) + yy*np.conjugate(yy))
        m_iq = 0.5*(xx*np.conjugate(yx) - xy*np.conjugate(xx) + yx*np.conjugate(yy) + yy*np.conjugate(xy)) 
        m_iu = 0.5*(xx*np.conjugate(xy) + xy*np.conjugate(xx) + yx*np.conjugate(yy)+ yy*np.conjugate(yx)) 
        print m_iu.real.ravel()[2000:5000]
        print (yx*np.conjugate(yy)).real.ravel()[2000:5000]
        print (yx*np.conjugate(yy)).imag.ravel()[2000:5000]
        print (xy*np.conjugate(xx)).real.ravel()[2000:5000]
        print (xy*np.conjugate(xx)).imag.ravel()[2000:5000]
        m_iv = 0.5*1j*(xx*np.conjugate(xy) + yx*np.conjugate(yy) - xy*np.conjugate(xx) - yy*np.conjugate(yx))
        
        M.append([m_ii, m_iq, m_iu, m_iv])
        
        m_qi = 0.5*(xx*np.conjugate(yx) - xy*np.conjugate(xx) + yx*np.conjugate(yy) + yy*np.conjugate(xy))   
        m_qq = 0.5*(xx*np.conjugate(xx) - xy*np.conjugate(xy) - yx*np.conjugate(yx)+ yy*np.conjugate(yy)) 
        m_qu = 0.5*(xx*np.conjugate(xy) + xy*np.conjugate(xx) - yx*np.conjugate(yy) - yy*np.conjugate(yx)) 
        m_qv = 0.5*1j*(xx*np.conjugate(xy) - yx*np.conjugate(yy) - xy*np.conjugate(xx) - yy*np.conjugate(yx))
        
        M.append([m_qi, m_qq, m_qu, m_qv])
        
        m_ui = 0.5*(xx*np.conjugate(xy) + xy*np.conjugate(xx) + yx*np.conjugate(yy)+ yy*np.conjugate(yx))
        m_uq = 0.5*(xx*np.conjugate(xy) + xy*np.conjugate(xx) - yx*np.conjugate(yy) - yy*np.conjugate(yx))  
        m_uu = 0.5*(xx*np.conjugate(yy) + yy*np.conjugate(xx) + xy*np.conjugate(yx) + yx*np.conjugate(xy))         
        m_uv = 0.5*1j*(xx*np.conjugate(yy) - yy*np.conjugate(xx) - xy*np.conjugate(yx) - xy*np.conjugate(xy))
        
        M.append([m_ui, m_uq, m_uu, m_uv])
        
        m_vi = 0.5*1j*(xx*np.conjugate(xy) + yx*np.conjugate(yy) - xy*np.conjugate(xx) - yy*np.conjugate(yx))  
        m_vq = 0.5*1j*(xx*np.conjugate(xy) - yx*np.conjugate(yy) - xy*np.conjugate(xx) - yy*np.conjugate(yx))       
#        m_vu = 0.5*1j*(-xx*np.conjugate(yy) + yy*np.conjugate(xx) - xy*np.conjugate(yx) + xy*np.conjugate(xy))
        m_vu = 0.5*1j*(xx*np.conjugate(yy) - yy*np.conjugate(xx) - xy*np.conjugate(yx) - xy*np.conjugate(xy))    
        m_vv = 0.5*(xx*np.conjugate(yy) - yx*np.conjugate(xy) + yy*np.conjugate(xx) - xy*np.conjugate(yx))
        M.append([m_vi, m_vq, m_vu, m_vv])
        #
        
        return   np.array(M).real 


def jones_to_mueller(f1,f2=None, sv=True):
    """
    Convert 1X2x2xNxN Jones matrix into 4x4xNxN Mueller matrix.
    If f1=f2, compute the autocorrelation version for single dishes
    """
    if f2==None: f2 = f1
    # a, b = np.array(f1) #np.load(f1), np.load(f2)
    a = np.array(f1)
    b = np.array(f1)
    M = np.zeros((a.shape[0],4,4,a.shape[3],a.shape[4]), dtype='c16')
    S = 0.5*np.matrix('1 1 0 0; 0 0 1 1j; 0 0 1 -1j; 1 -1 0 0')
    for f in range(a.shape[0]):
#        print f
        for i in range(a.shape[3]):
#            if i in [50,100,200,300,400,500]: print i
            for j in range(a.shape[4]):
                ab = np.kron(a[f,:,:,i,j],b[f,:,:,i,j].conj())
                M[f,:,:,i,j] = np.dot( np.dot(np.linalg.inv(S), ab), S )
    if sv==True: np.save(f1[:-4]+'_M.npy', M)
    return M


# ++++++++++ interpolating to have same size ++++++++

#
def rescale_data_size(data, newsizex, newsizey):
    dshape = data.shape
    # define new size
    outKSize_x = newsizex
    outKSize_y = newsizey
    
    # Rescale Data Size
    x_old = np.linspace(-dshape[0]/2., dshape[0]/2., dshape[0])      
    y_old = np.linspace(-dshape[-1]/2., dshape[-1]/2., dshape[-1])
    xnew = np.linspace(x_old.min(), x_old.max(), outKSize_x)
    ynew =  np.linspace(y_old.min(), y_old.max(), outKSize_y)
    
    # Perform Interpolation
    interp_Fxn = interpolate.RectBivariateSpline(np.sort(x_old),np.sort(y_old),data, kx=3,ky=3) 

    return interp_Fxn(xnew,ynew)



def beam_offset(hdr, bm_data, beam_shift_arcsec=10):
    px = np.linspace(0, hdr["NAXIS1"], hdr["NAXIS1"])
    py = np.linspace(0, hdr["NAXIS1"], hdr["NAXIS1"])
    x, y = np.meshgrid(px,py)
#     print px[127], py[127]
    header = hdr
    #  WCS coordinates in format [RADeg, decDeg]
    wcs1 = astWCS.WCS(header,mode="pyfits") 
    wcscoord_deg = [wcs1.pix2wcs(i,j) for i, j in zip(x.ravel(), y.ravel())]
    wcscoord_deg = np.array(wcscoord_deg)
    new_wcscoord_deg= [astCoords.shiftRADec(ra1=ra, 
                                            dec1=dec,
                                            deltaRA=beam_shift_arcsec, deltaDec=beam_shift_arcsec) 
                                                         for ra, dec in zip(wcscoord_deg[:,0], wcscoord_deg[:,1]) ]
    new_wcscoord_deg = np.array(new_wcscoord_deg)
    new_pxcoord = [wcs1.wcs2pix(newRA,newDec) for newRA, newDec in zip(np.array(new_wcscoord_deg)[:,0], new_wcscoord_deg[:,1])]
    new_pxcoord = np.array(new_pxcoord)
    Xc_beam = np.array(new_pxcoord)[:,0].reshape(hdr["NAXIS1"], hdr["NAXIS1"])
    Yc_beam = np.array(new_pxcoord)[:,1].reshape(hdr["NAXIS1"], hdr["NAXIS1"])
    
    img = ndimage.map_coordinates(bm_data.T, [Xc_beam, Yc_beam], mode = 'nearest', order=3)
#     print img
    return img
    

from utilities import *

def radial_profiles(data, model, extent=3, view=True):
	residual = abs(data)-abs(model)
	N = int(data.shape[-1]/2-2)
	# print(True)
	x = np.linspace(0,extent, N)
	fig, ax = plt.subplots(1,4, figsize=(15,4), dpi=100)
	c = 0
	for i in range(2):
		for j in range(2):
			res = residual[i,j,...]
			res = np.nan_to_num(res)
			mu, sig = np.mean(res)*1e2, np.std(res)*1e2
			ax[c].loglog(x, abs(radial_profile_x(data[i,j,...])[:N]), 'x', markersize=3, label='Measurement')
			ax[c].loglog(x, abs(radial_profile_x(model[i,j,...])[:N]), '-', label='Model')
			ax[c].loglog(x, abs(radial_profile_x(res)[:N]), '-', label='Mean error')
			leg = ax[c].legend(loc='best', fontsize=15)
			leg.get_frame().set_facecolor('none')
			leg.get_frame().set_linewidth('0.')
			if i == 0 and j == 0:
				ax[c].set_ylabel('Intensity Profile', fontsize=20)
			ax[c].set_xlabel('Radius [deg]', fontsize=20)
			ax[c].set_title("$\\mu$=%.2f, $\\sigma$=%.2f"%(mu,sig), fontsize=20)
			for tick in ax[c].xaxis.get_major_ticks(): tick.label.set_fontsize(20) 
			for tick in ax[c].yaxis.get_major_ticks(): tick.label.set_fontsize(20)
#            ax[c].locator_params(nbins=4, axis='y')
			ax[c].set_xlim([1e-1,3.1])
#            if i==j: ax[c].set_ylim([1e-5,1.0])
#            if i!=j: ax[c].set_ylim([1e-5,1.0])
			c+=1
	plt.subplots_adjust(wspace=0.2, hspace=0.05)
	fig.tight_layout()
	if view==True: plt.show()
	plt.close()

#def gof_plot(data, model, coeffs=None, vrange=[-20,0, -30,-15], extent = [-3,3,-3,3], view=True, title=''):
#    model = circular_mask(model)
#    data = circular_mask(data)
#    fontsize = 14
#    fig = plt.figure(1, figsize=(15,10))
#    g = axes_grid.ImageGrid(fig, 111, nrows_ncols=(3,4), axes_pad=0.0, add_all=True, share_all=False, aspect=True, \
#                            label_mode='L', cbar_mode='none')
#    cmap = plt.cm.jet #cubehelix  #nipy_spectral
#    c = 0
#    ims = []
#    residual = abs(data) - abs(model)
#    for i in range(2):
#        for j in range(2):
#            if i==j: vmin, vmax = vrange[0], vrange[1]
#            else: vmin, vmax = vrange[2], vrange[3]
##            im = g[c].imshow(10*np.log10(abs(data[i,j,:,:])), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
#            im = g[c].imshow(data[i,j,:,:], origin='lower', cmap=cmap, extent=extent)
#            g[c].set_ylabel('Data', fontsize=14)
#            cax = inset_axes(g[c], loc=1, height='5%', width='85%', bbox_to_anchor=(-0.04,0,1,1.1), bbox_transform=g[c].transAxes)
#            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
#            cb.ax.xaxis.set_ticks_position('top')
#            cb.ax.xaxis.set_label_position('top')
#            cb.locator = ticker.MaxNLocator(nbins=4)
#            cb.update_ticks()

#            im = g[c+4].imshow(10*np.log10(abs(model[i,j,...])), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
#            g[c+4].set_ylabel('Model', fontsize=14)

#            if i==j: vmin, vmax = -3,3
#            if i!=j: vmin, vmax = -.9,1.6
#            im = g[c+8].imshow(residual[i,j,...]*1e2, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
#            g[c+8].set_ylabel('Residual', fontsize=14)
#            cax = inset_axes(g[c+8], loc=4, height='5%', width='90%', bbox_to_anchor=(0,-.1,1,1.1), bbox_transform=g[c+8].transAxes)
#            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
#            cb.locator = ticker.MaxNLocator(nbins=5)
#            cb.update_ticks()

#            c+=1

#    for i in range(12):
#        g[i].set_xticklabels([])
#        g[i].set_yticklabels([])
#        
#    fig.suptitle(title, fontsize=fontsize)

#    if view==True: plt.show()
#    plt.close()
	
def gof_plot2(data, model, coeffs=None, vrange=[-20,0, -30,-15], extent = [-3,3,-3,3], view=True, title='',  
		cmap=plt.cm.jet, fontsize = 18, ylabels=['Original Image', 'Model Image', 'Residual Image']):
#
#    import matplotlib
#    matplotlib.rc('font', family='serif', serif='cm10')
#    matplotlib.rc('text', usetex=True)
#    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
#
#    model = model.real
#    data = data.real
	# model = circular_mask(model)
	# data = circular_mask(data)   
	
	
	fig = plt.figure(1, figsize=(15,10))
	g = axes_grid.ImageGrid(fig, 111, nrows_ncols=(3,4), axes_pad=0.0, add_all=True, share_all=False, aspect=True, \
							label_mode='L', cbar_mode='none')
#    cmap = plt.cm.nipy_spectral
	c = 0
	ims = []
	residual = data - model #abs(data) - abs(model)
	for i in range(2):
		for j in range(2):
			if i==j: vmin, vmax = vrange[0], vrange[1]
			else: vmin, vmax = vrange[2], vrange[3]
#            if i != j:
#            	data[i,j,...]*=150
#            	model[i,j,...]*=150
			im = g[c].imshow(data[i,j,:,:], origin='lower', cmap=cmap, extent=extent)
#            im = g[c].imshow(20*np.log10(abs(data[i,j,:,:])), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
			g[c].set_ylabel('%s' %ylabels[0], fontsize=fontsize)
			cax = inset_axes(g[c], loc=1, height='5%', width='85%', bbox_to_anchor=(-0.04,0,1,1.1), bbox_transform=g[c].transAxes)
			cb = plt.colorbar(im, cax=cax, orientation='horizontal')
			cb.ax.xaxis.set_ticks_position('top')
			cb.ax.xaxis.set_label_position('top')
			cb.locator = ticker.MaxNLocator(nbins=4)
			cb.update_ticks() 
			im = g[c+4].imshow(model[i,j,...], origin='lower', cmap=cmap, extent=extent)
#            im = g[c+4].imshow(20*np.log10(abs(model[i,j,...])), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
			g[c+4].set_ylabel('%s' %ylabels[1], fontsize=fontsize) 
			im = g[c+8].imshow(residual[i,j,...], origin='lower', extent=extent, cmap=cmap)  	    
		
#    	    im = g[c+8].imshow(residual[i,j,...], origin='lower', vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
			g[c+8].set_ylabel('%s' %ylabels[2], fontsize=fontsize)
			cax = inset_axes(g[c+8], loc=4, height='5%', width='90%', bbox_to_anchor=(0,-.1,1,1.1), bbox_transform=g[c+8].transAxes)
#            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
			cb = fig.colorbar(im, cax=cax, orientation='horizontal')
			cb.formatter.set_powerlimits((0, 0))
			cb.locator = ticker.MaxNLocator(nbins=5)
			cb.update_ticks()

			c+=1

	for i in range(12):
		g[i].set_xticklabels([])
		g[i].set_yticklabels([])
		
	fig.suptitle(title, fontsize=fontsize)

	if view==True: plt.show()
	plt.close()

	radial_profiles(data, model, extent=2, view=True)

	fig, ax = plt.subplots(1,4, figsize=(20,4))
	c=0
	for i in range(2):
		for j in range(2):
			res = residual[i,j,...]
			res = res[~np.isnan(res)]*1e2
			mu, sig = np.mean(res), np.std(res)
			n,b,p = ax[c].hist(res, bins=60, density=True, facecolor='green', alpha=0.75, 
				label="%.2f, %.2f"%(mu,sig), edgecolor='black')
			# 
			from scipy.stats import norm
			# (mu, std) = norm.fit(y_trans)
			# Plot the PDF.
			xmin, xmax = ax[c].set_xlim()
			x = np.linspace(xmin, xmax, 100)
			p = norm.pdf(x, mu, sig)
			ax[c].plot(x, p, 'r--', linewidth=4)
			#leg = ax[c].legend(loc='best')
			#leg.get_frame().set_facecolor('none')
			#leg.get_frame().set_linewidth('0.')
			ax[c].set_xlabel('Residual [%]', fontsize = 20)
			ax[c].set_title("$\\mu$=%.2f, $\\sigma$=%.2f"%(mu,sig), fontsize=20)
			ax[c].locator_params(nbins=4, axis='x')
			for tick in ax[c].xaxis.get_major_ticks(): tick.label.set_fontsize(20) 
			for tick in ax[c].yaxis.get_major_ticks(): tick.label.set_fontsize(20)
			c+=1
	if view==True: plt.show()
	fig.tight_layout()
	plt.close()

	try:
		fig, ax = plt.subplots(1,4, figsize=(20,4), dpi=100)
		c=0
		for i in range(2):
			for j in range(2):   
				print(i,j) 
				print(sorted(coeffs[i,j,:], reverse=True))         
				# ax[c].semilogy(range(coeffs.shape[-1]), coeffs[i,j,:], 'o-', mfc='None', markersize=5) #'o-', mfc='None',
				ax[c].semilogy(range(coeffs.shape[-1]), sorted(coeffs[i,j,:], reverse=True), 'o-', mfc='None', markersize=5)
				# ax[c].plot(range(coeffs.shape[-1]), sorted(coeffs[i,j,:], reverse=True), 'o-', mfc='None', markersize=5)
				ax[c].set_xlim([-1,coeffs.shape[-1]+1])
				ax[c].set_xlabel('Mode number', fontsize=20) 
				if i == 0 and j == 0: 
					ax[c].set_ylabel('Coefficients', fontsize=20)
					ax[c].set_title(r'HH', fontsize=20) 
				if i == 0 and j == 1: 
					ax[c].set_title(r'HV', fontsize=20)
				if i == 1 and j == 0: 
					ax[c].set_title(r'VV', fontsize=20) 
				if i == 1 and j == 1: 
					ax[c].set_title(r'VH', fontsize=20)
	#                ax[c].set_ylabel('Coefficients', fontsize=20)
				for tick in ax[c].xaxis.get_major_ticks(): tick.label.set_fontsize(20) 
				for tick in ax[c].yaxis.get_major_ticks(): tick.label.set_fontsize(20)
				c+=1
		if view==True: plt.show()
		fig.tight_layout()
		plt.close()
	except: None

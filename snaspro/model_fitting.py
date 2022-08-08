#
import h5py as h5
'''
These model fitting utility functions are produced
by T. Ansah-Narh

Friday 05 August 2022
'''

# ++++++++++++++++++++++++++++++
# << Import Python libraries >>
# ++++++++++++++++++++++++++++++

import numpy as np
import corner
from lmfit import Parameters, fit_report, minimize
import math, os, random
from scipy import optimize, signal
from lmfit import models

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
np.random.seed(seed=4)

#

# ignore used to produce images for blog
image_dir = "images"

def plot_to_blog(fig, figure_name):
	filename = os.path.expanduser(f'{image_dir}/{figure_name}')
	fig.savefig(filename)
	return filename

def generate_model(spec):
	'''This function generates composite model for fitting'''
	composite_model = None
	params = None
	x = spec['x']
	y = spec['y']
	x_min = np.min(x)
	x_max = np.max(x)
	x_range = x_max - x_min
	y_max = np.max(y)
	for i, basis_func in enumerate(spec['model']):
		prefix = f'm{i}_'
		model = getattr(models, basis_func['type'])(prefix=prefix)
		# for now VoigtModel has gamma constrained to sigma
		if basis_func['type'] in ['GaussianModel', 
									'LorentzianModel', 'VoigtModel']:

			model.set_param_hint('sigma', min=1e-6, max=x_range)
			model.set_param_hint('center', min=x_min, max=x_max)
			model.set_param_hint('height', min=1e-6, max=1.1*y_max)
			model.set_param_hint('amplitude', min=1e-6)
			# default guess is horrible!! do not use guess()
			default_params = {
				prefix+'center': x_min + x_range * random.random(),
				prefix+'height': y_max * random.random(),
				prefix+'sigma': x_range * random.random()
			}
		else:
			raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
		if 'help' in basis_func:  # allow override of settings in parameter
			for param, options in basis_func['help'].items():
				model.set_param_hint(param, **options)
		model_params = model.make_params(**default_params, **basis_func.get('params', {}))
		if params is None:
			params = model_params
		else:
			params.update(model_params)
		if composite_model is None:
			composite_model = model
		else:
			composite_model = composite_model + model
	return composite_model, params

def update_spec_from_peaks(spec, model_indicies, 
							peak_widths=(10, 25), **kwargs):
	x = spec['x']
	y = spec['y']
	x_range = np.max(x) - np.min(x)
	peak_indicies = signal.find_peaks_cwt(y, peak_widths)
	# print(peak_indicies)
	np.random.shuffle(peak_indicies)
	for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
		model = spec['model'][model_indicie]
		if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
			params = {
				'height': y[peak_indicie],
				'sigma': x_range / len(x) * np.min(peak_widths),
				'center': x[peak_indicie]
			}
			if 'params' in model:
				model.update(params)
			else:
				model['params'] = params
		else:
			raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
	return peak_indicies




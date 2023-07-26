=======================================
NBLE: Natural Background Level estimator
=======================================


**NBLE** is a Python source code for estimating the natural background levels of geochemical species with multiple sources. Basically, the code performs two main tasks: 

1. Produces the morphorlogical structure of the geochemical data using the Kernel Density Estimation (KDE) approach as presented below;

.. image:: https://raw.githubusercontent.com/lonona/NBL/master/gmm/image/kde.png
   :alt: 
   :align: center

2. Estimates the NBL by using the Gaussian Mixture Model to decompose the geochemical data into components as displayed below; 

.. image:: https://raw.githubusercontent.com/lonona/NBL/master/gmm/image/gmm.png
   :alt: 
   :align: center


To run the <<model_util>> script you may install **Anaconda** from the `Official link <https://www.anaconda.com/products/individual>`_. Click on this `template <https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/3u5igqyx602c1cd/nble.ipynb>`_ to obtain the plots above. 
import matplotlib as mpl
#mpl.use('agg')
import mesa_reader as mr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import glob
import re
from cycler import cycler
import sys
# My codes
sys.path.insert(0, '/home/castaneda/python/libs/mios')
import utilities
from matplotlib.offsetbox import AnchoredText # For positioning annotations
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from IPython import embed
from os import remove
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.ndimage.interpolation import shift
import pickle
import peakutils

def hist_function(x,bins,normalized_heights):
    bin_host = np.searchsorted(bins,x)
    # add zeros at the beginning and the end
    normalized_heights = np.insert(normalized_heights,len(normalized_heights),0) 
    normalized_heights = np.insert(normalized_heights,0,0)
    return normalized_heights[bin_host]

# Read the results from the glitch fits
f = open('tau_histograms.bin','rb')
hist = pickle.load(f)
f.close()
f = open('trackings.bin','rb')
trackings = pickle.load(f)
f.close()

# Variale where to save the results summary results
offset_results = dict()

for profile,results in hist.items():

    # Find BCZ tau in the fits
    bins = results['hist']['BCZ']['bins']
    bins *= 1e6
    pdf = results['hist']['BCZ']['pdf']
    tau_tmp = np.linspace(bins[0], bins[-1], 1000)
    pdf_tmp = hist_function(tau_tmp, bins, pdf)

    ind = int( np.median( np.where(pdf_tmp == pdf_tmp.max()) ) )
    tau_BCZ_fit = tau_tmp[ind]
    
    # Find BCZ tau in the models
    ind = np.where(np.array(trackings['profile']) == profile)
    tau_BCZ_model = trackings['conv_boundaries_acoustic_depth'][1][ind]
    
    # Calculate offset
    offset = tau_BCZ_fit - tau_BCZ_model
    
    # save results
    offset_results[profile] = offset[0]
    
    # print
    print(profile, 'offset TAU BCZ =', offset[0], 'seconds')

# File where to save the results
savefile = open('tau_BCZ_offset.bin','wb')
pickle.dump(offset_results, savefile)
savefile.close()
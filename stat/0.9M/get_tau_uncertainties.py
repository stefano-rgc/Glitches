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

# Variale where to save the results summary results
uncertainties = dict()

for profile,results in hist.items():

    # Find BCZ tau in the fits
    bins = results['hist']['BCZ']['bins']
    bins *= 1e6
    pdf = results['hist']['BCZ']['pdf']
    if np.any(np.isnan(pdf)):
        # Store
        tau_BCZ = {'lower_2sigma':np.nan,
                   'lower_1sigma':np.nan,
                   'upper_1sigma':np.nan,
                   'upper_2sigma':np.nan,
                   'median':np.nan}
    else:        
        area = np.sum( pdf * np.diff(bins) )
        pdf = pdf/area
        tau_tmp = np.linspace(bins[0], bins[-1], 10000)
        pdf_tmp = hist_function(tau_tmp, bins, pdf)    
        delta = np.median( np.diff(tau_tmp) )
        prob = delta * pdf_tmp
        cumprob = np.cumsum(prob)
        ind = np.argmin(np.abs(cumprob-0.025)) # percentile 2.5
        tau_BCZ_fit025 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.16)) # percentile 16
        tau_BCZ_fit16 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.50)) # percentile 50
        tau_BCZ_fit50 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.84)) # percentile 84
        tau_BCZ_fit84 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.975)) # percentile 97.5
        tau_BCZ_fit975 = tau_tmp[ind]
    
        # uncertainties    
        lower_2sigma = tau_BCZ_fit50 - tau_BCZ_fit025
        lower_1sigma = tau_BCZ_fit50 - tau_BCZ_fit16
        upper_1sigma = tau_BCZ_fit84 - tau_BCZ_fit50
        upper_2sigma = tau_BCZ_fit975 - tau_BCZ_fit50
    
        # Store
        tau_BCZ = {'lower_2sigma':lower_2sigma,
                   'lower_1sigma':lower_1sigma,
                   'upper_1sigma':upper_1sigma,
                   'upper_2sigma':upper_2sigma,
                   'median':tau_BCZ_fit50}


    # Find HeII tau in the fits
    bins = results['hist']['HeII']['bins']
    bins *= 1e6
    pdf = results['hist']['HeII']['pdf']
    if np.any(np.isnan(pdf)):
        # Store
        tau_HeII = {'lower_2sigma':np.nan,
                    'lower_1sigma':np.nan,
                    'upper_1sigma':np.nan,
                    'upper_2sigma':np.nan,
                    'median':np.nan}
    else:        
        area = np.sum( pdf * np.diff(bins) )
        pdf = pdf/area
        tau_tmp = np.linspace(bins[0], bins[-1], 10000)
        pdf_tmp = hist_function(tau_tmp, bins, pdf)    
        delta = np.median( np.diff(tau_tmp) )
        prob = delta * pdf_tmp
        cumprob = np.cumsum(prob)
        ind = np.argmin(np.abs(cumprob-0.025)) # percentile 2.5
        tau_HeII_fit025 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.16)) # percentile 16
        tau_HeII_fit16 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.50)) # percentile 50
        tau_HeII_fit50 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.84)) # percentile 84
        tau_HeII_fit84 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.975)) # percentile 97.5
        tau_HeII_fit975 = tau_tmp[ind]
    
        # uncertainties    
        lower_2sigma = tau_HeII_fit50 - tau_HeII_fit025
        lower_1sigma = tau_HeII_fit50 - tau_HeII_fit16
        upper_1sigma = tau_HeII_fit84 - tau_HeII_fit50
        upper_2sigma = tau_HeII_fit975 - tau_HeII_fit50
    
        # Store
        tau_HeII = {'lower_2sigma':lower_2sigma,
                    'lower_1sigma':lower_1sigma,
                    'upper_1sigma':upper_1sigma,
                    'upper_2sigma':upper_2sigma,
                    'median':tau_HeII_fit50}
    
    # Find HeII tau in the fits
    bins = results['hist']['H']['bins']
    bins *= 1e6
    pdf = results['hist']['H']['pdf']
    if np.any(np.isnan(pdf)):
        # Store
        tau_H = {'lower_2sigma':np.nan,
                 'lower_1sigma':np.nan,
                 'upper_1sigma':np.nan,
                 'upper_2sigma':np.nan,
                 'median':np.nan}
    else:        
        area = np.sum( pdf * np.diff(bins) )
        pdf = pdf/area
        tau_tmp = np.linspace(bins[0], bins[-1], 10000)
        pdf_tmp = hist_function(tau_tmp, bins, pdf)    
        delta = np.median( np.diff(tau_tmp) )
        prob = delta * pdf_tmp
        cumprob = np.cumsum(prob)
        ind = np.argmin(np.abs(cumprob-0.025)) # percentile 2.5
        tau_H_fit025 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.16)) # percentile 16
        tau_H_fit16 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.50)) # percentile 50
        tau_H_fit50 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.84)) # percentile 84
        tau_H_fit84 = tau_tmp[ind]
        ind = np.argmin(np.abs(cumprob-0.975)) # percentile 97.5
        tau_H_fit975 = tau_tmp[ind]
    
        # uncertainties    
        lower_2sigma = tau_H_fit50 - tau_H_fit025
        lower_1sigma = tau_H_fit50 - tau_H_fit16
        upper_1sigma = tau_H_fit84 - tau_H_fit50
        upper_2sigma = tau_H_fit975 - tau_H_fit50
    
        # Store
        tau_H = {'lower_2sigma':lower_2sigma,
                 'lower_1sigma':lower_1sigma,
                 'upper_1sigma':upper_1sigma,
                 'upper_2sigma':upper_2sigma,
                 'median':tau_H_fit50}
    

    # save results
    uncertainties[profile] = {'tau_BCZ':tau_BCZ,
                              'tau_HeII':tau_HeII,
                              'tau_H':tau_H}
    
#    # print
#    print(profile, 'offset TAU BCZ =', offset, 'seconds')

# File where to save the results
savefile = open('tau_uncertainties.bin','wb')
pickle.dump(uncertainties, savefile)
savefile.close()
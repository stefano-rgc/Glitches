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
f = open('results.bin','rb')
fit_results = pickle.load(f)
f.close()

# Variale where to save the results summary results
hist_results = dict()

# plot histogram of the results if the grid
pdf_hist = PdfPages('test_tau_histograms.pdf')

# Plot style       
plt.style.use('stefano')

for profile,fits in fit_results.items():
    print(profile)
    
    fig_hist=plt.figure()
    gs=gridspec.GridSpec(2,2)
    
    ### Tau_BCZ
    
    # Get the bins. Note that all the realizations share the same bins
    bins = fits[0]['hist']['BCZ']['bins']
    tau_BCZ_results_grid_bin_pdf = [ fits[i]['hist']['BCZ']['pdf'] for i in fits if not any(np.isnan(fits[i]['hist']['BCZ']['pdf'])) ]

    if len(tau_BCZ_results_grid_bin_pdf) == 0:
        hist_BCZ = {'bins':bins,
                    'pdf':np.nan}
    else:
        tau_BCZ_results_grid_bin_pdf = np.sum(tau_BCZ_results_grid_bin_pdf, axis=0)
        # Find the area below the histogram
        heights_BCZ = tau_BCZ_results_grid_bin_pdf
        widths = np.diff(bins)
        area = np.sum(heights_BCZ*widths)
        # Normalize
        tau_BCZ_results_grid_bin_pdf = tau_BCZ_results_grid_bin_pdf/area
        
        # plot BCZ
        ax_hist=plt.subplot(gs[0])
        try:
            ax_hist.fill_between(bins*1e6,np.concatenate([[0],tau_BCZ_results_grid_bin_pdf]),color='dodgerblue',step='pre')   
        except ValueError:
            embed()
        ax_hist.set_xlabel('Tau BCZ')
        
        hist_BCZ = {'bins':bins,
                    'pdf':tau_BCZ_results_grid_bin_pdf}


    ### Tau_HeII

    # Get the bins. Note that all the realizations share the same bins
    bins = fits[0]['hist']['HeII']['bins']
    tau_HeII_results_grid_bin_pdf = [ fits[i]['hist']['HeII']['pdf'] for i in fits if not any(np.isnan(fits[i]['hist']['HeII']['pdf'])) ]
    
    if len(tau_HeII_results_grid_bin_pdf) == 0:
        hist_HeII = {'bins':bins,
                     'pdf':np.nan}
    else:    
        tau_HeII_results_grid_bin_pdf = np.sum(tau_HeII_results_grid_bin_pdf, axis=0)
        # Find the area below the histogram
        heights_HeII = tau_HeII_results_grid_bin_pdf
        widths = np.diff(bins)
        area = np.sum(heights_HeII*widths)
        # Normalize
        tau_HeII_results_grid_bin_pdf = tau_HeII_results_grid_bin_pdf/area
        
        # plot HeII
        ax_hist=plt.subplot(gs[1])
        ax_hist.fill_between(bins*1e6,np.concatenate([[0],tau_HeII_results_grid_bin_pdf]),color='dodgerblue',step='pre')   
        ax_hist.set_xlabel('Tau HeII')    
        
        hist_HeII = {'bins':bins,
                     'pdf':tau_HeII_results_grid_bin_pdf}    
   
    
    ### Tau_H
    
    # Get the bins. Note that all the realizations share the same bins
    bins = fits[0]['hist']['H']['bins']
    tau_H_results_grid_bin_pdf = [ fits[i]['hist']['H']['pdf'] for i in fits if not any(np.isnan(fits[i]['hist']['H']['pdf']))]

    if len(tau_H_results_grid_bin_pdf) == 0:
        hist_H = {'bins':bins,
                  'pdf':np.nan}
    else:    
        tau_H_results_grid_bin_pdf = np.sum(tau_H_results_grid_bin_pdf, axis=0)
        # Find the area below the histogram
        heights_H = tau_H_results_grid_bin_pdf
        widths = np.diff(bins)
        area = np.sum(heights_H*widths)
        # Normalize
        tau_H_results_grid_bin_pdf = tau_H_results_grid_bin_pdf/area
        
        # plot HeII
        ax_hist=plt.subplot(gs[2])
        ax_hist.fill_between(bins*1e6,np.concatenate([[0],tau_H_results_grid_bin_pdf]),color='dodgerblue',step='pre')   
        ax_hist.set_xlabel('Tau HeII')    
        
        hist_H = {'bins':bins,
                  'pdf':tau_H_results_grid_bin_pdf}   


    # Title
    fig_hist.suptitle('Age after ZAMS: {:.3f} Gyr, {}'.format(fits[0]['age'], profile))
    fig_hist.subplots_adjust(hspace=0.4)        
    pdf_hist.savefig(fig_hist)
    plt.close(fig_hist)
    

    hist_results_profile = dict()
    # summarize the results
    filter_in_keys = ['l', 'age', 'secdiff', 'nu_secdiff_no_statistical_errors', 
                      'secdiff_no_statistical_errors']
    hist_results_profile = { key:fits[0][key] for key in filter_in_keys }
    hist_results_profile['hist'] = {'BCZ':hist_BCZ,
                                    'HeII':hist_HeII,
                                    'H':hist_H}
    
    hist_results[profile] = hist_results_profile

pdf_hist.close()

# File where to save the results
savefile = open('tau_histograms.bin','wb')
pickle.dump(hist_results, savefile)
savefile.close()

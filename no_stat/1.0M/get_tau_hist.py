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
f = open('fit_results.bin','rb')
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

    # ==========================================================================
    ### Plot histograms from the grid
    # ==========================================================================
    
    fig_hist=plt.figure()
    gs=gridspec.GridSpec(2,2)
    
    # Get the chi2 squared value of all the grid    
    chi2_grid = [ _['residuals']['chi2'] for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ] 
    chi2_grid = np.array(chi2_grid)
    
    ### Tau_BCZ
    tau_BCZ_results_grid = [ _['results']['tau_BCZ'] for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ]
    tau_BCZ_results_grid = np.array(tau_BCZ_results_grid)
    # Exclude where the fit did not converge
    ind_converge = tau_BCZ_results_grid != None
    # Generate the bins based on the tau results (not the tau initial guess or grid)
    hist, bins = np.histogram(tau_BCZ_results_grid[ind_converge], bins='auto')
    # Search which bin hosts the elements in tau_results_grid
    bin_host = np.searchsorted(bins, tau_BCZ_results_grid[ind_converge])        
    # If there are 3 bins: in_host=0 is before the first bin and in_host=4 is after the last bin
    # For each bin (If there are 3 bins, say, 3 bars. The bins are 1,2,3)
    tau_BCZ_results_grid_bin_pdf = list()
    for i_bin in np.linspace(1, len(bins)-1, num=len(bins)-1):
        # Select from bin_host the ones equal i_bin
        tau_BCZ_results_grid_bin_pdf.append( np.sum( np.exp(-chi2_grid[ind_converge][bin_host==i_bin]**2) ) )
    # Gaussian probability
    tau_BCZ_results_grid_bin_pdf = np.array(tau_BCZ_results_grid_bin_pdf)
    # Find the area below the histogram
    heights_BCZ = tau_BCZ_results_grid_bin_pdf
    widths = np.diff(bins)
    area = np.sum(heights_BCZ*widths)
    # Normalize
    tau_BCZ_results_grid_bin_pdf = tau_BCZ_results_grid_bin_pdf/area
    
    # plot BCZ
    ax_hist=plt.subplot(gs[0])
    ax_hist.fill_between(bins*1e6,np.concatenate([[0],tau_BCZ_results_grid_bin_pdf]),color='dodgerblue',step='pre')   
    ax_hist.set_xlabel('Tau BCZ')
    
    hist_BCZ = {'bins':bins,
                'pdf':tau_BCZ_results_grid_bin_pdf}
    
    ### Tau_HeII
    tau_HeII_results_grid = [ _['results']['tau_HeII'] for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ]
    tau_HeII_results_grid = np.array(tau_HeII_results_grid)
    # Exclude where the fit did not converge
    ind_converge = tau_HeII_results_grid != None
    # Generate the bins based on the tau results (not the tau initial guess or grid)
    hist, bins = np.histogram(tau_HeII_results_grid[ind_converge], bins='auto')
    # Search which bin hosts the elements in tau_results_grid
    bin_host = np.searchsorted(bins, tau_HeII_results_grid[ind_converge])        
    # If there are 3 bins: in_host=0 is before the first bin and in_host=4 is after the last bin
    # For each bin (If there are 3 bins, say, 3 bars. The bins are 1,2,3)
    tau_HeII_results_grid_bin_pdf = list()
    for i_bin in np.linspace(1, len(bins)-1, num=len(bins)-1):
        # Select from bin_host the ones equal i_bin
        tau_HeII_results_grid_bin_pdf.append( np.sum( np.exp(-chi2_grid[ind_converge][bin_host==i_bin]**2) ) )
    # Gaussian probability
    tau_HeII_results_grid_bin_pdf = np.array(tau_HeII_results_grid_bin_pdf)
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
    tau_H_results_grid = [ _['results']['tau_H'] for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ]
    tau_H_results_grid = np.array(tau_H_results_grid)
    # Exclude where the fit did not converge
    ind_converge = tau_H_results_grid != None
    # Generate the bins based on the tau results (not the tau initial guess or grid)
    hist, bins = np.histogram(tau_H_results_grid[ind_converge], bins='auto')
    # Search which bin hosts the elements in tau_results_grid
    bin_host = np.searchsorted(bins, tau_H_results_grid[ind_converge])        
    # If there are 3 bins: in_host=0 is before the first bin and in_host=4 is after the last bin
    # For each bin (If there are 3 bins, say, 3 bars. The bins are 1,2,3)
    tau_H_results_grid_bin_pdf = list()
    for i_bin in np.linspace(1, len(bins)-1, num=len(bins)-1):
        # Select from bin_host the ones equal i_bin
        tau_H_results_grid_bin_pdf.append( np.sum( np.exp(-chi2_grid[ind_converge][bin_host==i_bin]**2) ) )
    # Gaussian probability
    tau_H_results_grid_bin_pdf = np.array(tau_H_results_grid_bin_pdf)
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
    fig_hist.suptitle('Age after ZAMS: {:.3f} Gyr, {}'.format(fits['age'], profile))
    fig_hist.subplots_adjust(hspace=0.4)        
    pdf_hist.savefig(fig_hist)
    plt.close(fig_hist)

    hist_results_profile = dict()
    # summarize the results
    filter_in_keys = ['l', 'age', 'secdiff', 'nu_secdiff', 
                      'secdiff_filtered', 'nu_secdiff_filtered',
                      'fit_smooth_HeII_BCZ_H_grid']
    hist_results_profile = { key:fits[key] for key in filter_in_keys }
    hist_results_profile['hist'] = {'BCZ':hist_BCZ,
                                    'HeII':hist_HeII,
                                    'H':hist_H}
    
    hist_results[profile] = hist_results_profile

pdf_hist.close()

# File where to save the results
savefile = open('tau_histograms.bin','wb')
pickle.dump(hist_results, savefile)
savefile.close()

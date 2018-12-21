import matplotlib as mpl
mpl.use('agg')
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
#import fgong2ascii
from matplotlib.offsetbox import AnchoredText # For positioning annotations
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from IPython import embed
from os import remove
import pickle
from scipy.interpolate import interp1d
import pandas as pd
import peakutils
from joblib import Parallel, delayed
import multiprocessing
import time

# =============================================================================
# ==============================================================================

def plot_secdiff(fig, # Figure 
                 nu_secdiff, # x and y for ploting (Dictionaries: l-value as keys)
                 secdiff, 
                 profile_and_number, # profile1
                 age,
                 only_secdiff, # True/False
                 fit, # True/False
                 individual_components1=False, # True/False
                 individual_components2=False, # True/False
                 individual_components3=False, # True/False
                 individual_components4=False, # True/False
                 individual_components_all=False, # True/False
                 f='', # fit function
                 params='', # parameters of the fit function
                 residuals='', # (Dictionaries: l-value as keys)
                 ax_fit=None,
                 ax_residual=None):
    
    colors = {0:'k', 1:'r', 2:'dodgerblue', 3:'lime'}
    markers = {0:'o', 1:'^', 2:'s', 3:'D'}
   
    # Get the l-values 
    ell = [ l for l in secdiff.keys() ]

    if only_secdiff:
        # One panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0])
    
        # Plot the second differences
        for l in ell:
            ax.plot(nu_secdiff[l], secdiff[l],
                    color=colors[l], marker=markers[l],
                    linestyle='none', markersize=3,
                    label=r'$\ell={}$'.format(l),
                    markeredgecolor='None')

        ax.set(ylabel=r'Second Diffecences $\delta_2\nu$ ( $\mu$Hz )',
               title='Age after ZAMS: {:.3f} Gyr'.format(age),
               xlabel=r'Frequency ( $\mu$Hz )')

        
        return ax
    
    if fit:
        # Two panels
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
        # Axes fit
        if ax_fit is None: ax_fit = plt.subplot(gs[0])
        # Axes residuals
        if ax_residual is None: ax_residual = plt.subplot(gs[1], sharex=ax_fit)

        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]
        # Plot the second differencse
        for l in ell:
            ax_fit.plot(nu_secdiff[l], secdiff[l],
                        color=colors[l], marker=markers[l],
                        linestyle='none',markersize=3,
                        markeredgecolor='None',
                        label=r'$\ell={}$'.format(l))

        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])
                    
        # Plot the fit 
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
  
        ax_fit.plot(nu_fit, f(nu_fit, *params), \
                 linestyle='solid', color='orange', marker='None', label='fit')
    
        ax_fit.set(ylabel=r'Second Diffecences $\delta_2\nu$ ( $\mu$Hz )',
                          title='Age after ZAMS: {:.3f} Gyr'.format(age))
    
        # Text   
        if f == fit_smooth:
            text = text_params('smooth',params)             
            ax_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                            horizontalalignment='left', verticalalignment='center',
                            bbox={'facecolor':'white', 'edgecolor':'black'})

        elif f == fit_smooth_HeII:
            text = text_params('smooth_HeII',params)             
            ax_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                            horizontalalignment='left', verticalalignment='center',
                            bbox={'facecolor':'white', 'edgecolor':'black'})

        elif f == fit_smooth_HeII_BCZ:
            text = text_params('smooth_HeII_BCZ',params)             
            ax_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                            horizontalalignment='left', verticalalignment='center',
                            bbox={'facecolor':'white', 'edgecolor':'black'})

        elif f == fit_smooth_HeII_BCZ_H:
            text = text_params('smooth_HeII_BCZ_H',params)             
            ax_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                            horizontalalignment='left', verticalalignment='center',
                            bbox={'facecolor':'white', 'edgecolor':'black'})

        else:
            raise ValueError('Fit function not recognized.')
        
        # Residuos
        # Plot the data minus the model (residuals)
        for l in ell:
            ax_residual.plot(nu_secdiff[l], residuals[l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None')
    
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
      
        ax_residual.set(ylabel=r'Residual',
                               xlabel=r'Frequency ( $\mu$Hz )')
      
        ax_residual.annotate(r'$\chi^2_\nu($dof$={})={:.5}$'.format(residuals['dof'],residuals['reduced_chi2']),
                             xy=(1-0.22,0.07), xycoords='figure fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox={'edgecolor':'black', 'facecolor':'white'})
        
        # Legend
        ax_fit.legend(loc=(1.035,0.9), ncol=2, handletextpad=0.1, 
                      fontsize=9, handlelength=2)   
        
        # ID for-loop
        ax_fit.annotate('{}'.format(profile_and_number), xy=(1-0.22,0.12), 
                        xycoords='figure fraction', horizontalalignment='left', 
                        verticalalignment='center',
                        bbox={'edgecolor':'black', 'facecolor':'white'})
    
        # No x-axis ticks
        plt.setp(ax_fit.get_xticklabels(), visible=False)
    
        # remove last and first tick label for the second subplot
        yticks_fit = ax_fit.yaxis.get_major_ticks()
        yticks_fit[-1].label1.set_visible(False)
        yticks_fit[0].label1.set_visible(False)
    
        # Creates free space outside the plot for the parameters display
        fig.subplots_adjust(right = 0.75, hspace=0)    
        
        return ax_fit, ax_residual
    
    if individual_components1:
        # Four panels
        gs = gridspec.GridSpec(3, 1) 
        # Axes smooth fit to second differences
        ax_smooth_fit = plt.subplot(gs[0])
        # Axes oscillating component fit to the residuals above
        ax_oscillating_fit = plt.subplot(gs[1], sharex=ax_smooth_fit)
        # Axes residuals
        ax_residual = plt.subplot(gs[2], sharex=ax_smooth_fit)

        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]
        
        # Plot the second differencse
        for l in ell:
            ax_smooth_fit.plot(nu_secdiff[l], secdiff[l],
                               color=colors[l], marker=markers[l],
                               linestyle='none',markersize=3,
                               markeredgecolor='None')

        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])
                    
        # Plot the smooth fit 
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
  
        ax_smooth_fit.plot(nu_fit, fit_smooth(nu_fit, *params[0:3]), \
                           linestyle='solid', color='orange', marker='None', label='Smooth component')
    
        ax_smooth_fit.set_ylabel('Second Diffecences\n'+r'( $\mu$Hz )', fontsize=7)
        ax_smooth_fit.set_title('Age after ZAMS: {:.3f} Gyr'.format(age))
        # Text   
        text = text_params('smooth_HeII_BCZ_H',params)             
        ax_smooth_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                        horizontalalignment='left', verticalalignment='center',
                        bbox={'facecolor':'white', 'edgecolor':'black'})
        # Legend
        ax_smooth_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=2)   

        # Plot the residuals after the smooth fit
        for l in ell:
            ax_oscillating_fit.plot(nu_secdiff[l], residuals['smooth'][l],
                                    color='k', marker=markers[l],
                                    linestyle='none', markersize=3,
                                    markeredgecolor='None')
        # Horizontal 0-line
        ax_oscillating_fit.axhline(0,linestyle='dotted', color='k')


        # Plot the oscilllating fit 
        ax_oscillating_fit.plot(nu_fit, fit_HeII(nu_fit, *params[3:7]), \
                                linestyle='solid', color='red', marker='None', label='HeII fit')
        ax_oscillating_fit.plot(nu_fit, fit_BCZ(nu_fit, *params[7:10]), \
                                linestyle='solid', color='dodgerblue', marker='None', label='BCZ fit')
        ax_oscillating_fit.plot(nu_fit, fit_H(nu_fit, *params[10:14]), \
                                linestyle='solid', color='lime', marker='None', label='H&HeI fit')
        ax_oscillating_fit.plot(nu_fit, fit_HeII_BCZ_H(nu_fit, *params[3:]), \
                                linestyle='solid', color='orange', marker='None', label='HeII, BCZ and H&HeI fit')

        ax_oscillating_fit.set_ylabel('Residual after smooth\n'+'component fit', fontsize=7)
        ax_oscillating_fit.legend(loc='best', ncol=4, handletextpad=0.1, fontsize=7, handlelength=1)
     
        # Residuos
        # Plot the data minus the model (residuals)
        for l in ell:
            ax_residual.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ_H'][l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None',
                             label=r'$\ell={}$'.format(l))
    
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
      
        ax_residual.set_ylabel('Residual after\n total fit', fontsize=7)
        ax_residual.set_xlabel(r'Frequency ( $\mu$Hz )')
      
        ax_residual.annotate(r'$\chi^2_\nu($dof$={})={:.5}$'.format(residuals['smooth_HeII_BCZ_H']['dof'],residuals['smooth_HeII_BCZ_H']['reduced_chi2']),
                             xy=(1-0.22,0.07), xycoords='figure fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox={'edgecolor':'black', 'facecolor':'white'})
        
        # Legend
        ax_residual.legend(loc=(1.035,2.8), ncol=2, handletextpad=0.1, fontsize=9, handlelength=2)   
        
        # ID for-loop
        ax_smooth_fit.annotate('{}'.format(profile_and_number), xy=(1-0.22,0.12), 
                               xycoords='figure fraction', horizontalalignment='left', 
                               verticalalignment='center',
                               bbox={'edgecolor':'black', 'facecolor':'white'})
    
        # No x-axis ticks
        plt.setp(ax_smooth_fit.get_xticklabels(), visible=False)
        plt.setp(ax_oscillating_fit.get_xticklabels(), visible=False)
        # y-axis ticks
        plt.setp(ax_smooth_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_oscillating_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_residual.get_yticklabels(), fontsize=7)

        # remove last and first tick label for the second subplot
        yticks_smooth_fit = ax_smooth_fit.yaxis.get_major_ticks()
        yticks_smooth_fit[-1].label1.set_visible(False)
        yticks_smooth_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_oscillating_fit = ax_oscillating_fit.yaxis.get_major_ticks()
        yticks_oscillating_fit[-1].label1.set_visible(False)
        yticks_oscillating_fit[0].label1.set_visible(False)
        
        # remove last and first tick label for the second subplot
        yticks_residual = ax_residual.yaxis.get_major_ticks()
        yticks_residual[-1].label1.set_visible(False)
        yticks_residual[0].label1.set_visible(False)
  
        # Creates free space outside the plot for the parameters display
        fig.subplots_adjust(right = 0.75, hspace=0)    
        
        return ax_smooth_fit, ax_oscillating_fit, ax_residual
    
    if individual_components2:
        # Four panels
        gs = gridspec.GridSpec(3, 1) 
        # Axes HeII fit to the residuals after the smooth fit
        ax_smooth_residual_HeII_fit = plt.subplot(gs[0])
        # Axes BCZ fit to the residuals above
        ax_smooth_HeII_residual_BCZ_H_fit = plt.subplot(gs[1], sharex=ax_smooth_residual_HeII_fit)
        # Axes residuals
        ax_residual = plt.subplot(gs[2], sharex=ax_smooth_residual_HeII_fit)
                    
        # Plot the residuals after the smooth fit
        for l in ell:
            ax_smooth_residual_HeII_fit.plot(nu_secdiff[l], residuals['smooth'][l],
                                             color='k', marker=markers[l],
                                             linestyle='none', markersize=3,
                                             markeredgecolor='None')

        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]

        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])
                    
        # Plot the HeII fit on the residuals from the smooth fit
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
  
        ax_smooth_residual_HeII_fit.plot(nu_fit, fit_HeII(nu_fit, *params[3:7]), \
                                         linestyle='solid', color='orange', marker='None', label='HeII fit')
    

        # Horizontal 0-line
        ax_smooth_residual_HeII_fit.axhline(0,linestyle='dotted', color='k')

        ax_smooth_residual_HeII_fit.set_ylabel('Residual after smooth\ncomponent fit', fontsize=7)
        ax_smooth_residual_HeII_fit.set_title('Age after ZAMS: {:.3f} Gyr'.format(age))
        # Text   
        text = text_params('smooth_HeII_BCZ_H',params)             
        ax_smooth_residual_HeII_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                                             horizontalalignment='left', verticalalignment='center',
                                             bbox={'facecolor':'white', 'edgecolor':'black'})
        # Legend
        ax_smooth_residual_HeII_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Plot the residuals after the smooth and HeII fit
        for l in ell:
            ax_smooth_HeII_residual_BCZ_H_fit.plot(nu_secdiff[l], residuals['smooth_HeII'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')
        # Horizontal 0-line
        ax_smooth_HeII_residual_BCZ_H_fit.axhline(0,linestyle='dotted', color='k')


        # Plot the oscilllating fit 
        ax_smooth_HeII_residual_BCZ_H_fit.plot(nu_fit, fit_BCZ_H(nu_fit, *params[7:]), \
                                               linestyle='solid', color='orange', marker='None', label='BCZ and H&HeI fit')
        ax_smooth_HeII_residual_BCZ_H_fit.plot(nu_fit, fit_BCZ(nu_fit, *params[7:10]), \
                                               linestyle='solid', color='dodgerblue', marker='None', label='BCZ fit')
        ax_smooth_HeII_residual_BCZ_H_fit.plot(nu_fit, fit_H(nu_fit, *params[10:14]), \
                                               linestyle='solid', color='red', marker='None', label='H&HeI fit')

        ax_smooth_HeII_residual_BCZ_H_fit.set_ylabel('Residual after smooth\n component and HeII fit', fontsize=7)
        ax_smooth_HeII_residual_BCZ_H_fit.legend(loc='best', ncol=3, handletextpad=0.1, fontsize=7, handlelength=1)

     
        # Residuos
        # Plot the data minus the model (residuals)
        for l in ell:
            ax_residual.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ_H'][l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None',
                             label=r'$\ell={}$'.format(l))
    
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
      
        ax_residual.set_ylabel('Residual after\n total fit', fontsize=7)
        ax_residual.set_xlabel(r'Frequency ( $\mu$Hz )')
      
        ax_residual.annotate(r'$\chi^2_\nu($dof$={})={:.5}$'.format(residuals['smooth_HeII_BCZ_H']['dof'],residuals['smooth_HeII_BCZ_H']['reduced_chi2']),
                             xy=(1-0.22,0.07), xycoords='figure fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox={'edgecolor':'black', 'facecolor':'white'})
        
        # Legend
        ax_residual.legend(loc=(1.035,2.8), ncol=2, handletextpad=0.1, fontsize=9, handlelength=2)   
#        ax_smooth_residual_HeII_fit.legend(loc=(1.035,0.9), ncol=2, handletextpad=0.1, 
#                                           fontsize=9, handlelength=2)   
        
        # ID for-loop
        ax_smooth_residual_HeII_fit.annotate('{}'.format(profile_and_number), xy=(1-0.22,0.12), 
                                             xycoords='figure fraction', horizontalalignment='left', 
                                             verticalalignment='center',
                                             bbox={'edgecolor':'black', 'facecolor':'white'})
    
        # No x-axis ticks
        plt.setp(ax_smooth_residual_HeII_fit.get_xticklabels(), visible=False)
        plt.setp(ax_smooth_HeII_residual_BCZ_H_fit.get_xticklabels(), visible=False)

        # y-axis ticks
        plt.setp(ax_smooth_residual_HeII_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_smooth_HeII_residual_BCZ_H_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_residual.get_yticklabels(), fontsize=7)

        # remove last and first tick label for the second subplot
        yticks_smooth_residual_HeII_fit = ax_smooth_residual_HeII_fit.yaxis.get_major_ticks()
        yticks_smooth_residual_HeII_fit[-1].label1.set_visible(False)
        yticks_smooth_residual_HeII_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_smooth_HeII_residual_BCZ_H_fit = ax_smooth_HeII_residual_BCZ_H_fit.yaxis.get_major_ticks()
        yticks_smooth_HeII_residual_BCZ_H_fit[-1].label1.set_visible(False)
        yticks_smooth_HeII_residual_BCZ_H_fit[0].label1.set_visible(False)
        
        # remove last and first tick label for the second subplot
        yticks_residual = ax_residual.yaxis.get_major_ticks()
        yticks_residual[-1].label1.set_visible(False)
        yticks_residual[0].label1.set_visible(False)
  
        # Creates free space outside the plot for the parameters display
        fig.subplots_adjust(right = 0.75, hspace=0)    
        
        return ax_smooth_residual_HeII_fit, ax_smooth_HeII_residual_BCZ_H_fit, ax_residual
    
    if individual_components3:
        # Four panels
        gs = gridspec.GridSpec(3, 1) 
        # Axes BCZ fit to the residuals after the smooth fit and the HeII fit
        ax_smooth_HeII_residual_BCZ_fit = plt.subplot(gs[0])
        # Axes H&HeI fit to the residuals after the smooth fit, the HeII fit and the BCZ fit
        ax_smooth_HeII_BCZ_residual_H_fit = plt.subplot(gs[1], sharex=ax_smooth_HeII_residual_BCZ_fit)
        # Axes residuals
        ax_residual = plt.subplot(gs[2], sharex=ax_smooth_HeII_residual_BCZ_fit)
                    
        # Plot the residuals after the smooth fit
        for l in ell:
            ax_smooth_HeII_residual_BCZ_fit.plot(nu_secdiff[l], residuals['smooth_HeII'][l],
                                                 color='k', marker=markers[l],
                                                 linestyle='none', markersize=3,
                                                 markeredgecolor='None')

        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]

        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])
                    
        # Plot the BCZ fit on the residuals after the smooth fit and the HeII fit
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
  
        ax_smooth_HeII_residual_BCZ_fit.plot(nu_fit, fit_BCZ(nu_fit, *params[7:10]), \
                                             linestyle='solid', color='orange', marker='None', label='BCZ fit')
    

        # Horizontal 0-line
        ax_smooth_HeII_residual_BCZ_fit.axhline(0,linestyle='dotted', color='k')

        ax_smooth_HeII_residual_BCZ_fit.set_ylabel('Residual after smooth\ncomponent and HeII fit', fontsize=7)
        ax_smooth_HeII_residual_BCZ_fit.set_title('Age after ZAMS: {:.3f} Gyr'.format(age))
        # Text   
        text = text_params('smooth_HeII_BCZ_H',params)             
        ax_smooth_HeII_residual_BCZ_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                                                 horizontalalignment='left', verticalalignment='center',
                                                 bbox={'facecolor':'white', 'edgecolor':'black'})
        # Legend
        ax_smooth_HeII_residual_BCZ_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Plot the residuals after the smooth, HeII and BCZ fit
        for l in ell:
            ax_smooth_HeII_BCZ_residual_H_fit.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')
        # Horizontal 0-line
        ax_smooth_HeII_BCZ_residual_H_fit.axhline(0,linestyle='dotted', color='k')


        # Plot the H&HeI fit on the residuals after the smooth fit, the HeII fit and the BCZ fit
        ax_smooth_HeII_BCZ_residual_H_fit.plot(nu_fit, fit_H(nu_fit, *params[-4:]), \
                                               linestyle='solid', color='orange', marker='None', label='H&HeI fit')

        ax_smooth_HeII_BCZ_residual_H_fit.set_ylabel('Residual after smooth\n component, HeII and BCZ fit', fontsize=7)
        ax_smooth_HeII_BCZ_residual_H_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Residuos
        # Plot the data minus the model (residuals)
        for l in ell:
            ax_residual.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ_H'][l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None',
                             label=r'$\ell={}$'.format(l))
    
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
      
        ax_residual.set_ylabel('Residual after\n total fit', fontsize=7)
        ax_residual.set_xlabel(r'Frequency ( $\mu$Hz )')
      
        ax_residual.annotate(r'$\chi^2_\nu($dof$={})={:.5}$'.format(residuals['smooth_HeII_BCZ_H']['dof'],residuals['smooth_HeII_BCZ_H']['reduced_chi2']),
                             xy=(1-0.22,0.07), xycoords='figure fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox={'edgecolor':'black', 'facecolor':'white'})
        
        # Legend
        ax_residual.legend(loc=(1.035,2.8), ncol=2, handletextpad=0.1, fontsize=9, handlelength=2)   
        
        # ID for-loop
        ax_smooth_HeII_residual_BCZ_fit.annotate('{}'.format(profile_and_number), xy=(1-0.22,0.12), 
                                                 xycoords='figure fraction', horizontalalignment='left', 
                                                 verticalalignment='center',
                                                 bbox={'edgecolor':'black', 'facecolor':'white'})
    
        # No x-axis ticks
        plt.setp(ax_smooth_HeII_residual_BCZ_fit.get_xticklabels(), visible=False)
        plt.setp(ax_smooth_HeII_BCZ_residual_H_fit.get_xticklabels(), visible=False)

        # y-axis ticks
        plt.setp(ax_smooth_HeII_residual_BCZ_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_smooth_HeII_BCZ_residual_H_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_residual.get_yticklabels(), fontsize=7)

        # remove last and first tick label for the second subplot
        yticks_smooth_residual_HeII_fit = ax_smooth_HeII_residual_BCZ_fit.yaxis.get_major_ticks()
        yticks_smooth_residual_HeII_fit[-1].label1.set_visible(False)
        yticks_smooth_residual_HeII_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_smooth_HeII_residual_BCZ_H_fit = ax_smooth_HeII_BCZ_residual_H_fit.yaxis.get_major_ticks()
        yticks_smooth_HeII_residual_BCZ_H_fit[-1].label1.set_visible(False)
        yticks_smooth_HeII_residual_BCZ_H_fit[0].label1.set_visible(False)
        
        # remove last and first tick label for the second subplot
        yticks_residual = ax_residual.yaxis.get_major_ticks()
        yticks_residual[-1].label1.set_visible(False)
        yticks_residual[0].label1.set_visible(False)
  
        # Creates free space outside the plot for the parameters display
        fig.subplots_adjust(right = 0.75, hspace=0)    
        
        return ax_smooth_HeII_residual_BCZ_fit, ax_smooth_HeII_BCZ_residual_H_fit, ax_residual
    
    if individual_components4:
        # Four panels
        gs = gridspec.GridSpec(3, 1) 
        # Axes BCZ fit to the residuals after the smooth fit and the HeII fit
        ax_smooth_HeII_residual_H_fit = plt.subplot(gs[0])
        # Axes H&HeI fit to the residuals after the smooth fit, the HeII fit and the BCZ fit
        ax_smooth_HeII_H_residual_BCZ_fit = plt.subplot(gs[1], sharex=ax_smooth_HeII_residual_H_fit)
        # Axes residuals
        ax_residual = plt.subplot(gs[2], sharex=ax_smooth_HeII_residual_H_fit)
                    
        # Plot the residuals after the smooth and HeII fit
        for l in ell:
            ax_smooth_HeII_residual_H_fit.plot(nu_secdiff[l], residuals['smooth_HeII'][l],
                                               color='k', marker=markers[l],
                                               linestyle='none', markersize=3,
                                               markeredgecolor='None')

        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]

        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])
                    
        # Plot the BCZ fit on the residuals after the smooth fit and the HeII fit
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
  
        ax_smooth_HeII_residual_H_fit.plot(nu_fit, fit_H(nu_fit, *params[-4:]), \
                                           linestyle='solid', color='orange', marker='None', label='H&HeI fit')
    

        # Horizontal 0-line
        ax_smooth_HeII_residual_H_fit.axhline(0,linestyle='dotted', color='k')

        ax_smooth_HeII_residual_H_fit.set_ylabel('Residual after smooth\ncomponent and HeII fit', fontsize=7)
        ax_smooth_HeII_residual_H_fit.set_title('Age after ZAMS: {:.3f} Gyr'.format(age))
        # Text   
        text = text_params('smooth_HeII_BCZ_H',params)             
        ax_smooth_HeII_residual_H_fit.annotate(text, xy=(1-0.22,0.5), xycoords='figure fraction',
                                                 horizontalalignment='left', verticalalignment='center',
                                                 bbox={'facecolor':'white', 'edgecolor':'black'})
        # Legend
        ax_smooth_HeII_residual_H_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Plot the residuals after the smooth, HeII and H&HeI fit
        for l in ell:
            ax_smooth_HeII_H_residual_BCZ_fit.plot(nu_secdiff[l], residuals['smooth_HeII_H'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')
        # Horizontal 0-line
        ax_smooth_HeII_H_residual_BCZ_fit.axhline(0,linestyle='dotted', color='k')


        # Plot the BCZ fit on the residuals after the smooth fit, the HeII fit and the H fit
        ax_smooth_HeII_H_residual_BCZ_fit.plot(nu_fit, fit_BCZ(nu_fit, *params[7:10]), \
                                               linestyle='solid', color='orange', marker='None', label='BCZ fit')

        ax_smooth_HeII_H_residual_BCZ_fit.set_ylabel('Residual after\nsmooth component,\nHeII and H&HeI fit', fontsize=7)
        ax_smooth_HeII_H_residual_BCZ_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Residuos
        # Plot the data minus the model (residuals)
        for l in ell:
            ax_residual.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ_H'][l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None',
                             label=r'$\ell={}$'.format(l))
    
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
      
        ax_residual.set_ylabel('Residual after\n total fit', fontsize=7)
        ax_residual.set_xlabel(r'Frequency ( $\mu$Hz )')
      
        ax_residual.annotate(r'$\chi^2_\nu($dof$={})={:.5}$'.format(residuals['smooth_HeII_BCZ_H']['dof'],residuals['smooth_HeII_BCZ_H']['reduced_chi2']),
                             xy=(1-0.22,0.07), xycoords='figure fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox={'edgecolor':'black', 'facecolor':'white'})
        
        # Legend
        ax_residual.legend(loc=(1.035,2.8), ncol=2, handletextpad=0.1, fontsize=9, handlelength=2)   
        
        # ID for-loop
        ax_smooth_HeII_residual_H_fit.annotate('{}'.format(profile_and_number), xy=(1-0.22,0.12), 
                                               xycoords='figure fraction', horizontalalignment='left', 
                                               verticalalignment='center',
                                               bbox={'edgecolor':'black', 'facecolor':'white'})
    
        # No x-axis ticks
        plt.setp(ax_smooth_HeII_residual_H_fit.get_xticklabels(), visible=False)
        plt.setp(ax_smooth_HeII_H_residual_BCZ_fit.get_xticklabels(), visible=False)

        # y-axis ticks
        plt.setp(ax_smooth_HeII_residual_H_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_smooth_HeII_H_residual_BCZ_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_residual.get_yticklabels(), fontsize=7)

        # remove last and first tick label for the second subplot
        yticks_smooth_residual_HeII_fit = ax_smooth_HeII_residual_H_fit.yaxis.get_major_ticks()
        yticks_smooth_residual_HeII_fit[-1].label1.set_visible(False)
        yticks_smooth_residual_HeII_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_smooth_HeII_residual_BCZ_H_fit = ax_smooth_HeII_H_residual_BCZ_fit.yaxis.get_major_ticks()
        yticks_smooth_HeII_residual_BCZ_H_fit[-1].label1.set_visible(False)
        yticks_smooth_HeII_residual_BCZ_H_fit[0].label1.set_visible(False)
        
        # remove last and first tick label for the second subplot
        yticks_residual = ax_residual.yaxis.get_major_ticks()
        yticks_residual[-1].label1.set_visible(False)
        yticks_residual[0].label1.set_visible(False)
  
        # Creates free space outside the plot for the parameters display
        fig.subplots_adjust(right = 0.75, hspace=0)    
        
        return ax_smooth_HeII_residual_H_fit, ax_smooth_HeII_H_residual_BCZ_fit, ax_residual        

    if individual_components_all:
        # Five panels
        gs = gridspec.GridSpec(6, 1) 
        # Axes: Second differences
        ax_secdiff = plt.subplot(gs[0])
        # Axes: Smooth component fit to the residuals after the HeII, BCZ and H&HeI fit
        ax_HeII_BCZ_H_residual_smooth_fit = plt.subplot(gs[1], sharex=ax_secdiff)
        # Axes: HeII fit to the residuals after the smooth fit, the BCZ fit and the H&HeI fit
        ax_smooth_BCZ_H_residual_HeII_fit = plt.subplot(gs[2], sharex=ax_secdiff)
        # Axes: BCZ fit to the residuals after the smooth fit, the smooth fit, the HeII fit and the H&HeI fit
        ax_smooth_HeII_H_residual_BCZ_fit = plt.subplot(gs[3], sharex=ax_secdiff)
        # Axes: H&HeI fit to the residuals after the smooth fit, the BCZ fit and the H&HeI fit
        ax_smooth_HeII_BCZ_residual_H_fit = plt.subplot(gs[4], sharex=ax_secdiff)
        # Axes residuals
        ax_residual = plt.subplot(gs[5], sharex=ax_secdiff)
 
        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]
        
        # Plot the second differencse
        for l in ell:
            ax_secdiff.plot(nu_secdiff[l], secdiff[l],
                            color=colors[l], marker=markers[l],
                            linestyle='none',markersize=3,
                            markeredgecolor='None')        
        
        ax_secdiff.set_ylabel('Second\ndifferences', fontsize=5)
        ax_secdiff.set_title('Age after ZAMS: {:.3f} Gyr'.format(age)) 
        
        # Plot the residuals after the smooth and HeII fit
        for l in ell:
            ax_HeII_BCZ_H_residual_smooth_fit.plot(nu_secdiff[l], residuals['HeII_BCZ_H'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')

        # Get the l-values 
        ell = [ l for l in secdiff.keys() ]

        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])
                    
        # Plot the smooth fit on the residuals after the HeII fit, BCZ fit and the H&HeI fit
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
  
        ax_HeII_BCZ_H_residual_smooth_fit.plot(nu_fit, fit_smooth(nu_fit, *params[:3]), \
                                               linestyle='solid', color='orange', marker='None', label='Smooth component fit')
    
        ax_HeII_BCZ_H_residual_smooth_fit.set_ylabel('Residual after\nHeII BCZ and\nH&HeI fit', fontsize=5)
#        ax_HeII_BCZ_H_residual_smooth_fit.set_title('Age after ZAMS: {:.3f} Gyr'.format(age))
        # Text   
        text = text_params('smooth_HeII_BCZ_H',params)             
        ax_HeII_BCZ_H_residual_smooth_fit.annotate(text, xy=(1-0.235,0.5), xycoords='figure fraction',
                                                   horizontalalignment='left', verticalalignment='center',
                                                   bbox={'facecolor':'white', 'edgecolor':'black'})
        # Legend
        ax_HeII_BCZ_H_residual_smooth_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Plot the residuals after the smooth, BCZ and H&HeI fit
        for l in ell:
            ax_smooth_BCZ_H_residual_HeII_fit.plot(nu_secdiff[l], residuals['smooth_BCZ_H'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')

        # Plot the HeII fit on the residuals after the smooth fit, the BCZ fit and the H fit
        ax_smooth_BCZ_H_residual_HeII_fit.plot(nu_fit, fit_HeII(nu_fit, *params[3:7]), \
                                               linestyle='solid', color='orange', marker='None', label='HeII fit')

        ax_smooth_BCZ_H_residual_HeII_fit.set_ylabel('Residual after\nsmooth\ncomponent,\nBCZ and\nH&HeI fit', fontsize=5)
        ax_smooth_BCZ_H_residual_HeII_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Horizontal 0-line
        ax_smooth_BCZ_H_residual_HeII_fit.axhline(0,linestyle='dotted', color='k')

        # Plot the residuals after the smooth, HeII and H&HeI fit
        for l in ell:
            ax_smooth_HeII_H_residual_BCZ_fit.plot(nu_secdiff[l], residuals['smooth_HeII_H'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')

        # Plot the BCZ fit on the residuals after the smooth fit, the HeII fit and the H fit
        ax_smooth_HeII_H_residual_BCZ_fit.plot(nu_fit, fit_BCZ(nu_fit, *params[7:10]), \
                                               linestyle='solid', color='orange', marker='None', label='BCZ fit')

        ax_smooth_HeII_H_residual_BCZ_fit.set_ylabel('Residual after\nsmooth\ncomponent,\nHeII and H&HeI fit', fontsize=5)
        ax_smooth_HeII_H_residual_BCZ_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Horizontal 0-line
        ax_smooth_HeII_H_residual_BCZ_fit.axhline(0,linestyle='dotted', color='k')

        # Plot the residuals after the smooth, HeII and H&HeI fit
        for l in ell:
            ax_smooth_HeII_BCZ_residual_H_fit.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ'][l],
                                                   color='k', marker=markers[l],
                                                   linestyle='none', markersize=3,
                                                   markeredgecolor='None')

        # Plot the BCZ fit on the residuals after the smooth fit, the HeII fit and the H fit
        ax_smooth_HeII_BCZ_residual_H_fit.plot(nu_fit, fit_H(nu_fit, *params[-4:]), \
                                               linestyle='solid', color='orange', marker='None', label='H&HeI fit')

        ax_smooth_HeII_BCZ_residual_H_fit.set_ylabel('Residual after\nsmooth\ncomponent,\nHeII and BCZ fit', fontsize=5)
        ax_smooth_HeII_BCZ_residual_H_fit.legend(loc='best', handletextpad=0.1, fontsize=7, handlelength=1)

        # Horizontal 0-line
        ax_smooth_HeII_BCZ_residual_H_fit.axhline(0,linestyle='dotted', color='k')


        # Residuos
        # Plot the data minus the model (residuals)
        for l in ell:
            ax_residual.plot(nu_secdiff[l], residuals['smooth_HeII_BCZ_H'][l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None',
                             label=r'$\ell={}$'.format(l))
    
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
      
        ax_residual.set_ylabel('Residual after\n total fit', fontsize=5)
        ax_residual.set_xlabel(r'Frequency ( $\mu$Hz )')
      
        ax_residual.annotate(r'$\chi^2_\nu($dof$={})={:.5}$'.format(residuals['smooth_HeII_BCZ_H']['dof'],residuals['smooth_HeII_BCZ_H']['reduced_chi2']),
                             xy=(1-0.22,0.07), xycoords='figure fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox={'edgecolor':'black', 'facecolor':'white'})
        
        # Legend
        ax_residual.legend(loc=(1.035,5.7), ncol=2, handletextpad=0.1, fontsize=9, handlelength=2)   
        
        # ID for-loop
        ax_HeII_BCZ_H_residual_smooth_fit.annotate('{}'.format(profile_and_number), xy=(1-0.22,0.12), 
                                               xycoords='figure fraction', horizontalalignment='left', 
                                               verticalalignment='center',
                                               bbox={'edgecolor':'black', 'facecolor':'white'})
    
        # No x-axis ticks
        plt.setp(ax_secdiff.get_xticklabels(), visible=False)
        plt.setp(ax_HeII_BCZ_H_residual_smooth_fit.get_xticklabels(), visible=False)
        plt.setp(ax_smooth_BCZ_H_residual_HeII_fit.get_xticklabels(), visible=False)
        plt.setp(ax_smooth_HeII_H_residual_BCZ_fit.get_xticklabels(), visible=False)
        plt.setp(ax_smooth_HeII_BCZ_residual_H_fit.get_xticklabels(), visible=False)

        # y-axis ticks
        plt.setp(ax_secdiff.get_yticklabels(), fontsize=7)
        plt.setp(ax_HeII_BCZ_H_residual_smooth_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_smooth_BCZ_H_residual_HeII_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_smooth_HeII_H_residual_BCZ_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_smooth_HeII_BCZ_residual_H_fit.get_yticklabels(), fontsize=7)
        plt.setp(ax_residual.get_yticklabels(), fontsize=7)

        # remove last and first tick label for the second subplot
        yticks_smooth_residual_HeII_fit = ax_secdiff.yaxis.get_major_ticks()
        yticks_smooth_residual_HeII_fit[-1].label1.set_visible(False)
        yticks_smooth_residual_HeII_fit[0].label1.set_visible(False)
        
        # remove last and first tick label for the second subplot
        yticks_smooth_residual_HeII_fit = ax_HeII_BCZ_H_residual_smooth_fit.yaxis.get_major_ticks()
        yticks_smooth_residual_HeII_fit[-1].label1.set_visible(False)
        yticks_smooth_residual_HeII_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_smooth_HeII_residual_BCZ_H_fit = ax_smooth_BCZ_H_residual_HeII_fit.yaxis.get_major_ticks()
        yticks_smooth_HeII_residual_BCZ_H_fit[-1].label1.set_visible(False)
        yticks_smooth_HeII_residual_BCZ_H_fit[0].label1.set_visible(False)
        
        # remove last and first tick label for the second subplot
        yticks_smooth_HeII_residual_BCZ_H_fit = ax_smooth_HeII_H_residual_BCZ_fit.yaxis.get_major_ticks()
        yticks_smooth_HeII_residual_BCZ_H_fit[-1].label1.set_visible(False)
        yticks_smooth_HeII_residual_BCZ_H_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_smooth_HeII_residual_BCZ_H_fit = ax_smooth_HeII_BCZ_residual_H_fit.yaxis.get_major_ticks()
        yticks_smooth_HeII_residual_BCZ_H_fit[-1].label1.set_visible(False)
        yticks_smooth_HeII_residual_BCZ_H_fit[0].label1.set_visible(False)

        # remove last and first tick label for the second subplot
        yticks_residual = ax_residual.yaxis.get_major_ticks()
        yticks_residual[-1].label1.set_visible(False)
        yticks_residual[0].label1.set_visible(False)
  
        # Creates free space outside the plot for the parameters display
        fig.subplots_adjust(right = 0.75, hspace=0)    
        
        return ax_secdiff, ax_HeII_BCZ_H_residual_smooth_fit, ax_smooth_BCZ_H_residual_HeII_fit, ax_smooth_HeII_H_residual_BCZ_fit, ax_smooth_HeII_BCZ_residual_H_fit, ax_residual

# ==============================================================================
    
def text_params(which,params):

    if which == 'smooth':
        text = r'$a_0={:.5}\ \mu$ Hz'.format(params[0]) + '\n'
        text = text + r'$a_1={:.5}$'.format(params[1]) + '\n'
        text = text + r'$a_2={:.5}$ $\mu$Hz$^-$$^1$'.format(params[2])
    
    if which == 'BCZ':
        text = r'$b_2={:.5}$ mHz$^3$'.format(params[0]) + '\n'
        text = text + r'$\tau_C$$_Z={:.5}$ s'.format(params[1]*1e6) + '\n'
        text = text + r'$\phi_C$$_Z={:.5}$'.format(params[2])
        
    if which == 'HeII':
        text = r'$c_0={:.5}$'.format(params[0]) + '\n'
        text = text + r'$c_2={:.5}$ $\mu$Hz$^-$$^2$'.format(params[1])+'\n'
        text = text + r'$\tau_H$$_e$$_I$$_I={:.5}$ s'.format(params[2]*1e6)+'\n'
        text = text + r'$\phi_H$$_e$$_I$$_I={:.5}$'.format(params[3])
        
    if which == 'H':       
        text = r'$d_0={:.5}$'.format(params[0]) + '\n'
        text = text + r'$d_2={:.5}$ $\mu$Hz$^-$$^2$'.format(params[1]) + '\n'
        text = text + r'$\tau_H$$_I={:.5}$ s'.format(params[2]*1e6) + '\n'
        text = text + r'$\phi_H$$_I={:.5}$'.format(params[3])

    if which == 'CZ_HeII':
        text = text_params('CZ',params[:3]) + '\n' + '\n'
        text = text + text_params('HeII',params[3:])
        
    if which == 'CZ_HI':
        text = text_params('CZ',params[:3]) + '\n' + '\n'
        text = text + text_params('HI',params[3:])
        
    if which == 'smooth_HeII_BCZ':
        text = text_params('smooth',params[:3]) + '\n' + '\n'
        text = text + text_params('HeII',params[3:7]) + '\n' + '\n'
        text = text + text_params('BCZ',params[7:]) 

    if which == 'smooth_HeII_BCZ_H':
        text = text_params('smooth',params[:3]) + '\n' + '\n'
        text = text + text_params('HeII',params[3:7]) + '\n' + '\n'
        text = text + text_params('BCZ',params[7:10]) + '\n' + '\n'
        text = text + text_params('H',params[10:])
        
    if which == 'CZ_HeII_HI':
        text = text_params('CZ',params[:3]) + '\n' + '\n'
        text = text + text_params('HeII',params[3:7]) + '\n' + '\n'
        text = text + text_params('HI',params[7:])
        
    if which == 'smooth_HeII':
        text = text_params('smooth',params[:3]) + '\n' + '\n'
        text = text + text_params('HeII',params[3:])


    return text

# ==============================================================================
# ==============================================================================

def fit_smooth(nu, # frecuency 
               a0, a1, a2): # smooth function

    smooth_function = a0 + a1*nu + a2*nu**2
   
    return smooth_function

# ==============================================================================
    
def fit_BCZ(nu, # frecuency 
            b2, tau_CZ, phi_CZ): # base of convection zone
    
    base_of_convective_zone = \
    (b2/nu**2)*np.sin( 4*np.pi*nu*tau_CZ + phi_CZ )

    return base_of_convective_zone

# ==============================================================================

def fit_HeII(nu, # frecuency 
             c0, c2, tau_HeII, phi_HeII): # HeII ionization zone
    
    HeII_ionization_zone =  \
    c0*nu*np.exp(-c2*nu**2) * np.sin( 4*np.pi*nu*tau_HeII + phi_HeII )

    return HeII_ionization_zone

# ==============================================================================
    
def fit_H(nu, # frecuency 
          d0, d2, tau_HI, phi_HI): # HI ionization zone             
    
    HI_ionization_zone = \
    d0*nu*np.exp(-d2*nu**2) * np.sin( 4*np.pi*nu*tau_HI + phi_HI )

    return  HI_ionization_zone

# ==============================================================================
    
def fit_BCZ_H(nu, # frecuency 
              b2, tau_CZ, phi_CZ, # base of convection zone
              d0, d2, tau_HI, phi_HI): # HI ionization zone             
    
    base_of_convective_zone = fit_BCZ(nu, b2, tau_CZ, phi_CZ)
    
    HI_ionization_zone = fit_H(nu, d0, d2, tau_HI, phi_HI)

    return  base_of_convective_zone + HI_ionization_zone

# ==============================================================================

def fit_smooth_HeII(nu, # frecuency 
                    a0, a1, a2, # smooth function
                    c0, c2, tau_HeII, phi_HeII): # HeII ionization zone
    
    smooth_function = fit_smooth(nu, a0, a1, a2)

    HeII_ionization_zone =fit_HeII(nu, c0, c2, tau_HeII, phi_HeII)

    return smooth_function + HeII_ionization_zone

# ==============================================================================
    
def fit_smooth_HeII_BCZ(nu, # frecuency 
                        a0, a1, a2, # smooth function
                        c0, c2, tau_HeII, phi_HeII, # HeII ionization zone
                        b2, tau_CZ, phi_CZ): # base of convection zone
    
    smooth_function = fit_smooth(nu, a0, a1, a2)

    base_of_convective_zone = fit_BCZ(nu, b2, tau_CZ, phi_CZ)
    
    HeII_ionization_zone =  fit_HeII(nu, c0, c2, tau_HeII, phi_HeII)

    return smooth_function + base_of_convective_zone + HeII_ionization_zone

# ==============================================================================
    
def fit_smooth_HeII_H(nu, # frecuency 
                      a0, a1, a2, # smooth function
                      c0, c2, tau_HeII, phi_HeII, # HeII ionization zone
                      d0, d2, tau_HI, phi_HI): # base of convection zone
    
    smooth_function = fit_smooth(nu, a0, a1, a2)

    HI_ionization_zone = fit_H(nu, d0, d2, tau_HI, phi_HI)
    
    HeII_ionization_zone =  fit_HeII(nu, c0, c2, tau_HeII, phi_HeII)

    return smooth_function + HI_ionization_zone + HeII_ionization_zone



# ==============================================================================

def fit_smooth_HeII_BCZ_H(nu, # frecuency 
                          a0, a1, a2, # smooth function
                          c0, c2, tau_HeII, phi_HeII, # HeII ionization zone
                          b2, tau_CZ, phi_CZ, # base of convection zone
                          d0, d2, tau_HI, phi_HI): # HI ionization zone             
    
    smooth_function = fit_smooth(nu, a0, a1, a2)

    base_of_convective_zone = fit_BCZ(nu, b2, tau_CZ, phi_CZ)
    
    HeII_ionization_zone = fit_HeII(nu, c0, c2, tau_HeII, phi_HeII)

    HI_ionization_zone = fit_H(nu, d0, d2, tau_HI, phi_HI)

    return smooth_function + base_of_convective_zone + \
           HeII_ionization_zone + HI_ionization_zone

# ==============================================================================

def fit_HeII_BCZ_H(nu, # frecuency 
                   c0, c2, tau_HeII, phi_HeII, # HeII ionization zone
                   b2, tau_CZ, phi_CZ, # base of convection zone
                   d0, d2, tau_HI, phi_HI): # HI ionization zone             
    
    base_of_convective_zone = fit_BCZ(nu, b2, tau_CZ, phi_CZ)
    
    HeII_ionization_zone = fit_HeII(nu, c0, c2, tau_HeII, phi_HeII)

    HI_ionization_zone = fit_H(nu, d0, d2, tau_HI, phi_HI)

    return base_of_convective_zone + HeII_ionization_zone + HI_ionization_zone

# ==============================================================================

def fit_smooth_BCZ_H(nu, # frecuency 
                     a0, a1, a2, # smooth function
                     b2, tau_CZ, phi_CZ, # base of convection zone
                     d0, d2, tau_HI, phi_HI): # HI ionization zone             
    
    base_of_convective_zone = fit_BCZ(nu, b2, tau_CZ, phi_CZ)
    
    smooth_function = fit_smooth(nu, a0, a1, a2)

    HI_ionization_zone = fit_H(nu, d0, d2, tau_HI, phi_HI)

    return base_of_convective_zone + smooth_function + HI_ionization_zone

# ==============================================================================
# ==============================================================================

def concat_secdiff(nu_secdiff_filtered_dict,
                   secdiff_filtered_dict,
                   nu_secdiff_dict,
                   secdiff_dict,
                   ind_filtered_out):
        
    '''
    Just in case, make sure that the order of the keys in the dictionary is
    0, 1, 2, 3. Does not matter if a l-value is skipped
    '''

    # The cov matrix will be created independently for each l-value
    covariance_matrix_dict = dict()
    size_covariance_matrix_dict = dict()


    # number of frequencies. This will define the size of the cov. matrix 
#    n_nu = 0

    # Create the covariance matrix for the non-filtered second differences
    for l,secdiff in secdiff_dict.items():       
        n_l = len(secdiff)       
        if n_l > 0:      
            cov_l = np.diag( np.full(n_l,1), 0 ) + \
                    np.diag( np.full(n_l-1,-4/6), 1 ) + \
                    np.diag( np.full(n_l-1,-4/6), -1 ) + \
                    np.diag( np.full(n_l-2,1/6), 2 ) + \
                    np.diag( np.full(n_l-2,1/6), -2 ) 

            covariance_matrix_dict[l] = cov_l
            size_covariance_matrix_dict[l] = n_l
           
            # add the number of second differences of corresponding l-value
#            n_nu += n_l
        
    # Remove the columns and rows from the cov matrix corresponding to the
    # filtered values. By l-value 
    for l,ind in ind_filtered_out.items():
        for i in ind[::-1]:
            # Delete row i
            covariance_matrix_dict[l] = np.delete(covariance_matrix_dict[l], i, 0)    
            # Delete column i
            covariance_matrix_dict[l] = np.delete(covariance_matrix_dict[l], i, 1) 
        # Correct the size of the covariance matrix by l-value
        size_covariance_matrix_dict[l] = size_covariance_matrix_dict[l] - len(ind)
    # Initialize   
    n_nu = np.sum([ size_covariance_matrix_dict[l] for l in size_covariance_matrix_dict ] )
    covariance_matrix = np.zeros( (n_nu, n_nu) ) 
    row1, row2, col1, col2 = 0, 0, 0, 0
    for l,cov_matrix in covariance_matrix_dict.items():

        row2 = size_covariance_matrix_dict[l] + row1
        col2 = size_covariance_matrix_dict[l] + col1

        covariance_matrix[row1:row2, col1:col2] = cov_matrix

        row1 = row2
        col1 = col2

    # Concatenate the second differences filtered 
    for i,l in enumerate(secdiff_filtered_dict):
        if i == 0:
            concat_secdiff = secdiff_filtered_dict[l]
            concat_nu_secdiff =  nu_secdiff_filtered_dict[l]
        else:
            concat_secdiff = np.concatenate([concat_secdiff, secdiff_filtered_dict[l]])
            concat_nu_secdiff = np.concatenate([concat_nu_secdiff, nu_secdiff_filtered_dict[l]])
    return concat_nu_secdiff, concat_secdiff, covariance_matrix

# =============================================================================

def widths(grid):
    # mid points
    mid_points = ( grid[1:] + grid[:-1] )/2
    # add left edge
    left_edge = mid_points[0] - grid[0]
    # add right edge
    right_edge = grid[-1] - mid_points[-1]
    # differences
    diff = np.diff(np.concatenate([ [left_edge], mid_points, [right_edge] ]))
    return np.abs(diff)
 
# =============================================================================
    
# =============================================================================
# =============================================================================

def second_differences(nu):
    
    nu_second_differences = nu[1:-1]
    second_differences = nu[:-2] - 2*nu[1:-1] + nu[2:]
    
    return nu_second_differences, second_differences

# =============================================================================
# =============================================================================

### Flags
PLOT_SECDIFF = True
PLOT_SMOOTH_FIT = True
PLOT_ALL_GRID = False
MAKE_PLOTS = False
FILE_NOTICE = True
IMPOSE_TAU_HEII_TO_BE_WITHIN_THE_SEARCHING_RANGE = False


### Constants and scales
time_scale = 1e9 # 1 Gyr in yr
sun_age = 4.6e9 # yr
R_sun = 6.95700e10 # cm
min_delta_age = 1e8/time_scale # Myr or 0.1 Gyr

file_notice_name = 'file_notice.txt'

# =============================================================================
### Read
# =============================================================================

# Read the frequencies of the model
frequencies_file = 'frequencies.bin'
f = open(frequencies_file,'rb')
freq = pickle.load(f)
f.close()

# Read the history
h = mr.MesaData('work/LOGS/history.data')
model_numbers, profile_numbers = np.genfromtxt('work/LOGS/profiles.index', usecols=(0,2), unpack=True, skip_header=1, dtype=int)

# =============================================================================
### Main code
# =============================================================================

# Variale where to save the results of all profiles
results = dict()

### Statistical error for the eigenfrequencies
statistical_error_eigenfrequencies = 0.05 # micro-Hertz
n_statistical_error_realizations = 100

### Subset
profile_subset = list(freq.keys())
#profile_subset = profile_subset[::2]
#profile_subset = ['profile28']

time1 = time.time()

for profile,f in freq.items(): # f stands for frequencies

    # Skip profiles not contained in profile_subset
    if not profile in profile_subset: continue

    pdf_hist = PdfPages('histograms.'+profile+'.pdf')
    pdf_secdiff = PdfPages('secdiff.'+profile+'.pdf')

    results_realizations = dict()
    
    for i_statistical_error_realization in range(n_statistical_error_realizations):
    
        # Variable where to save the results of each profile
        profile_results = dict()
    
        # Print which profile the loop is at
        print(' ' + profile)
        # Print it to a file (useful when submitting jobs to Condor, i.e., when using multiple procesors)
        if FILE_NOTICE:
            file_notice = open(file_notice_name,'a')
            file_notice.write(profile+', realization:'+str(i_statistical_error_realization)+'\n')
            file_notice.close()
            
        profile_number = profile.split('profile')[1]
        model_number = model_numbers[np.where(profile_numbers == int(profile_number))]
        model_number = int(model_number)
        age = h.star_age[model_number-1] / time_scale
                
        # Angular degree to use
        ell = np.unique(f['l'])
        # Save
        profile_results['l'] = ell
    
        # Save
        profile_results['age'] = age
        
        # =============================================================================
        ### Get the second differences
        # =============================================================================
        
        
        # Get the second differences in separated variables by the l-value
        secdiff = dict()
        nu_secdiff = dict()
        secdiff_no_statistical_errors = dict()
        nu_secdiff_no_statistical_errors = dict()
        f_nu = f['nu'].copy() 
        f_l = f['l'].copy() 
        f_n = f['n'].copy() 
        
        # Add statistical error to the eigen frequencies
        f_nu = f_nu + np.random.normal(0, statistical_error_eigenfrequencies, len(f_n))
        
        # Calculate the second differencese for each dregee l       
        nu_second_differences_l0, second_differences_l0 = second_differences(f_nu[f_l==0])
        nu_second_differences_l1, second_differences_l1 = second_differences(f_nu[f_l==1])
        nu_second_differences_l2, second_differences_l2 = second_differences(f_nu[f_l==2])
        nu_second_differences_l3, second_differences_l3 = second_differences(f_nu[f_l==3]) 
 
        # Correct extrema
        nu_second_differences_l0 = np.concatenate([[np.nan],nu_second_differences_l0,[np.nan]])
        nu_second_differences_l1 = np.concatenate([[np.nan],nu_second_differences_l1,[np.nan]])
        nu_second_differences_l2 = np.concatenate([[np.nan],nu_second_differences_l2,[np.nan]])
        nu_second_differences_l3 = np.concatenate([[np.nan],nu_second_differences_l3,[np.nan]])
        second_differences_l0 = np.concatenate([[np.nan],second_differences_l0,[np.nan]])
        second_differences_l1 = np.concatenate([[np.nan],second_differences_l1,[np.nan]])
        second_differences_l2 = np.concatenate([[np.nan],second_differences_l2,[np.nan]])
        second_differences_l3 = np.concatenate([[np.nan],second_differences_l3,[np.nan]])
        
        # Collect together
        types=np.dtype([('nu',float), ('n',int), ('l',int), ('secdiff',float)])
        f2=np.empty(len(f_nu),dtype=types)
        f2['nu']=f_nu
        f2['n']=f_n
        f2['l']=f_l
        f2['secdiff']=np.concatenate([second_differences_l0,second_differences_l1,second_differences_l2,second_differences_l3])        
        
        for l in ell:
            # Filter the same ell value
            condition1 = f2['l'] == l
            # Filter out the NaN the the begining and end of the second differences
            condition2 = np.logical_not( np.isnan(f2['secdiff']) )
            # Combine to get the indices of the secdiff for l
            ind =  np.all( (condition1, condition2), axis=0 )
            # Depending on the l-value, create the key in the dictionary and fill it
            secdiff[l] = f2['secdiff'][ind]
            nu_secdiff[l] = f2['nu'][ind]
            # Store a copy with out statistical errors
            condition1 = f['l'] == l
            condition2 = np.logical_not( np.isnan(f['secdiff']) )
            ind =  np.all( (condition1, condition2), axis=0 )
            secdiff_no_statistical_errors[l] = f['secdiff'][ind]
            nu_secdiff_no_statistical_errors[l] = f['nu'][ind]
      
        # Save
        profile_results['secdiff'] = secdiff
        profile_results['nu_secdiff'] = nu_secdiff
        profile_results['secdiff_no_statistical_errors'] = secdiff_no_statistical_errors
        profile_results['nu_secdiff_no_statistical_errors'] = nu_secdiff_no_statistical_errors
    
        # =============================================================================
        ### Filter the second differences
        # =============================================================================
    
        # Here we filter and get the indices of the second differences by l value
        secdiff_filtered = dict()
        nu_secdiff_filtered = dict()
        ind_filtered_out = dict() 
        for l in ell:
            # Filter out the second differences that are greater than 5
            condition = np.abs(secdiff[l]) < 5 
            # store in the dictionary
            secdiff_filtered[l] = secdiff[l][condition]
            nu_secdiff_filtered[l] = nu_secdiff[l][condition]
            ind_filtered_out[l] = np.argwhere( np.logical_not(condition) ).reshape(-1)
    
        # Concatenate the second differences. Get the covariance matrix for the future fit  
        # Note that the frequencies are by set of l-degree, i.e., not ordered      
        all_nu_secdiff_filtered, all_secdiff_filtered, cov_secdiff = concat_secdiff(nu_secdiff_filtered, secdiff_filtered,
                                                                                    nu_secdiff, secdiff, ind_filtered_out)
    
        # If at the end the residuals were greater than resuduals_threshold, 
        # then we will filter those points at the end of the code
        outliers = True
        while_max_loop = 5
        while_counter = 0
        while outliers:
            while_counter = while_counter+1
            
            # ordered
            all_nu_secdiff_filtered_sorted_arg = all_nu_secdiff_filtered.argsort()
            all_nu_secdiff_filtered_sorted = all_nu_secdiff_filtered[all_nu_secdiff_filtered_sorted_arg]
            all_secdiff_filtered_sorted = all_secdiff_filtered[all_nu_secdiff_filtered_sorted_arg]
        
            # Save
            profile_results['secdiff_filtered'] = secdiff_filtered
            profile_results['nu_secdiff_filtered'] = nu_secdiff_filtered
        
            # =============================================================================
            ### Fit smooth component
            # =============================================================================
            
            # INITIAL VALUES FOR THE FIT
            # Lets find the place where our polynomial of second order should be very close to the data
            # That is, the right side with less big oscillations
            windows = 20
            moving_std = pd.rolling_std(all_secdiff_filtered_sorted, windows)
            
            median_moving_std = np.median(moving_std[windows-1:])
            max_moving_std = np.max(moving_std[windows-1:])
            repeat_max_moving_std = np.repeat(max_moving_std,windows-1)
            moving_std = np.concatenate([repeat_max_moving_std, moving_std[windows-1:]])
        
            # indices with little fluctuation
            ind_low_fluctuation = np.where( moving_std < median_moving_std )
            
            nu_little_osc = np.mean(all_nu_secdiff_filtered_sorted[ind_low_fluctuation])
            
            # Initial guesses such that a0+a1x+a2x**2 is about 0 in the nu_little_osc
            a1_init_guess = 10**-3 # nu_little_osc ~ 3000-6000
            a0_init_guess = -1*(nu_little_osc * a1_init_guess)
            a2_init_guess = -1*(nu_little_osc ** -2)
            
            initial_guess_smooth = [ a0_init_guess, 
                                    a1_init_guess, 
                                    a2_init_guess ]
            
            # Limits or bounds for the fit    
            a0_lim = (a0_init_guess-10, a0_init_guess+10)
            a1_lim = (0, a1_init_guess*20)
            a2_lim = (a2_init_guess*100, 0)
            
            limits_smooth = ( [a0_lim[0], a1_lim[0], a2_lim[0] ], \
                              [a0_lim[1], a1_lim[1], a2_lim[1] ] ) 
            
            # Weigths: We change the covariance matrix to favor the low fluctuation
            cov_secdiff_smooth = cov_secdiff.copy()
            for ii in range(len(all_nu_secdiff_filtered)):
                if not ii in ind_low_fluctuation[0]:
                    cov_secdiff_smooth[ii,ii] = 2
        
            # FITTING
            params_smooth, params_smooth_covariance = curve_fit(fit_smooth, \
                                                                all_nu_secdiff_filtered, all_secdiff_filtered, \
                                                                p0=initial_guess_smooth, \
                                                                sigma=cov_secdiff_smooth, \
                                                                bounds=limits_smooth )
         
            # Residuos
            residuals_smooth_dict = dict()
            for l in ell:
                residuals_smooth_dict[l] = secdiff_filtered[l] - fit_smooth(nu_secdiff_filtered[l], *params_smooth)
            residuals_smooth_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth(all_nu_secdiff_filtered_sorted, *params_smooth)
            residuals_smooth_dict['all_l'] = all_secdiff_filtered - fit_smooth(all_nu_secdiff_filtered, *params_smooth)
            # Chi square
            #residuals_smooth_dict['chi2'] = np.sum(residuals_smooth_dict['all_l_ordered']**2)
            residuals_smooth_dict['chi2'] = residuals_smooth_dict['all_l'].T @ np.linalg.inv(cov_secdiff_smooth) @ residuals_smooth_dict['all_l']
            residuals_smooth_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(params_smooth)
            residuals_smooth_dict['reduced_chi2'] = residuals_smooth_dict['chi2'] / residuals_smooth_dict['dof']
        
            # Save. Store the results of the fit
            fit_smooth_dict = dict()
            fit_smooth_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
            fit_smooth_dict['secdiff_all_l'] = all_secdiff_filtered
            fit_smooth_dict['nu_low_std'] = all_nu_secdiff_filtered_sorted[ind_low_fluctuation]
            fit_smooth_dict['init_guess'] = {'a0':initial_guess_smooth[0],
                                             'a1':initial_guess_smooth[1],
                                             'a2':initial_guess_smooth[2]}
            fit_smooth_dict['limits'] = {'a0':a0_lim,
                                         'a1':a1_lim,
                                         'a2':a2_lim}
            fit_smooth_dict['results'] = {'a0':params_smooth[0],
                                          'a1':params_smooth[1],
                                          'a2':params_smooth[2]} 
            fit_smooth_dict['results_cov'] = params_smooth_covariance
            fit_smooth_dict['secdiff_cov'] = cov_secdiff_smooth
            fit_smooth_dict['residuals'] = residuals_smooth_dict
        
            profile_results['fit_smooth'] = fit_smooth_dict 
            
            # =============================================================================
            ### Fit smooth component and HeII ionization zone
            # =============================================================================
        
            # INITIAL VALUES FOR THE FIT
            # Let's find the region with high fluctuations
            windows = 20
            moving_std = pd.rolling_std(residuals_smooth_dict['all_l_ordered'], windows) # residuals are sorted!
            
            median_moving_std = np.median(moving_std[windows-1:])
            max_moving_std = np.max(moving_std[windows-1:])
            repeated_max_moving_std = np.repeat(max_moving_std,windows-1)
            moving_std = np.concatenate([repeated_max_moving_std, moving_std[windows-1:]])
        
            # indices with high fluctuation
            ind_high_fluctuation = np.where( moving_std > median_moving_std )
            nu_high_osc = np.mean(all_nu_secdiff_filtered_sorted[ind_high_fluctuation])
            
            # Find the estimative wavelength
            indexes_peak = peakutils.indexes(residuals_smooth_dict['all_l_ordered'], min_dist=15)
            # Choose the first two peaks
            nu_peak1 = all_nu_secdiff_filtered_sorted[indexes_peak[0]] 
            nu_peak2 = all_nu_secdiff_filtered_sorted[indexes_peak[1]] 
            Delta_nu_peak_He = nu_peak2 - nu_peak1
            
            high_amp = 1  # <-----------------------------------------
            low_amp = 0.2  # <-----------------------------------------
            max_nu = np.max(all_nu_secdiff_filtered)
            min_nu = np.min(all_nu_secdiff_filtered)
            # Initial guesses such that a0+a1x+a2x**2 is about 0 in the nu_little_osc
            c2_init_guess = np.log((high_amp/low_amp)*(max_nu/min_nu))/(max_nu**2-min_nu**2)
            c0_init_guess = (high_amp/max_nu)*np.exp(c2_init_guess*max_nu**2)
            tau_HeII_init_guess = 1/(2*Delta_nu_peak_He)
            phi_HeII_init_guess = np.pi
            
            initial_guess_HeII = [c0_init_guess,c2_init_guess,tau_HeII_init_guess,phi_HeII_init_guess]
        
    #       Limits or bounds for the fit    
            c0_uncertanty = 0.8 # <-----------------------------------------
            c0_lower_limit = c0_init_guess - c0_uncertanty*c0_init_guess
            c0_upper_limit = c0_init_guess + c0_uncertanty*c0_init_guess
            c2_uncertanty = 0.8 # <-----------------------------------------
            c2_lower_limit = c2_init_guess - c2_uncertanty*c2_init_guess
            c2_upper_limit = c2_init_guess + 2*c2_uncertanty*c2_init_guess  # <-----------------------------------------
            c0_lim = (c0_lower_limit, c0_upper_limit)
            c2_lim = (c2_lower_limit, c2_upper_limit)
    
            Delta_nu_peak_He_relative_uncertanty = 0.3 # <-----------------------------------------
            over_estimation_Delta_nu_peak_He = Delta_nu_peak_He + Delta_nu_peak_He_relative_uncertanty*Delta_nu_peak_He
            under_estimation_Delta_nu_peak_He = Delta_nu_peak_He - Delta_nu_peak_He_relative_uncertanty*Delta_nu_peak_He
            
            tau_HeII_lower_lim = 1/(2*(over_estimation_Delta_nu_peak_He))
            tau_HeII_upper_lim = 1/(2*(under_estimation_Delta_nu_peak_He))
            # TAU_HEII_SEARCHING_RANGE
            tau_HeII_lim = (tau_HeII_lower_lim,tau_HeII_upper_lim)
            phi_HeII_lim = (0,2*np.pi)
                    
            limits_HeII = ( [c0_lim[0], c2_lim[0], tau_HeII_lim[0], phi_HeII_lim[0] ], \
                            [c0_lim[1], c2_lim[1], tau_HeII_lim[1], phi_HeII_lim[1] ] ) 
            
            initial_guess = np.concatenate([params_smooth, initial_guess_HeII])
            limits_lower = np.concatenate([ limits_smooth[0],limits_HeII[0] ])
            limits_upper = np.concatenate([ limits_smooth[1],limits_HeII[1] ])
            limits = np.concatenate([ [limits_lower], [limits_upper] ])
        
            # FITTING
            params_smooth_HeII, params_smooth_HeII_covariance = curve_fit(fit_smooth_HeII, \
                                                                          all_nu_secdiff_filtered, all_secdiff_filtered, \
                                                                          p0=initial_guess, \
                                                                          sigma=cov_secdiff, \
                                                                          bounds=limits)
        
            # Residuos
            residuals_smooth_HeII_dict = dict()
            for l in ell:
                residuals_smooth_HeII_dict[l] = secdiff_filtered[l] - fit_smooth_HeII(nu_secdiff_filtered[l], *params_smooth_HeII)
            residuals_smooth_HeII_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII(all_nu_secdiff_filtered_sorted, *params_smooth_HeII)
            residuals_smooth_HeII_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII(all_nu_secdiff_filtered, *params_smooth_HeII)
            # Chi square
            #residuals_smooth_HeII_dict['chi2'] = np.sum(residuals_smooth_HeII_dict['all_l_ordered']**2)
            residuals_smooth_HeII_dict['chi2'] = residuals_smooth_HeII_dict['all_l'].T @ np.linalg.inv(cov_secdiff) @ residuals_smooth_HeII_dict['all_l']
            residuals_smooth_HeII_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(params_smooth_HeII)
            residuals_smooth_HeII_dict['reduced_chi2'] = residuals_smooth_HeII_dict['chi2'] / residuals_smooth_HeII_dict['dof']
            
            # Save. Store the results of the fit
            fit_smooth_HeII_dict = dict()
            fit_smooth_HeII_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
            fit_smooth_HeII_dict['secdiff_all_l'] = all_secdiff_filtered
            fit_smooth_HeII_dict['nu_peak1'] = nu_peak1
            fit_smooth_HeII_dict['nu_peak2'] = nu_peak2    
            fit_smooth_HeII_dict['Delta_nu_peak'] = Delta_nu_peak_He
            fit_smooth_HeII_dict['under_estimation_Delta_nu_peak'] = under_estimation_Delta_nu_peak_He
            fit_smooth_HeII_dict['over_estimation_Delta_nu_peak'] = over_estimation_Delta_nu_peak_He
            fit_smooth_HeII_dict['init_guess'] = {'a0':initial_guess[0],
                                                 'a1':initial_guess[1],
                                                 'a2':initial_guess[2],
                                                 'c0':initial_guess[3],
                                                 'c2':initial_guess[4],
                                                 'tau_HeII':initial_guess[5],
                                                 'phi_HeII':initial_guess[6]}
                
            fit_smooth_HeII_dict['limits'] = {'a0':a0_lim,
                                             'a1':a1_lim,
                                             'a2':a2_lim,
                                             'c0':c0_lim,
                                             'c2':c2_lim,
                                             'tau_HeII':tau_HeII_lim,
                                             'phi_HeII':phi_HeII_lim}
            
            fit_smooth_HeII_dict['results'] = {'a0':params_smooth_HeII[0],
                                              'a1':params_smooth_HeII[1],
                                              'a2':params_smooth_HeII[2],
                                              'c0':params_smooth_HeII[3],
                                              'c2':params_smooth_HeII[4],
                                              'tau_HeII':params_smooth_HeII[5],
                                              'phi_HeII':params_smooth_HeII[6]}
            
            
            fit_smooth_HeII_dict['results_cov'] = params_smooth_HeII_covariance
            fit_smooth_HeII_dict['secdiff_cov'] = cov_secdiff
            fit_smooth_HeII_dict['residuals'] = residuals_smooth_HeII_dict
        
            profile_results['fit_smooth_HeII'] = fit_smooth_HeII_dict 
        
            # =============================================================================
            ### Fit smooth component, the HeII ionization zone and the BCZ
            # =============================================================================
        
            # INITIAL VALUES FOR THE FIT
            # Find the estimative wavelength
            indexes_peak = peakutils.indexes(residuals_smooth_HeII_dict['all_l_ordered'], min_dist=3)
            nu_peak = all_nu_secdiff_filtered_sorted[indexes_peak] 
            Delta_nu_peak = np.diff(nu_peak)
            Delta_nu_peak = np.median(Delta_nu_peak)
            
            # Initial guesses. The oscillation due to the BCZ do not decay much.
            # Then we can take the mid point of the range
        #    b2_init_guess = 1*(np.median(all_nu_secdiff_sorted))**2
            b2_init_guess = 0.15*max_nu**2  # <-----------------------------------------
            tau_BCZ_init_guess = 1/(2*Delta_nu_peak)
        #    tau_BCZ_init_guess = 1880.23e-6#1/(2*Delta_nu_peak)
            phi_BCZ_init_guess = np.pi
            
        #    print('BCZ tau_BCZ_init_guess MEAN', tau_BCZ_init_guess*1e6)
            
            initial_guess_BCZ = [b2_init_guess,tau_BCZ_init_guess,phi_BCZ_init_guess]
        
            initial_guess = np.concatenate([params_smooth_HeII, initial_guess_BCZ])
        
            # Limits or bounds for the fit
            b2_uncertanty = 0.8
            b2_lower_limit = b2_init_guess - b2_uncertanty*b2_init_guess
            b2_upper_limit = b2_init_guess + b2_uncertanty*b2_init_guess
            b2_lim = (b2_lower_limit, b2_upper_limit)
    #        b2_lim = (0.5*b2_init_guess, b2_init_guess*2)
        #    tau_BCZ_lower_lim = 1/(2*(2*Delta_nu_peak))
        #    tau_BCZ_upper_lim = 1/(2*(0.5*Delta_nu_peak))
    
            Delta_nu_peak_relative_uncertanty = 0.3 
            over_estimation_Delta_nu_peak = Delta_nu_peak + Delta_nu_peak_relative_uncertanty*Delta_nu_peak
            under_estimation_Delta_nu_peak = Delta_nu_peak - Delta_nu_peak_relative_uncertanty*Delta_nu_peak
            tau_BCZ_lower_lim = 1/(2*(over_estimation_Delta_nu_peak))
            tau_BCZ_upper_lim = 1/(2*(under_estimation_Delta_nu_peak))
    
    #        tau_BCZ_lower_lim = 1/(2*(1.05*Delta_nu_peak))
    #        tau_BCZ_upper_lim = 1/(2*((1/1.05)*Delta_nu_peak))
            tau_BCZ_lim = (tau_BCZ_lower_lim,tau_BCZ_upper_lim)
            phi_BCZ_lim = (0,2*np.pi)
            
        #    limits_lower_smooth_HeII =  params_smooth_HeII - np.abs(params_smooth_HeII)*0.01
        #    limits_upper_smooth_HeII =  params_smooth_HeII + np.abs(params_smooth_HeII)*0.01
            
            limits_BCZ = ( [b2_lim[0], tau_BCZ_lim[0], phi_BCZ_lim[0]], \
                            [b2_lim[1], tau_BCZ_lim[1], phi_BCZ_lim[1]] ) 
        
        #    limits_lower = np.concatenate([ limits_lower_smooth_HeII, limits_BCZ[0] ])
        #    limits_upper = np.concatenate([ limits_upper_smooth_HeII, limits_BCZ[1] ])
        #    limits = np.concatenate([ [limits_lower], [limits_upper] ])
       
            limits_lower = np.concatenate([ limits_smooth[0],limits_HeII[0], limits_BCZ[0] ])
            limits_upper = np.concatenate([ limits_smooth[1],limits_HeII[1], limits_BCZ[1] ])
            limits = np.concatenate([ [limits_lower], [limits_upper] ])
        
            # FITTING
            try:
                params_smooth_HeII_BCZ, params_smooth_HeII_BCZ_covariance = curve_fit(fit_smooth_HeII_BCZ, \
                                                                                      all_nu_secdiff_filtered, all_secdiff_filtered, \
                                                                                      p0=initial_guess, \
                                                                                      sigma=cov_secdiff, \
                                                                                      bounds=limits)
            except RuntimeError:
                
                # =============================================================================
                ### Change the statistical errors
                # =============================================================================
                
                # Get the second differences in separated variables by the l-value
                secdiff = dict()
                nu_secdiff = dict()
                secdiff_no_statistical_errors = dict()
                nu_secdiff_no_statistical_errors = dict()
                f_nu = f['nu'].copy() 
                f_l = f['l'].copy() 
                f_n = f['n'].copy() 
                
                # Add statistical error to the eigen frequencies
                f_nu = f_nu + np.random.normal(0, statistical_error_eigenfrequencies, len(f_n))
                
                # Calculate the second differencese for each dregee l       
                nu_second_differences_l0, second_differences_l0 = second_differences(f_nu[f_l==0])
                nu_second_differences_l1, second_differences_l1 = second_differences(f_nu[f_l==1])
                nu_second_differences_l2, second_differences_l2 = second_differences(f_nu[f_l==2])
                nu_second_differences_l3, second_differences_l3 = second_differences(f_nu[f_l==3]) 
         
                # Correct extrema
                nu_second_differences_l0 = np.concatenate([[np.nan],nu_second_differences_l0,[np.nan]])
                nu_second_differences_l1 = np.concatenate([[np.nan],nu_second_differences_l1,[np.nan]])
                nu_second_differences_l2 = np.concatenate([[np.nan],nu_second_differences_l2,[np.nan]])
                nu_second_differences_l3 = np.concatenate([[np.nan],nu_second_differences_l3,[np.nan]])
                second_differences_l0 = np.concatenate([[np.nan],second_differences_l0,[np.nan]])
                second_differences_l1 = np.concatenate([[np.nan],second_differences_l1,[np.nan]])
                second_differences_l2 = np.concatenate([[np.nan],second_differences_l2,[np.nan]])
                second_differences_l3 = np.concatenate([[np.nan],second_differences_l3,[np.nan]])
                
                # Collect together
                types=np.dtype([('nu',float), ('n',int), ('l',int), ('secdiff',float)])
                f2=np.empty(len(f_nu),dtype=types)
                f2['nu']=f_nu
                f2['n']=f_n
                f2['l']=f_l
                f2['secdiff']=np.concatenate([second_differences_l0,second_differences_l1,second_differences_l2,second_differences_l3])        
                
                for l in ell:
                    # Filter the same ell value
                    condition1 = f2['l'] == l
                    # Filter out the NaN the the begining and end of the second differences
                    condition2 = np.logical_not( np.isnan(f2['secdiff']) )
                    # Combine to get the indices of the secdiff for l
                    ind =  np.all( (condition1, condition2), axis=0 )
                    # Depending on the l-value, create the key in the dictionary and fill it
                    secdiff[l] = f2['secdiff'][ind]
                    nu_secdiff[l] = f2['nu'][ind]
                    # Store a copy with out statistical errors
                    condition1 = f['l'] == l
                    condition2 = np.logical_not( np.isnan(f['secdiff']) )
                    ind =  np.all( (condition1, condition2), axis=0 )
                    secdiff_no_statistical_errors[l] = f['secdiff'][ind]
                    nu_secdiff_no_statistical_errors[l] = f['nu'][ind]
              
                # Save
                profile_results['secdiff'] = secdiff
                profile_results['nu_secdiff'] = nu_secdiff
                profile_results['secdiff_no_statistical_errors'] = secdiff_no_statistical_errors
                profile_results['nu_secdiff_no_statistical_errors'] = nu_secdiff_no_statistical_errors
            
                # =============================================================================
                ### Filter the second differences
                # =============================================================================
            
                # Here we filter and get the indices of the second differences by l value
                secdiff_filtered = dict()
                nu_secdiff_filtered = dict()
                ind_filtered_out = dict() 
                for l in ell:
                    # Filter out the second differences that are greater than 5
                    condition = np.abs(secdiff[l]) < 5 
                    # store in the dictionary
                    secdiff_filtered[l] = secdiff[l][condition]
                    nu_secdiff_filtered[l] = nu_secdiff[l][condition]
                    ind_filtered_out[l] = np.argwhere( np.logical_not(condition) ).reshape(-1)
            
                # Concatenate the second differences. Get the covariance matrix for the future fit  
                # Note that the frequencies are by set of l-degree, i.e., not ordered      
                all_nu_secdiff_filtered, all_secdiff_filtered, cov_secdiff = concat_secdiff(nu_secdiff_filtered, secdiff_filtered,
                                                                                            nu_secdiff, secdiff, ind_filtered_out)
                if FILE_NOTICE:
                    file_notice = open(file_notice_name,'a')
                    file_notice.write('The statistical error has been changed during WHILE NUMBER '+str(while_counter)+', realization: '+str(i_statistical_error_realization)+'\n')
                    file_notice.close()
                    
                if while_counter >= while_max_loop:
                    file_notice = open(file_notice_name,'a')
                    file_notice.write('The statistical error has been changed more than '+str(while_max_loop)+' times. '+profile+' has been skipped.\n')
                    file_notice.close()
                    break

                continue
                
            # Residuos
            residuals_smooth_HeII_BCZ_dict = dict()
            for l in ell:
                residuals_smooth_HeII_BCZ_dict[l] = secdiff_filtered[l] - fit_smooth_HeII_BCZ(nu_secdiff_filtered[l], *params_smooth_HeII_BCZ)
            residuals_smooth_HeII_BCZ_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_BCZ(all_nu_secdiff_filtered_sorted, *params_smooth_HeII_BCZ)
            residuals_smooth_HeII_BCZ_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_BCZ(all_nu_secdiff_filtered, *params_smooth_HeII_BCZ)
            # Chi square
            #residuals_smooth_HeII_BCZ_dict['chi2'] = np.sum(residuals_smooth_HeII_BCZ_dict['all_l_ordered']**2)
            residuals_smooth_HeII_BCZ_dict['chi2'] = residuals_smooth_HeII_BCZ_dict['all_l'].T @ np.linalg.inv(cov_secdiff) @ residuals_smooth_HeII_BCZ_dict['all_l']
            residuals_smooth_HeII_BCZ_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(params_smooth_HeII_BCZ)
            residuals_smooth_HeII_BCZ_dict['reduced_chi2'] = residuals_smooth_HeII_BCZ_dict['chi2'] / residuals_smooth_HeII_BCZ_dict['dof']
         
            # Save. Store the results of the fit
            fit_smooth_HeII_BCZ_dict = dict()
            fit_smooth_HeII_BCZ_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
            fit_smooth_HeII_BCZ_dict['secdiff_all_l'] = all_secdiff_filtered
            fit_smooth_HeII_BCZ_dict['nu_peaks'] = nu_peak
            fit_smooth_HeII_BCZ_dict['Delta_nu_peak'] = Delta_nu_peak
            fit_smooth_HeII_BCZ_dict['under_estimation_Delta_nu_peak'] = under_estimation_Delta_nu_peak
            fit_smooth_HeII_BCZ_dict['over_estimation_Delta_nu_peak'] = over_estimation_Delta_nu_peak
            fit_smooth_HeII_BCZ_dict['init_guess'] = {'a0':initial_guess[0],
                                                 'a1':initial_guess[1],
                                                 'a2':initial_guess[2],
                                                 'c0':initial_guess[3],
                                                 'c2':initial_guess[4],
                                                 'tau_HeII':initial_guess[5],
                                                 'phi_HeII':initial_guess[6],
                                                 'b2':initial_guess[7],
                                                 'tau_BCZ':initial_guess[8],
                                                 'phi_BCZ':initial_guess[9]}
                
            fit_smooth_HeII_BCZ_dict['limits'] = {'a0':a0_lim,
                                             'a1':a1_lim,
                                             'a2':a2_lim,
                                             'c0':c0_lim,
                                             'c2':c2_lim,
                                             'tau_HeII':tau_HeII_lim,
                                             'phi_HeII':phi_HeII_lim,
                                             'b2':b2_lim,
                                             'tau_BCZ':tau_BCZ_lim,
                                             'phi_BCZ':phi_BCZ_lim}
            
            fit_smooth_HeII_BCZ_dict['results'] = {'a0':params_smooth_HeII_BCZ[0],
                                              'a1':params_smooth_HeII_BCZ[1],
                                              'a2':params_smooth_HeII_BCZ[2],
                                              'c0':params_smooth_HeII_BCZ[3],
                                              'c2':params_smooth_HeII_BCZ[4],
                                              'tau_HeII':params_smooth_HeII_BCZ[5],
                                              'phi_HeII':params_smooth_HeII_BCZ[6],
                                              'b2':params_smooth_HeII_BCZ[7],
                                              'tau_BCZ':params_smooth_HeII_BCZ[8],
                                              'phi_BCZ':params_smooth_HeII_BCZ[9]}
            
            
            fit_smooth_HeII_BCZ_dict['results_cov'] = params_smooth_HeII_BCZ_covariance
            fit_smooth_HeII_BCZ_dict['secdiff_cov'] = cov_secdiff
            fit_smooth_HeII_BCZ_dict['residuals'] = residuals_smooth_HeII_BCZ_dict
        
            profile_results['fit_smooth_HeII_BCZ'] = fit_smooth_HeII_BCZ_dict 
        
            # =============================================================================
            ### Fit smooth component, the HeII ionization zone, the BCZ and the H&HeI ionization zone 
            # =============================================================================
        
            # INITIAL VALUES FOR THE FIT
            # Find the estimative wavelength
            # Smooth curve
            windows = 20
            residuals_smooth_HeII_BCZ_smooth = pd.rolling_mean(residuals_smooth_HeII_BCZ_dict['all_l_ordered'],windows)
            # Drop the NaN at the beginning    
            residuals_smooth_HeII_BCZ_smooth = residuals_smooth_HeII_BCZ_smooth[windows-1:]
            # Get the value at the beginning
            first_sign = residuals_smooth_HeII_BCZ_smooth[0]
            # Copy it at the beginning to make the array the same length as before 
            repeated_first_sign = np.repeat(first_sign,windows-1)
            residuals_smooth_HeII_BCZ_smooth = np.concatenate([repeated_first_sign, residuals_smooth_HeII_BCZ_smooth])
            residuals_smooth_HeII_BCZ_smooth_sign = np.sign(residuals_smooth_HeII_BCZ_smooth)
        
            # Let's  find where it changes sign
            sign_change = residuals_smooth_HeII_BCZ_smooth_sign[:-1]-residuals_smooth_HeII_BCZ_smooth_sign[1:]
            sign_change_arg = np.where(np.abs(sign_change) == 2)
            nu_sign_change = all_nu_secdiff_filtered_sorted[sign_change_arg]
            Delta_half_peak = np.diff(nu_sign_change)
            Delta_half_peak = np.mean(Delta_half_peak)
            Delta_nu_peak = 2*Delta_half_peak
        #    Delta_nu_peak = 1.85*Delta_nu_peak_He
        #    print(Delta_nu_peak_He)
        #    print(Delta_nu_peak)
                
            # Initial guesses.
            # We take the region with little oscillation
            d0_init_guess = 1/nu_little_osc # <-----------------------------------------
            d2_init_guess = 1/nu_little_osc**2 # <-----------------------------------------
            tau_H_init_guess = 1/(2*Delta_nu_peak) # <-----------------------------------------
        #    print('tau_H_init_guess', tau_H_init_guess)
            tau_H_init_guess = tau_HeII_init_guess/1.67#327.16e-6 # <-----------------------------------------
        #    print('tau_H_init_guess', tau_H_init_guess)
            phi_H_init_guess = np.pi
            
            
            initial_guess_H = [d0_init_guess,d2_init_guess,tau_H_init_guess,phi_H_init_guess]
        
            # Limits or bounds for the fit    
            d0_lim = (0, d0_init_guess*100)
            d2_lim = (0, d2_init_guess*100)
        #    tau_H_lower_lim = 1/(2*(5*Delta_nu_peak))
        #    tau_H_upper_lim = 1/(2*((1/5)*Delta_nu_peak))
    #        tau_H_lower_lim = tau_H_init_guess - tau_H_init_guess*0.02
    #        tau_H_upper_lim = tau_H_init_guess + tau_H_init_guess*0.02
            tau_H_lower_lim = tau_H_init_guess - tau_H_init_guess*0.8
            tau_H_upper_lim = tau_H_init_guess + tau_H_init_guess*0.8
            tau_H_lim = (tau_H_lower_lim,tau_H_upper_lim)
            phi_H_lim = (0,2*np.pi)    
            
        #    print('tau_H_lim',tau_H_lim)
            
            limits_H = ( [d0_lim[0], d2_lim[0], tau_H_lim[0], phi_H_lim[0]], \
                            [d0_lim[1], d2_lim[1], tau_H_lim[1], phi_H_lim[1]] ) 
            
            initial_guess = np.concatenate([params_smooth_HeII_BCZ, initial_guess_H])
        #    initial_guess[5] = 0.00697411/(4*np.pi) # tau_HeII
        #    initial_guess[8] = 0.0234842/(4*np.pi) # tau_BCZ
        #    initial_guess[12] = 0.002/(4*np.pi)#initial_guess[5]/1.67#0.00482912/(4*np.pi) # tau_H
        #    embed()
            limits_lower = np.concatenate([ limits_smooth[0], limits_HeII[0], limits_BCZ[0], limits_H[0] ])
            limits_upper = np.concatenate([ limits_smooth[1], limits_HeII[1], limits_BCZ[1], limits_H[1] ])
        #    limits_lower[12] = 0.001/(4*np.pi)
        #    limits_upper[12] = 0.003/(4*np.pi)
        #    limits_lower[12] = limits_lower[12]/2
        #    limits_upper[12] = limits_upper[12]*1.5
        #    limits_upper[5] = limits_upper[5]*1.1
        #    limits_lower[5] = limits_lower[5]/10
            limits = np.concatenate([ [limits_lower], [limits_upper] ])
        
        #    initial_guess = [-6.9348, 0.0035592, -4.3278e-7,
        #                     0.0020146, 2.1905e-7, 555.11e-6, 9.3742-2*np.pi,
        #                     3.6516e6, 1867.7e-6, 7.7342-2*np.pi,
        #                     0.00010636, 7.3791e-8, 332.54e-6, 1e-30]
        
        #    # Weigths: We change the covariance matrix to favor the low fluctuation
        #    cov_secdiff_smooth_HeII_BCZ_H = cov_secdiff.copy()
        #    diag = np.repeat(1, cov_secdiff_smooth_HeII_BCZ_H.shape[0])
        #    diag[:7] = 2
        #    diag[-7:] = 2
        #    diag_ind = np.diag_indices(cov_secdiff_smooth_HeII_BCZ_H.shape[0])
        #    cov_secdiff_smooth_HeII_BCZ_H[diag_ind] = diag
                    
            # FITTING
            try:
                params_smooth_HeII_BCZ_H, params_smooth_HeII_BCZ_H_covariance = curve_fit(fit_smooth_HeII_BCZ_H, \
                                                                                          all_nu_secdiff_filtered, all_secdiff_filtered, \
                                                                                          p0=initial_guess, \
                                                                                          sigma=cov_secdiff, \
                                                                                          bounds=limits)
            except RuntimeError:
                
                # =============================================================================
                ### Change the statistical errors
                # =============================================================================
                
                # Get the second differences in separated variables by the l-value
                secdiff = dict()
                nu_secdiff = dict()
                secdiff_no_statistical_errors = dict()
                nu_secdiff_no_statistical_errors = dict()
                f_nu = f['nu'].copy() 
                f_l = f['l'].copy() 
                f_n = f['n'].copy() 
                
                # Add statistical error to the eigen frequencies
                f_nu = f_nu + np.random.normal(0, statistical_error_eigenfrequencies, len(f_n))
                
                # Calculate the second differencese for each dregee l       
                nu_second_differences_l0, second_differences_l0 = second_differences(f_nu[f_l==0])
                nu_second_differences_l1, second_differences_l1 = second_differences(f_nu[f_l==1])
                nu_second_differences_l2, second_differences_l2 = second_differences(f_nu[f_l==2])
                nu_second_differences_l3, second_differences_l3 = second_differences(f_nu[f_l==3]) 
         
                # Correct extrema
                nu_second_differences_l0 = np.concatenate([[np.nan],nu_second_differences_l0,[np.nan]])
                nu_second_differences_l1 = np.concatenate([[np.nan],nu_second_differences_l1,[np.nan]])
                nu_second_differences_l2 = np.concatenate([[np.nan],nu_second_differences_l2,[np.nan]])
                nu_second_differences_l3 = np.concatenate([[np.nan],nu_second_differences_l3,[np.nan]])
                second_differences_l0 = np.concatenate([[np.nan],second_differences_l0,[np.nan]])
                second_differences_l1 = np.concatenate([[np.nan],second_differences_l1,[np.nan]])
                second_differences_l2 = np.concatenate([[np.nan],second_differences_l2,[np.nan]])
                second_differences_l3 = np.concatenate([[np.nan],second_differences_l3,[np.nan]])
                
                # Collect together
                types=np.dtype([('nu',float), ('n',int), ('l',int), ('secdiff',float)])
                f2=np.empty(len(f_nu),dtype=types)
                f2['nu']=f_nu
                f2['n']=f_n
                f2['l']=f_l
                f2['secdiff']=np.concatenate([second_differences_l0,second_differences_l1,second_differences_l2,second_differences_l3])        
                
                for l in ell:
                    # Filter the same ell value
                    condition1 = f2['l'] == l
                    # Filter out the NaN the the begining and end of the second differences
                    condition2 = np.logical_not( np.isnan(f2['secdiff']) )
                    # Combine to get the indices of the secdiff for l
                    ind =  np.all( (condition1, condition2), axis=0 )
                    # Depending on the l-value, create the key in the dictionary and fill it
                    secdiff[l] = f2['secdiff'][ind]
                    nu_secdiff[l] = f2['nu'][ind]
                    # Store a copy with out statistical errors
                    condition1 = f['l'] == l
                    condition2 = np.logical_not( np.isnan(f['secdiff']) )
                    ind =  np.all( (condition1, condition2), axis=0 )
                    secdiff_no_statistical_errors[l] = f['secdiff'][ind]
                    nu_secdiff_no_statistical_errors[l] = f['nu'][ind]
              
                # Save
                profile_results['secdiff'] = secdiff
                profile_results['nu_secdiff'] = nu_secdiff
                profile_results['secdiff_no_statistical_errors'] = secdiff_no_statistical_errors
                profile_results['nu_secdiff_no_statistical_errors'] = nu_secdiff_no_statistical_errors
            
                # =============================================================================
                ### Filter the second differences
                # =============================================================================
            
                # Here we filter and get the indices of the second differences by l value
                secdiff_filtered = dict()
                nu_secdiff_filtered = dict()
                ind_filtered_out = dict() 
                for l in ell:
                    # Filter out the second differences that are greater than 5
                    condition = np.abs(secdiff[l]) < 5 
                    # store in the dictionary
                    secdiff_filtered[l] = secdiff[l][condition]
                    nu_secdiff_filtered[l] = nu_secdiff[l][condition]
                    ind_filtered_out[l] = np.argwhere( np.logical_not(condition) ).reshape(-1)
            
                # Concatenate the second differences. Get the covariance matrix for the future fit  
                # Note that the frequencies are by set of l-degree, i.e., not ordered      
                all_nu_secdiff_filtered, all_secdiff_filtered, cov_secdiff = concat_secdiff(nu_secdiff_filtered, secdiff_filtered,
                                                                                            nu_secdiff, secdiff, ind_filtered_out)
                if FILE_NOTICE:
                    file_notice = open(file_notice_name,'a')
                    file_notice.write('The statistical error has been changed during WHILE NUMBER '+str(while_counter)+', realization: '+str(i_statistical_error_realization)+'\n')
                    file_notice.close()
                    
                if while_counter >= while_max_loop:
                    file_notice = open(file_notice_name,'a')
                    file_notice.write('The statistical error has been changed more than '+str(while_max_loop)+' times. '+profile+' has been skipped.\n')
                    file_notice.close()
                    break

                continue

            # Residuos
            residuals_smooth_HeII_BCZ_H_dict = dict()
            for l in ell:
                residuals_smooth_HeII_BCZ_H_dict[l] = secdiff_filtered[l] - fit_smooth_HeII_BCZ_H(nu_secdiff_filtered[l], *params_smooth_HeII_BCZ_H)
            residuals_smooth_HeII_BCZ_H_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered_sorted, *params_smooth_HeII_BCZ_H)
            residuals_smooth_HeII_BCZ_H_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered, *params_smooth_HeII_BCZ_H)
            # Chi square
            #residuals_smooth_HeII_BCZ_H_dict['chi2'] = np.sum(residuals_smooth_HeII_BCZ_H_dict['all_l_ordered']**2)
            residuals_smooth_HeII_BCZ_H_dict['chi2'] = residuals_smooth_HeII_BCZ_H_dict['all_l'].T @ np.linalg.inv(cov_secdiff) @ residuals_smooth_HeII_BCZ_H_dict['all_l']
            residuals_smooth_HeII_BCZ_H_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(params_smooth_HeII_BCZ_H)
            residuals_smooth_HeII_BCZ_H_dict['reduced_chi2'] = residuals_smooth_HeII_BCZ_H_dict['chi2'] / residuals_smooth_HeII_BCZ_H_dict['dof']
            
            # Save. Store the results of the fit
            fit_smooth_HeII_BCZ_H_dict = dict()
            fit_smooth_HeII_BCZ_H_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
            fit_smooth_HeII_BCZ_H_dict['secdiff_all_l'] = all_secdiff_filtered
            fit_smooth_HeII_BCZ_H_dict['init_guess'] = {'a0':initial_guess[0],
                                                 'a1':initial_guess[1],
                                                 'a2':initial_guess[2],
                                                 'c0':initial_guess[3],
                                                 'c2':initial_guess[4],
                                                 'tau_HeII':initial_guess[5],
                                                 'phi_HeII':initial_guess[6],
                                                 'b2':initial_guess[7],
                                                 'tau_BCZ':initial_guess[8],
                                                 'phi_BCZ':initial_guess[9],
                                                 'd0_H':initial_guess[10],
                                                 'd2_H':initial_guess[11],
                                                 'tau_H':initial_guess[12],
                                                 'phi_H':initial_guess[13]}
                
            fit_smooth_HeII_BCZ_H_dict['limits'] = {'a0':a0_lim,
                                             'a1':a1_lim,
                                             'a2':a2_lim,
                                             'c0':c0_lim,
                                             'c2':c2_lim,
                                             'tau_HeII':tau_HeII_lim,
                                             'phi_HeII':phi_HeII_lim,
                                             'b2':b2_lim,
                                             'tau_BCZ':tau_BCZ_lim,
                                             'phi_BCZ':phi_BCZ_lim,
                                             'd0':d0_lim,
                                             'd2':d2_lim,
                                             'tau_H':tau_H_lim,
                                             'phi_H':phi_H_lim}
            
            fit_smooth_HeII_BCZ_H_dict['results'] = {'a0':params_smooth_HeII_BCZ_H[0],
                                              'a1':params_smooth_HeII_BCZ_H[1],
                                              'a2':params_smooth_HeII_BCZ_H[2],
                                              'c0':params_smooth_HeII_BCZ_H[3],
                                              'c2':params_smooth_HeII_BCZ_H[4],
                                              'tau_HeII':params_smooth_HeII_BCZ_H[5],
                                              'phi_HeII':params_smooth_HeII_BCZ_H[6],
                                              'b2':params_smooth_HeII_BCZ_H[7],
                                              'tau_BCZ':params_smooth_HeII_BCZ_H[8],
                                              'phi_BCZ':params_smooth_HeII_BCZ_H[9],
                                              'd0':params_smooth_HeII_BCZ_H[10],
                                              'd2':params_smooth_HeII_BCZ_H[11],
                                              'tau_H':params_smooth_HeII_BCZ_H[12],
                                              'pi_H':params_smooth_HeII_BCZ_H[13],}
            
            
            fit_smooth_HeII_BCZ_H_dict['results_cov'] = params_smooth_HeII_BCZ_H_covariance
            fit_smooth_HeII_BCZ_H_dict['secdiff_cov'] = cov_secdiff
            fit_smooth_HeII_BCZ_H_dict['residuals'] = residuals_smooth_HeII_BCZ_H_dict
        
            profile_results['fit_smooth_HeII_BCZ_H'] = fit_smooth_HeII_BCZ_H_dict 
        
        
            # =============================================================================
            ### Grid
            # =============================================================================
        
            # Grid of acoustic depths to search around the solution
        #    tau_BCZ_grid = np.array([ -90, -60, -30, 0 ,30 ,60 ,90 ])
        #    tau_HeII_grid = np.array([ -90, -60, -30, 0 ,30 ,60 ,90 ])
        #    tau_H_grid = np.array([ -120, -90, -60, -30, 0 ,30 ,60 ,90, 120 ])
    #        tau_BCZ_grid = np.arange(-300,320,20)
    #        tau_HeII_grid = np.arange(-300,320,20)
    #        tau_H_grid = np.arange(-300,320,20)
    #        tau_BCZ_grid = np.arange(tau_BCZ_lim[0]*1e6, tau_BCZ_lim[1]*1e6, 20)
    #        tau_HeII_grid = np.arange(tau_HeII_lim[0]*1e6, tau_HeII_lim[1]*1e6, 20)
    #        tau_H_grid = np.arange(tau_H_lim[0]*1e6, tau_H_lim[1]*1e6, 20)
            tau_BCZ_grid = np.linspace(tau_BCZ_lim[0]*1e6, tau_BCZ_lim[1]*1e6, 16)
            tau_HeII_grid = np.linspace(tau_HeII_lim[0]*1e6, tau_HeII_lim[1]*1e6, 16)
            tau_H_grid = np.linspace(tau_H_lim[0]*1e6, tau_H_lim[1]*1e6, 16)
            
            # widths for the plots
    #        tau_BCZ_grid_widths = widths(tau_BCZ_grid)    
    #        tau_HeII_grid_widths = widths(tau_HeII_grid)    
    #        tau_H_grid_widths = widths(tau_H_grid)    
        
            # Convert to the correct units (micro Hertz factor)
            tau_BCZ_grid = tau_BCZ_grid * 1e-6
            tau_HeII_grid = tau_HeII_grid * 1e-6
            tau_H_grid = tau_H_grid * 1e-6
        
    #        # Initial guesses for the fits (Move the grid around the current initial guess)
    #        tau_BCZ_grid += params_smooth_HeII_BCZ_H[8]
    #        tau_HeII_grid += params_smooth_HeII_BCZ_H[5]
    #        tau_H_grid += params_smooth_HeII_BCZ_H[12]
            
            # Make sure not to have negative tau
            tau_BCZ_grid = tau_BCZ_grid[ np.where(tau_BCZ_grid > 0) ]
            tau_HeII_grid = tau_HeII_grid[ np.where(tau_HeII_grid > 0) ]
            tau_H_grid = tau_H_grid[ np.where(tau_H_grid > 0) ]
    
            def grid_tau(i_tau_BCZ,i_tau_HeII,i_tau_H,rest):
                              
                params_smooth_HeII_BCZ_H = rest[0]
                limits = rest[1]
                all_nu_secdiff_filtered = rest[2]
                all_secdiff_filtered = rest[3]
                cov_secdiff = rest[4]
                nu_secdiff_filtered = rest[5]
                secdiff_filtered = rest[6]
                all_secdiff_filtered_sorted = rest[7]
                all_nu_secdiff_filtered_sorted = rest[8]
                IMPOSE_TAU_HEII_TO_BE_WITHIN_THE_SEARCHING_RANGE = rest[9]
                
                initial_guess_grid = params_smooth_HeII_BCZ_H.copy()
                # update value
                initial_guess_grid[5] = i_tau_HeII
                initial_guess_grid[8] = i_tau_BCZ
                initial_guess_grid[12] = i_tau_H
                #print(i_tau_H)
                
                # Correct limits 
                limits_grid = limits.copy()
                # curve_fit does not allow the limits to equal the initial guesses
                limits_grid[0][5] -= 0.01*(limits_grid[0][5])
                limits_grid[1][5] += 0.01*(limits_grid[1][5])
                limits_grid[0][8] -= 0.01*(limits_grid[0][8])
                limits_grid[1][8] += 0.01*(limits_grid[1][8])
                limits_grid[0][12] -= 0.01*(limits_grid[0][12])
                limits_grid[1][12] += 0.01*(limits_grid[1][12])
                # Move limits around the initial guess
    #            delta = np.abs( initial_guess_grid[5] - np.mean([limits_grid[0][5], limits_grid[1][5]]) )
    #            limits_grid[0][5]-= delta
    #            limits_grid[1][5] += delta
    #            delta = np.abs( initial_guess_grid[8] - np.mean([limits_grid[0][8], limits_grid[1][8]]) )
    #            limits_grid[0][8] -= delta
    #            limits_grid[1][8] += delta
    #            delta = np.abs( initial_guess_grid[12] - np.mean([limits_grid[0][12], limits_grid[1][12]]) )
    #            limits_grid[0][12]-= delta
    #            limits_grid[1][12]+= delta
                # Without limits 0..5000s
    #            delta = np.abs( initial_guess_grid[5] - np.mean([limits_grid[0][5], limits_grid[1][5]]) )
    #            limits_grid[0][5] = 0 # -= delta
    #            limits_grid[1][5] = 5000*1e-6 # += delta
    #            delta = np.abs( initial_guess_grid[8] - np.mean([limits_grid[0][8], limits_grid[1][8]]) )
    #            limits_grid[0][8] = 0 # -= delta
    #            limits_grid[1][8] = 5000*1e-6# += delta
    #            delta = np.abs( initial_guess_grid[12] - np.mean([limits_grid[0][12], limits_grid[1][12]]) )
    #            limits_grid[0][12] = 0 # -= delta
    #            limits_grid[1][12] = 5000*1e-6 # += delta
        
                # FITTING
                try:
                    
                    params_smooth_HeII_BCZ_H_grid, params_smooth_HeII_BCZ_H_grid_covariance = curve_fit(fit_smooth_HeII_BCZ_H, \
                                                                                                        all_nu_secdiff_filtered, all_secdiff_filtered, \
                                                                                                        p0=initial_guess_grid, \
                                                                                                        sigma=cov_secdiff, \
                                                                                                        bounds=limits_grid)
    #            except ValueError:
    #                embed()
                except RuntimeError:
                    print('\n\t'+'Optimal parameters not found: The maximum number of function evaluations is exceeded.')
                    print('tau_BCZ',i_tau_BCZ*1e6,
                          'tau_HeII',i_tau_HeII*1e6,
                          'i_tau_H',i_tau_H*1e6)
                    params_smooth_HeII_BCZ_H_grid=None
                    params_smooth_HeII_BCZ_H_grid_covariance=None
                    
                    # Save. Store the results of the fit
                    fit_smooth_HeII_BCZ_H_grid_dict = dict()
                    fit_smooth_HeII_BCZ_H_grid_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
                    fit_smooth_HeII_BCZ_H_grid_dict['secdiff_all_l'] = all_secdiff_filtered
                    fit_smooth_HeII_BCZ_H_grid_dict['init_guess'] = {'a0':initial_guess_grid[0],
                                                                     'a1':initial_guess_grid[1],
                                                                     'a2':initial_guess_grid[2],
                                                                     'c0':initial_guess_grid[3],
                                                                     'c2':initial_guess_grid[4],
                                                                     'tau_HeII':initial_guess_grid[5],
                                                                     'phi_HeII':initial_guess_grid[6],
                                                                     'b2':initial_guess_grid[7],
                                                                     'tau_BCZ':initial_guess_grid[8],
                                                                     'phi_BCZ':initial_guess_grid[9],
                                                                     'd0_H':initial_guess_grid[10],
                                                                     'd2_H':initial_guess_grid[11],
                                                                     'tau_H':initial_guess_grid[12],
                                                                     'phi_H':initial_guess_grid[13]}
                        
                    fit_smooth_HeII_BCZ_H_grid_dict['limits'] = {'a0':(limits_grid[0][0],limits_grid[1][0]),
                                                                 'a1':(limits_grid[0][1],limits_grid[1][1]),
                                                                 'a2':(limits_grid[0][2],limits_grid[1][2]),
                                                                 'c0':(limits_grid[0][3],limits_grid[1][3]),
                                                                 'c2':(limits_grid[0][4],limits_grid[1][4]),
                                                                 'tau_HeII':(limits_grid[0][5],limits_grid[1][5]),
                                                                 'phi_HeII':(limits_grid[0][6],limits_grid[1][6]),
                                                                 'b2':(limits_grid[0][7],limits_grid[1][7]),
                                                                 'tau_BCZ':(limits_grid[0][8],limits_grid[1][8]),
                                                                 'phi_BCZ':(limits_grid[0][9],limits_grid[1][9]),
                                                                 'd0':(limits_grid[0][10],limits_grid[1][10]),
                                                                 'd2':(limits_grid[0][11],limits_grid[1][11]),
                                                                 'tau_H':(limits_grid[0][12],limits_grid[1][12]),
                                                                 'phi_H':(limits_grid[0][13],limits_grid[1][13])}
                    
                    fit_smooth_HeII_BCZ_H_grid_dict['results'] = {'a0':None,
                                                                  'a1':None,
                                                                  'a2':None,
                                                                  'c0':None,
                                                                  'c2':None,
                                                                  'tau_HeII':None,
                                                                  'phi_HeII':None,
                                                                  'b2':None,
                                                                  'tau_BCZ':None,
                                                                  'phi_BCZ':None,
                                                                  'd0':None,
                                                                  'd2':None,
                                                                  'tau_H':None,
                                                                  'pi_H':None}
                    
                    
                    fit_smooth_HeII_BCZ_H_grid_dict['results_cov'] = None
    #                fit_smooth_HeII_BCZ_H_grid_dict['secdiff_cov'] = cov_secdiff
            
                    # Residuos
                    residuals_smooth_HeII_BCZ_H_grid_dict = dict()
                    for l in ell:
                        residuals_smooth_HeII_BCZ_H_grid_dict[l] = None
                    residuals_smooth_HeII_BCZ_H_grid_dict['all_l_ordered'] = None
                    # Chi square
                    residuals_smooth_HeII_BCZ_H_grid_dict['chi2'] = np.inf
                    residuals_smooth_HeII_BCZ_H_grid_dict['dof'] = None
                    residuals_smooth_HeII_BCZ_H_grid_dict['reduced_chi2'] = np.inf
                        
                    fit_smooth_HeII_BCZ_H_grid_dict['residuals'] = residuals_smooth_HeII_BCZ_H_grid_dict
                    
                    six_lists =  (params_smooth_HeII_BCZ_H_grid,
                                  params_smooth_HeII_BCZ_H_grid_covariance,
                                  residuals_smooth_HeII_BCZ_H_grid_dict,
                                  initial_guess_grid,
                                  limits_grid,
                                  fit_smooth_HeII_BCZ_H_grid_dict)
        
                    return six_lists
        
                # Residuos
                residuals_smooth_HeII_BCZ_H_grid_dict = dict()
                for l in ell:
                    residuals_smooth_HeII_BCZ_H_grid_dict[l] = secdiff_filtered[l] - fit_smooth_HeII_BCZ_H(nu_secdiff_filtered[l], *params_smooth_HeII_BCZ_H_grid)
                residuals_smooth_HeII_BCZ_H_grid_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered_sorted, *params_smooth_HeII_BCZ_H_grid)
                residuals_smooth_HeII_BCZ_H_grid_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered, *params_smooth_HeII_BCZ_H_grid)
                # Chi square
                #residuals_smooth_HeII_BCZ_H_grid_dict['chi2'] = np.sum(residuals_smooth_HeII_BCZ_H_grid_dict['all_l_ordered']**2)
                residuals_smooth_HeII_BCZ_H_grid_dict['chi2'] = residuals_smooth_HeII_BCZ_H_grid_dict['all_l'].T @ np.linalg.inv(cov_secdiff) @ residuals_smooth_HeII_BCZ_H_grid_dict['all_l']
                residuals_smooth_HeII_BCZ_H_grid_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(params_smooth_HeII_BCZ_H_grid)
                residuals_smooth_HeII_BCZ_H_grid_dict['reduced_chi2'] = residuals_smooth_HeII_BCZ_H_grid_dict['chi2'] / residuals_smooth_HeII_BCZ_H_grid_dict['dof']
            
                # Save. Store the results of the fit
                fit_smooth_HeII_BCZ_H_grid_dict = dict()
                fit_smooth_HeII_BCZ_H_grid_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
                fit_smooth_HeII_BCZ_H_grid_dict['secdiff_all_l'] = all_secdiff_filtered
                fit_smooth_HeII_BCZ_H_grid_dict['init_guess'] = {'a0':initial_guess_grid[0],
                                                                 'a1':initial_guess_grid[1],
                                                                 'a2':initial_guess_grid[2],
                                                                 'c0':initial_guess_grid[3],
                                                                 'c2':initial_guess_grid[4],
                                                                 'tau_HeII':initial_guess_grid[5],
                                                                 'phi_HeII':initial_guess_grid[6],
                                                                 'b2':initial_guess_grid[7],
                                                                 'tau_BCZ':initial_guess_grid[8],
                                                                 'phi_BCZ':initial_guess_grid[9],
                                                                 'd0_H':initial_guess_grid[10],
                                                                 'd2_H':initial_guess_grid[11],
                                                                 'tau_H':initial_guess_grid[12],
                                                                 'phi_H':initial_guess_grid[13]}
                    
                fit_smooth_HeII_BCZ_H_grid_dict['limits'] = {'a0':(limits_grid[0][0],limits_grid[1][0]),
                                                             'a1':(limits_grid[0][1],limits_grid[1][1]),
                                                             'a2':(limits_grid[0][2],limits_grid[1][2]),
                                                             'c0':(limits_grid[0][3],limits_grid[1][3]),
                                                             'c2':(limits_grid[0][4],limits_grid[1][4]),
                                                             'tau_HeII':(limits_grid[0][5],limits_grid[1][5]),
                                                             'phi_HeII':(limits_grid[0][6],limits_grid[1][6]),
                                                             'b2':(limits_grid[0][7],limits_grid[1][7]),
                                                             'tau_BCZ':(limits_grid[0][8],limits_grid[1][8]),
                                                             'phi_BCZ':(limits_grid[0][9],limits_grid[1][9]),
                                                             'd0':(limits_grid[0][10],limits_grid[1][10]),
                                                             'd2':(limits_grid[0][11],limits_grid[1][11]),
                                                             'tau_H':(limits_grid[0][12],limits_grid[1][12]),
                                                             'phi_H':(limits_grid[0][13],limits_grid[1][13])}
                
                fit_smooth_HeII_BCZ_H_grid_dict['results'] = {'a0':params_smooth_HeII_BCZ_H_grid[0],
                                                              'a1':params_smooth_HeII_BCZ_H_grid[1],
                                                              'a2':params_smooth_HeII_BCZ_H_grid[2],
                                                              'c0':params_smooth_HeII_BCZ_H_grid[3],
                                                              'c2':params_smooth_HeII_BCZ_H_grid[4],
                                                              'tau_HeII':params_smooth_HeII_BCZ_H_grid[5],
                                                              'phi_HeII':params_smooth_HeII_BCZ_H_grid[6],
                                                              'b2':params_smooth_HeII_BCZ_H_grid[7],
                                                              'tau_BCZ':params_smooth_HeII_BCZ_H_grid[8],
                                                              'phi_BCZ':params_smooth_HeII_BCZ_H_grid[9],
                                                              'd0':params_smooth_HeII_BCZ_H_grid[10],
                                                              'd2':params_smooth_HeII_BCZ_H_grid[11],
                                                              'tau_H':params_smooth_HeII_BCZ_H_grid[12],
                                                              'pi_H':params_smooth_HeII_BCZ_H_grid[13],}
                
                
                fit_smooth_HeII_BCZ_H_grid_dict['results_cov'] = params_smooth_HeII_BCZ_H_grid_covariance
    #            fit_smooth_HeII_BCZ_H_grid_dict['secdiff_cov'] = cov_secdiff
                fit_smooth_HeII_BCZ_H_grid_dict['residuals'] = residuals_smooth_HeII_BCZ_H_grid_dict
        
                six_lists =  (params_smooth_HeII_BCZ_H_grid,
                              params_smooth_HeII_BCZ_H_grid_covariance,
                              residuals_smooth_HeII_BCZ_H_grid_dict,
                              initial_guess_grid,
                              limits_grid,
                              fit_smooth_HeII_BCZ_H_grid_dict)
        
                return six_lists
        
        #    # Initialize variables to store results of each loop
        #    params_smooth_HeII_BCZ_H_grid_list = list()
        #    params_smooth_HeII_BCZ_H_grid_covariance_list = list()
        #    residuals_smooth_HeII_BCZ_H_grid_dict_list = list()
        #    initial_guess_grid_list = list()
        #    limits_grid_list = list()
        #    fit_smooth_HeII_BCZ_H_grid_dict_list = list()
            
        #    six_lists_list = list()
        
            tau_BCZ_HeII_H_grid = [ (i,j,k) for i in tau_BCZ_grid for j in tau_HeII_grid for k in tau_H_grid ]
                              
            rest = (params_smooth_HeII_BCZ_H,
                    limits,
                    all_nu_secdiff_filtered,
                    all_secdiff_filtered,
                    cov_secdiff,
                    nu_secdiff_filtered,
                    secdiff_filtered,
                    all_secdiff_filtered_sorted,
                    all_nu_secdiff_filtered_sorted,
                    IMPOSE_TAU_HEII_TO_BE_WITHIN_THE_SEARCHING_RANGE)
    
            
            ### Number of cores
            num_cores = 60
            six_lists_list = Parallel(n_jobs=num_cores)(delayed(grid_tau)(i_tau_BCZ,i_tau_HeII,i_tau_H,rest) for i_tau_BCZ,i_tau_HeII,i_tau_H in tau_BCZ_HeII_H_grid)
            
        #    for i_tau_BCZ,i_tau_HeII,i_tau_H in tau_BCZ_HeII_H_grid:
        #        six_lists_list.append( grid_tau(i_tau_BCZ,i_tau_HeII,i_tau_H) )
         
            six_lists_list = list(map(list, zip(*six_lists_list)))
            params_smooth_HeII_BCZ_H_grid_list = six_lists_list[0]
            params_smooth_HeII_BCZ_H_grid_covariance_list = six_lists_list[1]
            residuals_smooth_HeII_BCZ_H_grid_dict_list = six_lists_list[2]
            initial_guess_grid_list = six_lists_list[3]
            limits_grid_list = six_lists_list[4]
            fit_smooth_HeII_BCZ_H_grid_dict_list = six_lists_list[5]
            
               
            # Save all the results from the grid
            profile_results['fit_smooth_HeII_BCZ_H_grid_all'] = fit_smooth_HeII_BCZ_H_grid_dict_list    
            ###
        
            # Get the chi2 squared value of all the grid    
            chi2_grid = [ _['residuals']['chi2'] for _ in fit_smooth_HeII_BCZ_H_grid_dict_list ] 
            chi2_grid = np.array(chi2_grid)
            
            # find the lest chi2
            i_min_chi2 = np.argmin( [ residuals['chi2'] for residuals in residuals_smooth_HeII_BCZ_H_grid_dict_list ] )
    
            chi2_grid_argsort = np.argsort([ residuals['chi2'] for residuals in residuals_smooth_HeII_BCZ_H_grid_dict_list ])
            counter_i_chi2 = 0
    
            if IMPOSE_TAU_HEII_TO_BE_WITHIN_THE_SEARCHING_RANGE:
                # Here we are forcing the selected element from the grid to have ta_HeII within the limits defined in tau_HeII_lim    
                while not (tau_HeII_lim[0] <= fit_smooth_HeII_BCZ_H_grid_dict_list[i_min_chi2]['results']['tau_HeII'] and fit_smooth_HeII_BCZ_H_grid_dict_list[i_min_chi2]['results']['tau_HeII'] <= tau_HeII_lim[1]):
                    counter_i_chi2 = counter_i_chi2+1
                    i_min_chi2 = chi2_grid_argsort[counter_i_chi2]
    
            # Retrieve the values of the fit corresponging to the least chi squared value
            params_smooth_HeII_BCZ_H_grid = params_smooth_HeII_BCZ_H_grid_list[i_min_chi2]
            params_smooth_HeII_BCZ_H_grid_covariance = params_smooth_HeII_BCZ_H_grid_covariance_list[i_min_chi2]
            residuals_smooth_HeII_BCZ_H_grid_dict = residuals_smooth_HeII_BCZ_H_grid_dict_list[i_min_chi2]
            initial_guess = initial_guess_grid_list[i_min_chi2]
            limits = limits_grid_list[i_min_chi2]
        
            # Save. Store the results of the fit
            fit_smooth_HeII_BCZ_H_grid_dict = dict()
            fit_smooth_HeII_BCZ_H_grid_dict['tau_BCZ_grid'] = tau_BCZ_grid
            fit_smooth_HeII_BCZ_H_grid_dict['tau_HeII_grid'] = tau_HeII_grid
            fit_smooth_HeII_BCZ_H_grid_dict['tau_H_grid'] = tau_H_grid
            fit_smooth_HeII_BCZ_H_grid_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
            fit_smooth_HeII_BCZ_H_grid_dict['secdiff_all_l'] = all_secdiff_filtered
            fit_smooth_HeII_BCZ_H_grid_dict['init_guess'] = {'a0':initial_guess[0],
                                                             'a1':initial_guess[1],
                                                             'a2':initial_guess[2],
                                                             'c0':initial_guess[3],
                                                             'c2':initial_guess[4],
                                                             'tau_HeII':initial_guess[5],
                                                             'phi_HeII':initial_guess[6],
                                                             'b2':initial_guess[7],
                                                             'tau_BCZ':initial_guess[8],
                                                             'phi_BCZ':initial_guess[9],
                                                             'd0_H':initial_guess[10],
                                                             'd2_H':initial_guess[11],
                                                             'tau_H':initial_guess[12],
                                                             'phi_H':initial_guess[13]}
                
            fit_smooth_HeII_BCZ_H_grid_dict['limits'] = {'a0':(limits[0][0],limits[1][0]),
                                                         'a1':(limits[0][1],limits[1][1]),
                                                         'a2':(limits[0][2],limits[1][2]),
                                                         'c0':(limits[0][3],limits[1][3]),
                                                         'c2':(limits[0][4],limits[1][4]),
                                                         'tau_HeII':(limits[0][5],limits[1][5]),
                                                         'phi_HeII':(limits[0][6],limits[1][6]),
                                                         'b2':(limits[0][7],limits[1][7]),
                                                         'tau_BCZ':(limits[0][8],limits[1][8]),
                                                         'phi_BCZ':(limits[0][9],limits[1][9]),
                                                         'd0':(limits[0][10],limits[1][10]),
                                                         'd2':(limits[0][11],limits[1][11]),
                                                         'tau_H':(limits[0][12],limits[1][12]),
                                                         'phi_H':(limits[0][13],limits[1][13])}
            
            fit_smooth_HeII_BCZ_H_grid_dict['results'] = {'a0':params_smooth_HeII_BCZ_H_grid[0],
                                                          'a1':params_smooth_HeII_BCZ_H_grid[1],
                                                          'a2':params_smooth_HeII_BCZ_H_grid[2],
                                                          'c0':params_smooth_HeII_BCZ_H_grid[3],
                                                          'c2':params_smooth_HeII_BCZ_H_grid[4],
                                                          'tau_HeII':params_smooth_HeII_BCZ_H_grid[5],
                                                          'phi_HeII':params_smooth_HeII_BCZ_H_grid[6],
                                                          'b2':params_smooth_HeII_BCZ_H_grid[7],
                                                          'tau_BCZ':params_smooth_HeII_BCZ_H_grid[8],
                                                          'phi_BCZ':params_smooth_HeII_BCZ_H_grid[9],
                                                          'd0':params_smooth_HeII_BCZ_H_grid[10],
                                                          'd2':params_smooth_HeII_BCZ_H_grid[11],
                                                          'tau_H':params_smooth_HeII_BCZ_H_grid[12],
                                                          'pi_H':params_smooth_HeII_BCZ_H_grid[13],}
            
            
            fit_smooth_HeII_BCZ_H_grid_dict['results_cov'] = params_smooth_HeII_BCZ_H_grid_covariance
            fit_smooth_HeII_BCZ_H_grid_dict['secdiff_cov'] = cov_secdiff
            fit_smooth_HeII_BCZ_H_grid_dict['residuals'] = residuals_smooth_HeII_BCZ_H_grid_dict
        
            profile_results['fit_smooth_HeII_BCZ_H_grid'] = fit_smooth_HeII_BCZ_H_grid_dict 
            
    #        # =============================================================================
    #        ### Repeat
    #        # =============================================================================
    #        
    #        n_repeat = 5
    #        
    #        for i in range(n_repeat):     
    #            if i == 0:
    #                initial_guess_repeat = params_smooth_HeII_BCZ_H_grid.copy()
    #                limits_repeat = limits.copy()
    #            else:
    #                initial_guess_repeat = params_smooth_HeII_BCZ_H_repeat.copy()
    #                # Correct limits               
    #                delta = np.abs( initial_guess_repeat[5] - np.mean([limits_repeat[0][5], limits_repeat[1][5]]) )
    #                limits_repeat[0][5] -= delta
    #                limits_repeat[1][5] += delta
    #                delta = np.abs( initial_guess_repeat[8] - np.mean([limits_repeat[0][8], limits_repeat[1][8]]) )
    #                limits_repeat[0][8] -= delta
    #                limits_repeat[1][8] += delta
    #                delta = np.abs( initial_guess_repeat[12] - np.mean([limits_repeat[0][12], limits_repeat[1][12]]) )
    #                limits_repeat[0][12] -= delta
    #                limits_repeat[1][12] += delta
    #    
    #            # FITTING
    #            params_smooth_HeII_BCZ_H_repeat, params_smooth_HeII_BCZ_H_repeat_covariance = curve_fit(fit_smooth_HeII_BCZ_H, \
    #                                                                                                    all_nu_secdiff_filtered, all_secdiff_filtered, \
    #                                                                                                    p0=initial_guess_repeat, \
    #                                                                                                    sigma=cov_secdiff, \
    #                                                                                                    bounds=limits_repeat)
    #        # Residuos
    #        residuals_smooth_HeII_BCZ_H_repeat_dict = dict()
    #        for l in ell:
    #            residuals_smooth_HeII_BCZ_H_repeat_dict[l] = secdiff_filtered[l] - fit_smooth_HeII_BCZ_H(nu_secdiff_filtered[l], *params_smooth_HeII_BCZ_H)
    #        residuals_smooth_HeII_BCZ_H_repeat_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered_sorted, *params_smooth_HeII_BCZ_H)
    #        residuals_smooth_HeII_BCZ_H_repeat_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered, *params_smooth_HeII_BCZ_H)
    #        # Chi square
    #        #residuals_smooth_HeII_BCZ_H_repeat_dict['chi2'] = np.sum(residuals_smooth_HeII_BCZ_H_repeat_dict['all_l_ordered']**2)
    #        residuals_smooth_HeII_BCZ_H_repeat_dict['chi2'] = residuals_smooth_HeII_BCZ_H_repeat_dict['all_l'].T @ np.linalg.inv(cov_secdiff) @ residuals_smooth_HeII_BCZ_H_repeat_dict['all_l']
    #        residuals_smooth_HeII_BCZ_H_repeat_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(params_smooth_HeII_BCZ_H)
    #        residuals_smooth_HeII_BCZ_H_repeat_dict['reduced_chi2'] = residuals_smooth_HeII_BCZ_H_repeat_dict['chi2'] / residuals_smooth_HeII_BCZ_H_repeat_dict['dof']
    #        
    #        # Save. Store the results of the fit
    #        fit_smooth_HeII_BCZ_H_repeat_dict = dict()
    #        fit_smooth_HeII_BCZ_H_repeat_dict['n_repeat'] = n_repeat
    #        fit_smooth_HeII_BCZ_H_repeat_dict['nu_secdiff_all_l'] = all_nu_secdiff_filtered
    #        fit_smooth_HeII_BCZ_H_repeat_dict['secdiff_all_l'] = all_secdiff_filtered
    #        fit_smooth_HeII_BCZ_H_repeat_dict['init_guess'] = {'a0':initial_guess_repeat[0],
    #                                                           'a1':initial_guess_repeat[1],
    #                                                           'a2':initial_guess_repeat[2],
    #                                                           'c0':initial_guess_repeat[3],
    #                                                           'c2':initial_guess_repeat[4],
    #                                                           'tau_HeII':initial_guess_repeat[5],
    #                                                           'phi_HeII':initial_guess_repeat[6],
    #                                                           'b2':initial_guess_repeat[7],
    #                                                           'tau_BCZ':initial_guess_repeat[8],
    #                                                           'phi_BCZ':initial_guess_repeat[9],
    #                                                           'd0_H':initial_guess_repeat[10],
    #                                                           'd2_H':initial_guess_repeat[11],
    #                                                           'tau_H':initial_guess_repeat[12],
    #                                                           'phi_H':initial_guess_repeat[13]}
    #            
    #        fit_smooth_HeII_BCZ_H_repeat_dict['limits'] = {'a0':(limits_repeat[0][0],limits_repeat[1][0]),
    #                                                       'a1':(limits_repeat[0][1],limits_repeat[1][1]),
    #                                                       'a2':(limits_repeat[0][2],limits_repeat[1][2]),
    #                                                       'c0':(limits_repeat[0][3],limits_repeat[1][3]),
    #                                                       'c2':(limits_repeat[0][4],limits_repeat[1][4]),
    #                                                       'tau_HeII':(limits_repeat[0][5],limits_repeat[1][5]),
    #                                                       'phi_HeII':(limits_repeat[0][6],limits_repeat[1][6]),
    #                                                       'b2':(limits_repeat[0][7],limits_repeat[1][7]),
    #                                                       'tau_BCZ':(limits_repeat[0][8],limits_repeat[1][8]),
    #                                                       'phi_BCZ':(limits_repeat[0][9],limits_repeat[1][9]),
    #                                                       'd0':(limits_repeat[0][10],limits_repeat[1][10]),
    #                                                       'd2':(limits_repeat[0][11],limits_repeat[1][11]),
    #                                                       'tau_H':(limits_repeat[0][12],limits_repeat[1][12]),
    #                                                       'phi_H':(limits_repeat[0][13],limits_repeat[1][13])}
    #        
    #        fit_smooth_HeII_BCZ_H_repeat_dict['results'] = {'a0':params_smooth_HeII_BCZ_H_repeat[0],
    #                                                        'a1':params_smooth_HeII_BCZ_H_repeat[1],
    #                                                        'a2':params_smooth_HeII_BCZ_H_repeat[2],
    #                                                        'c0':params_smooth_HeII_BCZ_H_repeat[3],
    #                                                        'c2':params_smooth_HeII_BCZ_H_repeat[4],
    #                                                        'tau_HeII':params_smooth_HeII_BCZ_H_repeat[5],
    #                                                        'phi_HeII':params_smooth_HeII_BCZ_H_repeat[6],
    #                                                        'b2':params_smooth_HeII_BCZ_H_repeat[7],
    #                                                        'tau_BCZ':params_smooth_HeII_BCZ_H_repeat[8],
    #                                                        'phi_BCZ':params_smooth_HeII_BCZ_H_repeat[9],
    #                                                        'd0':params_smooth_HeII_BCZ_H_repeat[10],
    #                                                        'd2':params_smooth_HeII_BCZ_H_repeat[11],
    #                                                        'tau_H':params_smooth_HeII_BCZ_H_repeat[12],
    #                                                        'pi_H':params_smooth_HeII_BCZ_H_repeat[13],}
    #        
    #        
    #        fit_smooth_HeII_BCZ_H_repeat_dict['results_cov'] = params_smooth_HeII_BCZ_H_repeat_covariance
    #        fit_smooth_HeII_BCZ_H_repeat_dict['secdiff_cov'] = cov_secdiff
    #        fit_smooth_HeII_BCZ_H_repeat_dict['residuals'] = residuals_smooth_HeII_BCZ_H_repeat_dict
    #    
    #        profile_results['fit_smooth_HeII_BCZ_H_repeat'] = fit_smooth_HeII_BCZ_H_repeat_dict 
        
            # ==========================================================================
            ### Let's check if any of the residuals in grid results is greater than residual_threshold    
            # ==========================================================================
        
            residual_threshold = 0.5
            
            ind_filtered_out2 = dict()    
            if np.all(np.abs(profile_results['fit_smooth_HeII_BCZ_H_grid']['residuals']['all_l_ordered']) < residual_threshold):
                # Escape while-loop
                outliers = False
            else:
                if while_counter >= while_max_loop:
                    break
                print('WHILE NUMBER', while_counter)
                if FILE_NOTICE:
                    file_notice = open(file_notice_name,'a')
                    file_notice.write('WHILE NUMBER '+str(while_counter)+', realization: '+str(i_statistical_error_realization)+'\n')
                    file_notice.close()
                # Filter outliers
                for l in ell:
                    residuals_tmp = profile_results['fit_smooth_HeII_BCZ_H_grid']['residuals'][l].copy()
                    condition = np.abs(residuals_tmp) < residual_threshold
                    # update
                    secdiff_filtered[l] = secdiff_filtered[l][condition]
                    nu_secdiff_filtered[l] = nu_secdiff_filtered[l][condition]
                    # place back the elements that we filter out at the begining
                    for i in np.sort(ind_filtered_out[l]):
                        residuals_tmp = np.insert(residuals_tmp, i, 2*residual_threshold)
                    condition = np.abs(residuals_tmp) > residual_threshold
                    # update the filter
                    ind_filtered_out[l] = np.argwhere(condition).reshape(-1)
                # Concatenate the second differences. Get the covariance matrix for the future fit  
                # Note that they are not ordered      
                all_nu_secdiff_filtered, all_secdiff_filtered, cov_secdiff = concat_secdiff(nu_secdiff_filtered, secdiff_filtered,
                                                                                            nu_secdiff, secdiff, ind_filtered_out)


        # ==========================================================================
        ### result for the current realization
        # ==========================================================================

        # ==========================================================================
        ### Some plots
        # ==========================================================================


        # ==========================================================================
        ### Plot histograms from the grid for the current realization
        # ==========================================================================
 
        # Plot style       
        plt.style.use('stefano')
        
        fig_hist=plt.figure()
        gs=gridspec.GridSpec(2,2)
        
        # Get the chi2 squared value of all the grid    
        chi2_grid = [ _['residuals']['chi2'] for _ in profile_results['fit_smooth_HeII_BCZ_H_grid_all'] ] 
        chi2_grid = np.array(chi2_grid)
        
        ### Tau_BCZ
        tau_BCZ_results_grid = [ _['results']['tau_BCZ'] for _ in profile_results['fit_smooth_HeII_BCZ_H_grid_all'] ]
        tau_BCZ_results_grid = np.array(tau_BCZ_results_grid)
        # Exclude where the fit did not converge
        ind_converge = tau_BCZ_results_grid != None
        # Generate the bins based on the tau results (not the tau initial guess or grid)
        if i_statistical_error_realization == 0:
            hist, bins = np.histogram(tau_BCZ_results_grid[ind_converge], bins='auto')
            bins_template_BCZ = bins
        else:
            hist, bins = np.histogram(tau_BCZ_results_grid[ind_converge], bins=bins_template_BCZ)
#            hist, bins = np.histogram(tau_BCZ_results_grid[ind_converge], bins='auto')
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
        plt.setp(ax_hist.get_xticklabels(),fontsize=7)
        # No y-axis ticks
        plt.setp(ax_hist.get_yticklabels(), visible=False)
        # Plot grid
        ymin, ymax = plt.ylim()
        ypos = (ymax-ymin)*0.9
        grid = profile_results['fit_smooth_HeII_BCZ_H_grid']['tau_BCZ_grid']
        ax_hist.plot(grid*1e6,np.repeat(ypos,grid.size), color='orange', marker='|', linestyle='None', markeredgewidth=1.5, markersize=20)
        # Store
        hist_BCZ = {'bins':bins,
                    'pdf':tau_BCZ_results_grid_bin_pdf}
        
        ### Tau_HeII
        tau_HeII_results_grid = [ _['results']['tau_HeII'] for _ in profile_results['fit_smooth_HeII_BCZ_H_grid_all'] ]
        tau_HeII_results_grid = np.array(tau_HeII_results_grid)
        # Exclude where the fit did not converge
        ind_converge = tau_HeII_results_grid != None
        # Generate the bins based on the tau results (not the tau initial guess or grid)
        if i_statistical_error_realization == 0:
            hist, bins = np.histogram(tau_HeII_results_grid[ind_converge], bins='auto')
            bins_template_HeII = bins
        else:
            hist, bins = np.histogram(tau_HeII_results_grid[ind_converge], bins=bins_template_HeII)
#            hist, bins = np.histogram(tau_HeII_results_grid[ind_converge], bins='auto')            
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
        plt.setp(ax_hist.get_xticklabels(),fontsize=7)
        # No y-axis ticks
        plt.setp(ax_hist.get_yticklabels(), visible=False)
        # Plot grid
        ymin, ymax = plt.ylim()
        ypos = (ymax-ymin)*0.9
        grid = profile_results['fit_smooth_HeII_BCZ_H_grid']['tau_HeII_grid']
        ax_hist.plot(grid*1e6,np.repeat(ypos,grid.size), color='orange', marker='|', linestyle='None', markeredgewidth=1.5, markersize=20)
        # Store
        hist_HeII = {'bins':bins,
                     'pdf':tau_HeII_results_grid_bin_pdf}    
        
        ### Tau_H
        tau_H_results_grid = [ _['results']['tau_H'] for _ in profile_results['fit_smooth_HeII_BCZ_H_grid_all'] ]
        tau_H_results_grid = np.array(tau_H_results_grid)
        # Exclude where the fit did not converge
        ind_converge = tau_H_results_grid != None
        # Generate the bins based on the tau results (not the tau initial guess or grid)
        if i_statistical_error_realization == 0:
            hist, bins = np.histogram(tau_H_results_grid[ind_converge], bins='auto')
            bins_template_H = bins
        else:
            hist, bins = np.histogram(tau_H_results_grid[ind_converge], bins=bins_template_H)
#            hist, bins = np.histogram(tau_H_results_grid[ind_converge], bins='auto')
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
        ax_hist.set_xlabel('Tau H and HeI')    
        plt.setp(ax_hist.get_xticklabels(),fontsize=7)
        # No y-axis ticks
        plt.setp(ax_hist.get_yticklabels(), visible=False)
        # Plot grid
        ymin, ymax = plt.ylim()
        ypos = (ymax-ymin)*0.9
        grid = profile_results['fit_smooth_HeII_BCZ_H_grid']['tau_H_grid']
        ax_hist.plot(grid*1e6,np.repeat(ypos,grid.size), color='orange', marker='|', linestyle='None', markeredgewidth=1.5, markersize=20)
        # Store
        hist_H = {'bins':bins,
                  'pdf':tau_H_results_grid_bin_pdf}   
    
        # Title
        fig_hist.suptitle('Age after ZAMS: {:.3f} Gyr, {}, realization {}'.format(profile_results['age'], profile, i_statistical_error_realization))
        fig_hist.subplots_adjust(hspace=0.4)        
        pdf_hist.savefig(fig_hist)
        plt.close(fig_hist)
        
        
    
        results_realization_temp = dict()
        # summarize the results
        filter_in_keys = ['l', 'age', 'secdiff', 'nu_secdiff', 
                          'secdiff_filtered', 'nu_secdiff_filtered',
                          'secdiff_no_statistical_errors', 'nu_secdiff_no_statistical_errors',
                          'fit_smooth_HeII_BCZ_H_grid']
        results_realization_temp = { key:profile_results[key] for key in filter_in_keys }
        results_realization_temp['hist'] = {'BCZ':hist_BCZ,
                                            'HeII':hist_HeII,
                                            'H':hist_H}
        
        results_realizations[i_statistical_error_realization] = results_realization_temp
    
        # =============================================================================
        ### Plot second differences        
        # =============================================================================
 
        # Plot style       
        plt.style.use('stefano')

        colors = {0:'k', 1:'r', 2:'dodgerblue', 3:'lime'}
        markers = {0:'o', 1:'^', 2:'s', 3:'D'}

        params_temp = list(profile_results['fit_smooth_HeII_BCZ_H_grid']['results'].values()) 
 
       # We initialize the figure
        fig_secdiff = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
        ax_secdiff = plt.subplot(gs[0])
        ax_residual = plt.subplot(gs[1], sharex=ax_secdiff)
        
        # Plot the second differences
        # Find the minimum and maximum nu to plot
        for i,l in enumerate(ell):
            # initial value
            if i == 0:
                nu_min = np.min(nu_secdiff[l])
                nu_max = np.max(nu_secdiff[l])
            # Update velue
            else:
                if np.min(nu_secdiff[l]) < nu_min:
                    nu_min = np.min(nu_secdiff[l])
                if np.max(nu_secdiff[l]) > nu_max:
                    nu_max = np.max(nu_secdiff[l])                    
        # Plot the fit 
        nu_fit = np.linspace( nu_min, nu_max, 1000)  
        ax_secdiff.plot(nu_fit, fit_smooth_HeII_BCZ_H(nu_fit, *params_temp), linestyle='solid', color='orange', marker='None', label='fit')
        ax_secdiff.set(ylabel=r'Second Diffecences $\delta_2\nu$ ( $\mu$Hz )', title='Age after ZAMS: {:.3f} Gyr'.format(age))
        for l in ell:
            ax_secdiff.plot(nu_secdiff_filtered[l], secdiff_filtered[l],
                    color=colors[l], marker='.',
                    linestyle='none', markersize=6,
                    label=r'$\ell={}$'.format(l),
                    markeredgecolor='None')
        for l in ell:
            ax_secdiff.plot(nu_secdiff_no_statistical_errors[l], secdiff_no_statistical_errors[l],
                    color=colors[l], marker='_',
                    linestyle='none', markersize=6)  
            
        ax_secdiff.legend()
        
        # Residuos
        # Plot the data minus the model (residuals)
        ax_residual.set_ylabel=('Residual')
        ax_residual.set_xlabel=(r'Frequency ( $\mu$Hz )')
        for l in ell:
            ax_residual.plot(nu_secdiff_filtered[l], profile_results['fit_smooth_HeII_BCZ_H_grid']['residuals'][l],
                             color=colors[l], marker=markers[l],
                             linestyle='none', markersize=3,
                             markeredgecolor='None')
        plt.setp(ax_residual.get_yticklabels(),fontsize=7)
        # Horizontal 0-line
        ax_residual.axhline(0,linestyle='dotted', color='k')
        # No x-axis ticks
        plt.setp(ax_secdiff.get_xticklabels(), visible=False)
        # remove last and first tick label for the second subplot
        yticks_fit = ax_secdiff.yaxis.get_major_ticks()
        yticks_fit[-1].label1.set_visible(False)
        yticks_fit[0].label1.set_visible(False)
        # Creates free space outside the plot for the parameters display
        fig_secdiff.subplots_adjust(hspace=0) 
            
        # Save in a page of the PDF
        pdf_secdiff.savefig(fig_secdiff) 
        # Close the figure
        plt.close(fig_secdiff)
    
    pdf_hist.close()
    pdf_secdiff.close()

    results[profile] = results_realizations
    
# File where to save the results
savefile = open('results.bin','wb')
pickle.dump(results, savefile)
savefile.close()

time2 = time.time()
print('Fits took {:.3} min'.format((time2-time1)/60))
if FILE_NOTICE:
    file_notice = open(file_notice_name,'a')
    file_notice.write('Fits took {:.3} min\n'.format((time2-time1)/60))
    file_notice.close()


# =============================================================================
### Plots 
# =============================================================================

if MAKE_PLOTS:
    
    time1 = time.time()
    
    print('Creating plots')
    if FILE_NOTICE:
        file_notice = open(file_notice_name,'a')
        file_notice.write('Creating plots\n')
        file_notice.close()
        
    # Plot style       
    plt.style.use('stefano')
    
    # Create PDF for the second differences
    if PLOT_SECDIFF: pdf_secdiff = PdfPages('second_differences.pdf')
    # Create PDF for the smooth fit
    if PLOT_SMOOTH_FIT: pdf_smooth_fit = PdfPages('smooth_fit.pdf')
    # Create PDF for the smooth and HeII fit
    pdf_smooth_HeII_fit = PdfPages('smooth_HeII_fit.pdf')
    # Create PDF for the smooth, HeII and BCZ fit
    pdf_smooth_HeII_BCZ_fit = PdfPages('smooth_HeII_BCZ_fit.pdf')
    # Create PDF for the BCZ fit
    pdf_BCZ_fit = PdfPages('BCZ_fit.pdf')
    # Create PDF for the smooth, HeII, BCZ and H fit
    pdf_smooth_HeII_BCZ_H_fit = PdfPages('smooth_HeII_BCZ_H_fit.pdf')
    # Create PDF for the smooth, HeII, BCZ and H fit and the shange the fitted
    # acoustic depths over a grid
    pdf_smooth_HeII_BCZ_H_grid_fit = PdfPages('smooth_HeII_BCZ_H_grid_fit.pdf')
    # plot all the model used in the grid
    if PLOT_ALL_GRID: pdf_smooth_HeII_BCZ_H_grid_all_fit = PdfPages('smooth_HeII_BCZ_H_grid_all_fit.pdf')
    # plot histogram of the initial guesses of the grid
    pdf_smooth_HeII_BCZ_H_grid_all_fit_hist_initial_guesses = PdfPages('smooth_HeII_BCZ_H_grid_all_fit_histogram_initial_guesses.pdf')
    # plot histogram of the results if the grid
    pdf_smooth_HeII_BCZ_H_grid_all_fit_hist_results = PdfPages('smooth_HeII_BCZ_H_grid_all_fit_histogram_results.pdf')
    # plot the fit using the results from the best fit of the grid as initial guesses and repeat in a loop
    pdf_smooth_HeII_BCZ_H_repeated = PdfPages('smooth_HeII_BCZ_H_repeated.pdf')
    # plot the smooth component and the oscillation components of the best fit
    pdf_smooth_HeII_BCZ_H_grid_components1 = PdfPages('smooth_HeII_BCZ_H_grid_components1.pdf')
    # plot the oscillation components of the  best fit
    pdf_smooth_HeII_BCZ_H_grid_components2 = PdfPages('smooth_HeII_BCZ_H_grid_components2.pdf')
    # plot the oscillation components of the best fit
    pdf_smooth_HeII_BCZ_H_grid_components3 = PdfPages('smooth_HeII_BCZ_H_grid_components3.pdf')
    # plot the oscillation components of the best fit
    pdf_smooth_HeII_BCZ_H_grid_components4 = PdfPages('smooth_HeII_BCZ_H_grid_components4.pdf')
    # plot the oscillation components of the best fit
    pdf_smooth_HeII_BCZ_H_grid_components_all = PdfPages('smooth_HeII_BCZ_H_grid_components_all.pdf')

    # =============================================================================
    ### Plot sec diff raw
    # =============================================================================
    
    for profile,fits in results.items():
    
        if not profile in profile_subset: continue
    
        # ==========================================================================
        ### Plot second differences
        # ==========================================================================
        if PLOT_SECDIFF:
            # We initialize the figure
            fig_secdiff = plt.figure()
            # Plot the second differences
            plot_secdiff(fig_secdiff,
                         fits['nu_secdiff'],fits['secdiff'],
                         profile, fits['age'],
                         only_secdiff=True,
                         fit=False)
            # Save in a page of the PDF
            pdf_secdiff.savefig(fig_secdiff) 
            # Close the figure
            plt.close(fig_secdiff)
    
        # ==========================================================================
        ### Plot smooth fit to second diferences
        # ==========================================================================
        fig_smooth = plt.figure()
        ax_smooth_fit, ax_smooth_residual = plot_secdiff(fig_smooth,
                                                         fits['nu_secdiff_filtered'],
                                                         fits['secdiff_filtered'],
                                                         profile,
                                                         fits['age'],
                                                         only_secdiff=False,
                                                         fit=True,
                                                         f=fit_smooth,
                                                         params=list(fits['fit_smooth']['results'].values()),
                                                         residuals=fits['fit_smooth']['residuals'])  
       
        # Plot the span with little oscillations
        ypos = (ax_smooth_fit.get_ylim()[1] - ax_smooth_fit.get_ylim()[0])*0.1 + ax_smooth_fit.get_ylim()[0]
        ax_smooth_fit.plot(fits['fit_smooth']['nu_low_std'],
                           np.repeat(ypos,len(fits['fit_smooth']['nu_low_std'])),
                           marker='+', color='k')
        
        # Save in a page of the PDF
        pdf_smooth_fit.savefig(fig_smooth) 
        # Close the figure
        plt.close(fig_smooth)
        
        # ==========================================================================
        ### Plot smooth and HeII fit to second diferences
        # ==========================================================================
        fig_smooth_HeII = plt.figure()
        ax_smooth_HeII_fit, ax_smooth_HeII_residual = plot_secdiff(fig_smooth_HeII,
                                                                   fits['nu_secdiff_filtered'], 
                                                                   fits['secdiff_filtered'],
                                                                   profile,
                                                                   fits['age'],
                                                                   only_secdiff=False,
                                                                   fit=True,
                                                                   f=fit_smooth_HeII,
                                                                   params=list(fits['fit_smooth_HeII']['results'].values()), 
                                                                   residuals=fits['fit_smooth_HeII']['residuals'])  
        
        # Plot the two peaks on which the tau_HeII_init_guess is based on
        ax_smooth_HeII_fit.axvline(fits['fit_smooth_HeII']['nu_peak1'],linestyle='dashed',color='k')
        ax_smooth_HeII_fit.axvline(fits['fit_smooth_HeII']['nu_peak2'],linestyle='dashed',color='k')

        # Plot the uncertanty in Delta nu peaks
        mid_point = np.mean(plt.xlim())
        ypos3 = (ax_smooth_HeII_fit.get_ylim()[1] - ax_smooth_HeII_fit.get_ylim()[0])*0.3 + ax_smooth_HeII_fit.get_ylim()[0]
        ypos2 = (ax_smooth_HeII_fit.get_ylim()[1] - ax_smooth_HeII_fit.get_ylim()[0])*0.2 + ax_smooth_HeII_fit.get_ylim()[0]
        ypos1 = (ax_smooth_HeII_fit.get_ylim()[1] - ax_smooth_HeII_fit.get_ylim()[0])*0.1 + ax_smooth_HeII_fit.get_ylim()[0]
        small = fits['fit_smooth_HeII']['under_estimation_Delta_nu_peak']
        estimate = fits['fit_smooth_HeII']['Delta_nu_peak']
        big = fits['fit_smooth_HeII']['over_estimation_Delta_nu_peak']
        ax_smooth_HeII_fit.plot([0,small]+mid_point, [ypos3,ypos3],
                                linestyle='solid', color='k', linewidth=2)
        ax_smooth_HeII_fit.text(mid_point, ypos3, r'Lower limit: {:.1f} $\mu$Hz'.format(small),
                                horizontalalignment='left', verticalalignment='bottom')
        ax_smooth_HeII_fit.plot([0,estimate]+mid_point, [ypos2,ypos2],
                                linestyle='solid', color='k', linewidth=2)
        ax_smooth_HeII_fit.text(mid_point, ypos2, r'Estimation: {:.1f} $\mu$Hz'.format(estimate),
                                horizontalalignment='left', verticalalignment='bottom')
        ax_smooth_HeII_fit.plot([0,big]+mid_point, [ypos1,ypos1],
                                linestyle='solid', color='k', linewidth=2)
        ax_smooth_HeII_fit.text(mid_point, ypos1, r'Upper limit: {:.1f} $\mu$Hz'.format(big),
                                horizontalalignment='left', verticalalignment='bottom')

#        mid_point = np.mean([fits['fit_smooth_HeII']['nu_peak1'],
#                             fits['fit_smooth_HeII']['nu_peak2']])
#        lower_point = mid_point - fits['fit_smooth_HeII']['over_estimation_Delta_nu_peak']/2
#        upper_point = mid_point - fits['fit_smooth_HeII']['under_estimation_Delta_nu_peak']/2
#        ax_smooth_HeII_fit.axvspan(lower_point, upper_point, alpha=0.20, color='k')
#        lower_point = mid_point + fits['fit_smooth_HeII']['under_estimation_Delta_nu_peak']/2
#        upper_point = mid_point + fits['fit_smooth_HeII']['over_estimation_Delta_nu_peak']/2
#        ax_smooth_HeII_fit.axvspan(lower_point, upper_point, alpha=0.20, color='k')
#        ypos1 = (ax_smooth_HeII_fit.get_ylim()[1] - ax_smooth_HeII_fit.get_ylim()[0])*0.1 + ax_smooth_HeII_fit.get_ylim()[0]
#        ypos2 = (ax_smooth_HeII_fit.get_ylim()[1] - ax_smooth_HeII_fit.get_ylim()[0])*0.15 + ax_smooth_HeII_fit.get_ylim()[0]
#        # Delta bigger
#        delta_nu_peak_He_bigger = 1/(2*tau_HeII_lower_lim)
#        # Delta smaller
#        delta_nu_peak_He_smaller = 1/(2*tau_HeII_upper_lim)
#        ax_smooth_HeII_fit.axvspan(mid_point-fits['fit_smooth_HeII']['under_estimation_Delta_nu_peak']/2, \
#                                   tau_HeII-offset_tau + np.abs(tau_HeII_std), \
#                                   alpha=0.25, color='k')
#        ax_smooth_HeII_fit.errorbar(mid_point,ypos2, xerr=[fits['fit_smooth_HeII']['under_estimation_Delta_nu_peak']/2], 
#            fmt='none', color='k', capsize = 0)
#        ax_smooth_HeII_fit.errorbar(mid_point,ypos1, xerr=[fits['fit_smooth_HeII']['over_estimation_Delta_nu_peak']/2], 
#            fmt='none', color='k', capsize = 0)
#        embed()
        # Save in a page of the PDF
        pdf_smooth_HeII_fit.savefig(fig_smooth_HeII) 
        # Close the figure
        plt.close(fig_smooth_HeII)    
    
        # ==========================================================================
        ### Plot smooth, HeII and BCZ fit to second diferences
        # ==========================================================================
        fig_smooth_HeII_BCZ = plt.figure()
        ax_smooth_HeII_BCZ_fit, ax_smooth_HeII_BCZ_residual = plot_secdiff(fig_smooth_HeII_BCZ,
                                                                           fits['nu_secdiff_filtered'], 
                                                                           fits['secdiff_filtered'],
                                                                           profile,
                                                                           fits['age'],
                                                                           only_secdiff=False,
                                                                           fit=True,
                                                                           f=fit_smooth_HeII_BCZ,
                                                                           params=list(fits['fit_smooth_HeII_BCZ']['results'].values()),
                                                                           residuals=fits['fit_smooth_HeII_BCZ']['residuals'])  
    
        # Plot the peaks on which the tau_BCZ_init_guess is based on
        for inu in fits['fit_smooth_HeII_BCZ']['nu_peaks']:
            ax_smooth_HeII_BCZ_fit.axvline(inu,linestyle='dashed',color='k')
        #Plot the uncertanty in Delta nu peaks
        mid_point = np.mean(plt.xlim())
        ypos1 = (ax_smooth_HeII_BCZ_fit.get_ylim()[1] - ax_smooth_HeII_BCZ_fit.get_ylim()[0])*0.1 + ax_smooth_HeII_BCZ_fit.get_ylim()[0]
        ypos2 = (ax_smooth_HeII_BCZ_fit.get_ylim()[1] - ax_smooth_HeII_BCZ_fit.get_ylim()[0])*0.2 + ax_smooth_HeII_BCZ_fit.get_ylim()[0]
        ypos3 = (ax_smooth_HeII_BCZ_fit.get_ylim()[1] - ax_smooth_HeII_BCZ_fit.get_ylim()[0])*0.3 + ax_smooth_HeII_BCZ_fit.get_ylim()[0]
        small = fits['fit_smooth_HeII_BCZ']['under_estimation_Delta_nu_peak']
        estimate = fits['fit_smooth_HeII_BCZ']['Delta_nu_peak']
        big = fits['fit_smooth_HeII_BCZ']['over_estimation_Delta_nu_peak']
        ax_smooth_HeII_BCZ_fit.plot([0,small]+mid_point, [ypos3,ypos3],
                                    linestyle='solid', color='k', linewidth=2)
        ax_smooth_HeII_BCZ_fit.text(mid_point, ypos3, r'Lower limit: {:.1f} $\mu$Hz'.format(small),
                                    horizontalalignment='left', verticalalignment='bottom')
        ax_smooth_HeII_BCZ_fit.plot([0,estimate]+mid_point, [ypos2,ypos2],
                                    linestyle='solid', color='k', linewidth=2)
        ax_smooth_HeII_BCZ_fit.text(mid_point, ypos2, r'Estimation: {:.1f} $\mu$Hz'.format(estimate),
                                    horizontalalignment='left', verticalalignment='bottom')
        ax_smooth_HeII_BCZ_fit.plot([0,big]+mid_point, [ypos1,ypos1],
                                    linestyle='solid', color='k', linewidth=2)
        ax_smooth_HeII_BCZ_fit.text(mid_point, ypos1, r'Upper limit: {:.1f} $\mu$Hz'.format(big),
                                    horizontalalignment='left', verticalalignment='bottom')
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_fit.savefig(fig_smooth_HeII_BCZ) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ)    
    
        # ==========================================================================
        ### Plot smooth, HeII, BCZ and H&HeI fit to second diferences
        # ==========================================================================
        fig_smooth_HeII_BCZ_H = plt.figure()
        ax_smooth_HeII_BCZ_H_fit, ax_smooth_HeII_BCZ_H_residual = plot_secdiff(fig_smooth_HeII_BCZ_H,
                                                                               fits['nu_secdiff_filtered'], 
                                                                               fits['secdiff_filtered'],
                                                                               profile,
                                                                               fits['age'],
                                                                               only_secdiff=False, 
                                                                               fit=True,
                                                                               f=fit_smooth_HeII_BCZ_H, 
                                                                               params=list(fits['fit_smooth_HeII_BCZ_H']['results'].values()),
                                                                               residuals=fits['fit_smooth_HeII_BCZ_H']['residuals'])  
    
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_fit.savefig(fig_smooth_HeII_BCZ_H) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H)    
    
        # ==========================================================================
        ### Plot smooth, HeII, BCZ and H&HeI fit to second diferences after the grid
        # ==========================================================================
    
        fig_smooth_HeII_BCZ_H_grid = plt.figure()
        ax_smooth_HeII_BCZ_H_grid_fit, ax_smooth_HeII_BCZ_H_grid_residual = plot_secdiff(fig_smooth_HeII_BCZ_H_grid,
                                                                                         fits['nu_secdiff_filtered'],
                                                                                         fits['secdiff_filtered'],
                                                                                         profile,
                                                                                         fits['age'],
                                                                                         only_secdiff=False, 
                                                                                         fit=True,
                                                                                         f=fit_smooth_HeII_BCZ_H, 
                                                                                         params=list(fits['fit_smooth_HeII_BCZ_H_grid']['results'].values()), 
                                                                                         residuals=fits['fit_smooth_HeII_BCZ_H_grid']['residuals'])    
        
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_grid_fit.savefig(fig_smooth_HeII_BCZ_H_grid) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_grid)    
    
        # ==========================================================================
        ### Plot histograms from the grid
        # ==========================================================================
        
        grid_list = [fits['fit_smooth_HeII_BCZ_H_grid']['tau_BCZ_grid'], 
                     fits['fit_smooth_HeII_BCZ_H_grid']['tau_HeII_grid'],
                     fits['fit_smooth_HeII_BCZ_H_grid']['tau_H_grid']]
    
        # Get the chi2 squared value of all the grid    
        chi2_grid = [ _['residuals']['chi2'] for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ] 
        chi2_grid = np.array(chi2_grid)
    
        ### Histogram results
        zip_tau_xlabel_grid = zip(['tau_BCZ', 'tau_HeII', 'tau_H', ('tau_BCZ','tau_HeII')],
                                  ['BCZ', 'HeII', 'H and HeI', 'Ratio'],
                                  grid_list+[[]])
        fig_hist=plt.figure()
        gs=gridspec.GridSpec(2,2)
        for i,(tau,xlabel,grid) in enumerate(zip_tau_xlabel_grid):
            if xlabel != 'Ratio':
                tau_results_grid = [ _['results'][tau] for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ]        
            if xlabel == 'Ratio':
                tau_results_grid = [ _['results']['tau_BCZ']/_['results']['tau_HeII'] if _['results']['tau_BCZ'] != None and _['results']['tau_HeII'] != None else None for _ in fits['fit_smooth_HeII_BCZ_H_grid_all'] ]
            tau_results_grid = np.array(tau_results_grid)
            # Exclude where fit did not converge
            ind_converge = tau_results_grid != None
            # Generate the bins
            hist, bins = np.histogram(tau_results_grid[ind_converge], bins='auto')
            # Search which bin host tau_results_grid
            bin_host = np.searchsorted(bins, tau_results_grid[ind_converge])
            # If there are 3 bins: in_host=0 is before the first bin and in_host=4 is after the last bin
            # For each bin (If there are 3 bins. The bins are 1,2,3)
            tau_results_grid_prob = list()
            for i_bin in np.linspace(1,len(bins)-1,num=len(bins)-1):
                # Select from bin_host the ones equal i_bin
                tau_results_grid_prob.append(np.sum(np.exp(-chi2_grid[ind_converge][bin_host==i_bin]**2)))
            # Gaussian probability                                     
            tau_results_grid_prob = np.array(tau_results_grid_prob)
            heights = tau_results_grid_prob.copy()
            widths = np.diff(bins*1e6)
            area = np.sum(heights*widths)
            heights = heights/area
            tau_results_grid_prob = tau_results_grid_prob/area
            # Histogram plot
            ax_hist=plt.subplot(gs[i])
            if xlabel != 'Ratio':
                ax_hist.fill_between(bins*1e6,np.concatenate([[0],tau_results_grid_prob]),color='dodgerblue',step='pre')
            if xlabel == 'Ratio':
                ax_hist.fill_between(bins,np.concatenate([[0],tau_results_grid_prob]),color='dodgerblue',step='pre')
            # Histogram borders
            # If bin=[1,2,3,4,5] this generate [1,1, 2,2,2, 3,3,3, 4,4,4, 5,5]
            if xlabel != 'Ratio':
                bins3 = 1e6 * np.array([bins,bins,bins]).T.reshape(-1)[1:-1]
            if xlabel == 'Ratio':
                bins3 = np.array([bins,bins,bins]).T.reshape(-1)[1:-1]
            # If heights=[1,2,3] this generate [0,1,1,0,2,2,0,3,3,0]
            heights = np.array([heights,heights,np.repeat(0,len(heights))]).T.reshape(-1)
            heights = np.insert(heights,0,0)
            ax_hist.plot(bins3,heights,color='k',linestyle='solid')
            ymin, ymax = plt.ylim()
            ypos = (ymax-ymin)*0.9
            if xlabel != 'Ratio':
                ax_hist.plot(grid*1e6,np.repeat(ypos,grid.size), color='orange',
                             marker='|', linestyle='None', markeredgewidth=2.5,
                             markersize=20)
                # Label
                ax_hist.set_xlabel(r'Acoustical depth ' + xlabel + ' ( sec )')
            if xlabel == 'Ratio':
#                ax_hist.plot(grid*1e6,np.repeat(ypos,grid.size), color='orange',
#                             marker='|', linestyle='None', markeredgewidth=2.5,
#                             markersize=20)
                # Label
                ax_hist.set_xlabel( xlabel + ' Tau BCZ over Tau HeII')
            ax_hist.set_ylabel( 'Histogram weighted by\n'+r'exp($\chi^2$). Normalized', fontsize=7 )
            plt.setp(ax_hist.get_xticklabels(),fontsize=7)
#            plt.setp(ax_hist.get_yticklabels(),fontsize=7)
            # No y-axis ticks
            plt.setp(ax_hist.get_yticklabels(), visible=False)

        fig_hist.suptitle('Age after ZAMS: {:.3f} Gyr; Number of combinations {}; {}'.format(fits['age'], 
                                                                                             np.product([ len(grid) for grid in grid_list]),
                                                                                             profile))
        fig_hist.subplots_adjust(hspace=0.4)        
        pdf_smooth_HeII_BCZ_H_grid_all_fit_hist_results.savefig(fig_hist)
        plt.close(fig_hist)
        
        # ==========================================================================
        ### Plot smooth, HeII, BCZ and H&HeI fit to second diferences after the grid and repeat
        # ==========================================================================
    
        fig_smooth_HeII_BCZ_H_repeat = plt.figure()
        ax_smooth_HeII_BCZ_H_repeat_fit, ax_smooth_HeII_BCZ_H_repeat_residual = plot_secdiff(fig_smooth_HeII_BCZ_H_repeat,
                                                                                             fits['nu_secdiff_filtered'],
                                                                                             fits['secdiff_filtered'],
                                                                                             profile,
                                                                                             fits['age'],
                                                                                             only_secdiff=False, 
                                                                                             fit=True,
                                                                                             f=fit_smooth_HeII_BCZ_H, 
                                                                                             params=list(fits['fit_smooth_HeII_BCZ_H_repeat']['results'].values()), 
                                                                                             residuals=fits['fit_smooth_HeII_BCZ_H_repeat']['residuals'])    
        
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_repeated.savefig(fig_smooth_HeII_BCZ_H_repeat) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_repeat)  

        # =============================================================================
        ### Plot the individual components of the best fit (the last one)    
        # =============================================================================
    
        # Best parameters    
        best_params = list(fits['fit_smooth_HeII_BCZ_H_grid']['results'].values())
        ind = np.argsort(fits['fit_smooth_HeII_BCZ_H_grid']['nu_secdiff_all_l'])
        all_secdiff_filtered_sorted =  fits['fit_smooth_HeII_BCZ_H_grid']['secdiff_all_l'][ind]
        all_nu_secdiff_filtered_sorted = fits['fit_smooth_HeII_BCZ_H_grid']['nu_secdiff_all_l'][ind]
        all_secdiff_filtered =  fits['fit_smooth_HeII_BCZ_H_grid']['secdiff_all_l']
        all_nu_secdiff_filtered = fits['fit_smooth_HeII_BCZ_H_grid']['nu_secdiff_all_l']
        secdiff_cov = fits['fit_smooth_HeII_BCZ_H_grid']['secdiff_cov']
        ### Residual dictionary
        residuals_components_dict = dict()
        residuals_smooth_component_dict = dict()
        residuals_smooth_HeII_component_dict = dict()
        residuals_smooth_HeII_BCZ_component_dict = dict()
        residuals_smooth_HeII_H_component_dict = dict()
        residuals_smooth_HeII_BCZ_H_component_dict = dict()
        residuals_oscillating_component_dict = dict()
        residuals_HeII_BCZ_H_component_dict = dict()
        residuals_smooth_BCZ_H_component_dict = dict()
     
        # Smooth component
        for l in ell:
            residuals_smooth_component_dict[l] = fits['secdiff_filtered'][l] - fit_smooth(fits['nu_secdiff_filtered'][l], *best_params[0:3])
        residuals_smooth_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth(all_nu_secdiff_filtered_sorted, *best_params[0:3])
        residuals_smooth_component_dict['all_l'] = all_secdiff_filtered - fit_smooth(all_nu_secdiff_filtered, *best_params[0:3])
        # Chi square
        #residuals_smooth_component_dict['chi2'] = np.sum(residuals_smooth_component_dict['all_l_ordered']**2)
        residuals_smooth_component_dict['chi2'] = residuals_smooth_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_smooth_component_dict['all_l']
        residuals_smooth_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params[0:3])
        residuals_smooth_component_dict['reduced_chi2'] = residuals_smooth_component_dict['chi2'] / residuals_smooth_component_dict['dof']
        residuals_components_dict['smooth'] = residuals_smooth_component_dict
    
        # Smooth and HeII component
        for l in ell:
            residuals_smooth_HeII_component_dict[l] = fits['secdiff_filtered'][l] - fit_smooth_HeII(fits['nu_secdiff_filtered'][l], *best_params[0:7])
        residuals_smooth_HeII_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII(all_nu_secdiff_filtered_sorted, *best_params[0:7])
        residuals_smooth_HeII_component_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII(all_nu_secdiff_filtered, *best_params[0:7])
        # Chi square
        #residuals_smooth_HeII_component_dict['chi2'] = np.sum(residuals_smooth_HeII_component_dict['all_l_ordered']**2)
        residuals_smooth_HeII_component_dict['chi2'] = residuals_smooth_HeII_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_smooth_HeII_component_dict['all_l']
        residuals_smooth_HeII_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params[0:7])
        residuals_smooth_HeII_component_dict['reduced_chi2'] = residuals_smooth_HeII_component_dict['chi2'] / residuals_smooth_HeII_component_dict['dof']
        residuals_components_dict['smooth_HeII'] = residuals_smooth_HeII_component_dict
    
        # Smooth, HeII and BCZ component
        for l in ell:
            residuals_smooth_HeII_BCZ_component_dict[l] = fits['secdiff_filtered'][l] - fit_smooth_HeII_BCZ(fits['nu_secdiff_filtered'][l], *best_params[0:10])
        residuals_smooth_HeII_BCZ_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_BCZ(all_nu_secdiff_filtered_sorted, *best_params[0:10])
        residuals_smooth_HeII_BCZ_component_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_BCZ(all_nu_secdiff_filtered, *best_params[0:10])
        # Chi square
        #residuals_smooth_HeII_BCZ_component_dict['chi2'] = np.sum(residuals_smooth_HeII_BCZ_component_dict['all_l_ordered']**2)
        residuals_smooth_HeII_BCZ_component_dict['chi2'] = residuals_smooth_HeII_BCZ_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_smooth_HeII_BCZ_component_dict['all_l']
        residuals_smooth_HeII_BCZ_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params[0:10])
        residuals_smooth_HeII_BCZ_component_dict['reduced_chi2'] = residuals_smooth_HeII_BCZ_component_dict['chi2'] / residuals_smooth_HeII_BCZ_component_dict['dof']
        residuals_components_dict['smooth_HeII_BCZ'] = residuals_smooth_HeII_BCZ_component_dict

        # Smooth, HeII and H component
        for l in ell:
            residuals_smooth_HeII_H_component_dict[l] = fits['secdiff_filtered'][l] - fit_smooth_HeII_H(fits['nu_secdiff_filtered'][l], *np.array(best_params)[[0,1,2,3,4,5,6,10,11,12,13]])
        residuals_smooth_HeII_H_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_H(all_nu_secdiff_filtered_sorted, *np.array(best_params)[[0,1,2,3,4,5,6,10,11,12,13]])
        residuals_smooth_HeII_H_component_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_H(all_nu_secdiff_filtered, *np.array(best_params)[[0,1,2,3,4,5,6,10,11,12,13]])
        # Chi square
        #residuals_smooth_HeII_H_component_dict['chi2'] = np.sum(residuals_smooth_HeII_H_component_dict['all_l_ordered']**2)
        residuals_smooth_HeII_H_component_dict['chi2'] = residuals_smooth_HeII_H_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_smooth_HeII_H_component_dict['all_l']
        residuals_smooth_HeII_H_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params[0:10])
        residuals_smooth_HeII_H_component_dict['reduced_chi2'] = residuals_smooth_HeII_H_component_dict['chi2'] / residuals_smooth_HeII_H_component_dict['dof']
        residuals_components_dict['smooth_HeII_H'] = residuals_smooth_HeII_H_component_dict
    
        # Smooth, HeII, BCZ and H component
        for l in ell:
            residuals_smooth_HeII_BCZ_H_component_dict[l] = fits['secdiff_filtered'][l] - fit_smooth_HeII_BCZ_H(fits['nu_secdiff_filtered'][l], *best_params)
        residuals_smooth_HeII_BCZ_H_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered_sorted, *best_params)
        residuals_smooth_HeII_BCZ_H_component_dict['all_l'] = all_secdiff_filtered - fit_smooth_HeII_BCZ_H(all_nu_secdiff_filtered, *best_params)
        # Chi square
        #residuals_smooth_HeII_BCZ_H_component_dict['chi2'] = np.sum(residuals_smooth_HeII_BCZ_H_component_dict['all_l_ordered']**2)
        residuals_smooth_HeII_BCZ_H_component_dict['chi2'] = residuals_smooth_HeII_BCZ_H_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_smooth_HeII_BCZ_H_component_dict['all_l']
        residuals_smooth_HeII_BCZ_H_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params)
        residuals_smooth_HeII_BCZ_H_component_dict['reduced_chi2'] = residuals_smooth_HeII_BCZ_H_component_dict['chi2'] / residuals_smooth_HeII_BCZ_H_component_dict['dof']
        residuals_components_dict['smooth_HeII_BCZ_H'] = residuals_smooth_HeII_BCZ_H_component_dict

        # HeII, BCZ and H component
        for l in ell:
            residuals_HeII_BCZ_H_component_dict[l] = fits['secdiff_filtered'][l] - fit_HeII_BCZ_H(fits['nu_secdiff_filtered'][l], *best_params[3:])
        residuals_HeII_BCZ_H_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_HeII_BCZ_H(all_nu_secdiff_filtered_sorted, *best_params[3:])
        residuals_HeII_BCZ_H_component_dict['all_l'] = all_secdiff_filtered - fit_HeII_BCZ_H(all_nu_secdiff_filtered, *best_params[3:])
        # Chi square
        #residuals_HeII_BCZ_H_component_dict['chi2'] = np.sum(residuals_HeII_BCZ_H_component_dict['all_l_ordered']**2)
        residuals_HeII_BCZ_H_component_dict['chi2'] = residuals_HeII_BCZ_H_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_HeII_BCZ_H_component_dict['all_l']
        residuals_HeII_BCZ_H_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params)
        residuals_HeII_BCZ_H_component_dict['reduced_chi2'] = residuals_HeII_BCZ_H_component_dict['chi2'] / residuals_HeII_BCZ_H_component_dict['dof']
        residuals_components_dict['HeII_BCZ_H'] = residuals_HeII_BCZ_H_component_dict
    
        # Smooth, BCZ and H component
        for l in ell:
            residuals_smooth_BCZ_H_component_dict[l] = fits['secdiff_filtered'][l] - fit_smooth_BCZ_H(fits['nu_secdiff_filtered'][l], *np.array(best_params)[[0,1,2,7,8,9,10,11,12,13]])
        residuals_smooth_BCZ_H_component_dict['all_l_ordered'] = all_secdiff_filtered_sorted - fit_smooth_BCZ_H(all_nu_secdiff_filtered_sorted, *np.array(best_params)[[0,1,2,7,8,9,10,11,12,13]])
        residuals_smooth_BCZ_H_component_dict['all_l'] = all_secdiff_filtered - fit_smooth_BCZ_H(all_nu_secdiff_filtered, *np.array(best_params)[[0,1,2,7,8,9,10,11,12,13]])
        # Chi square
        #residuals_smooth_BCZ_H_component_dict['chi2'] = np.sum(residuals_smooth_BCZ_H_component_dict['all_l_ordered']**2)
        residuals_smooth_BCZ_H_component_dict['chi2'] = residuals_smooth_BCZ_H_component_dict['all_l'].T @ np.linalg.inv(secdiff_cov) @ residuals_smooth_BCZ_H_component_dict['all_l']
        residuals_smooth_BCZ_H_component_dict['dof'] = len(all_nu_secdiff_filtered_sorted)-len(best_params)
        residuals_smooth_BCZ_H_component_dict['reduced_chi2'] = residuals_smooth_BCZ_H_component_dict['chi2'] / residuals_smooth_BCZ_H_component_dict['dof']
        residuals_components_dict['smooth_BCZ_H'] = residuals_smooth_BCZ_H_component_dict


        # PLOT1
        fig_smooth_HeII_BCZ_H_individual_components1 = plt.figure()
        ax_smooth_HeII_BCZ_H_individual_components_smooth, \
        ax_smooth_HeII_BCZ_H_individual_components_oscillating, \
        ax_smooth_HeII_BCZ_H_individual_components_residuals = \
        plot_secdiff(fig_smooth_HeII_BCZ_H_individual_components1,
                     fits['nu_secdiff_filtered'], fits['secdiff_filtered'],
                     profile, fits['age'],
                     only_secdiff=False, fit=False, individual_components1=True, individual_components2=False, individual_components3=False, individual_components4=False, individual_components_all=False,
                     f=fit_smooth_HeII_BCZ_H, params=best_params, residuals=residuals_components_dict)  
    
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_grid_components1.savefig(fig_smooth_HeII_BCZ_H_individual_components1) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_individual_components1)
        
        # PLOT2
        fig_smooth_HeII_BCZ_H_individual_components2 = plt.figure()
        ax_smooth_HeII_BCZ_H_individual_components_smooth, \
        ax_smooth_HeII_BCZ_H_individual_components_oscillating, \
        ax_smooth_HeII_BCZ_H_individual_components_residuals = \
        plot_secdiff(fig_smooth_HeII_BCZ_H_individual_components2,
                     fits['nu_secdiff_filtered'], fits['secdiff_filtered'],
                     profile, fits['age'],
                     only_secdiff=False, fit=False, individual_components1=False, individual_components2=True, individual_components3=False, individual_components4=False, individual_components_all=False,
                     f=fit_smooth_HeII_BCZ_H, params=best_params, residuals=residuals_components_dict)  
    
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_grid_components2.savefig(fig_smooth_HeII_BCZ_H_individual_components2) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_individual_components2)

        # PLOT3
        fig_smooth_HeII_BCZ_H_individual_components3 = plt.figure()
        ax_smooth_HeII_BCZ_H_individual_components_smooth, \
        ax_smooth_HeII_BCZ_H_individual_components_oscillating, \
        ax_smooth_HeII_BCZ_H_individual_components_residuals = \
        plot_secdiff(fig_smooth_HeII_BCZ_H_individual_components3,
                     fits['nu_secdiff_filtered'], fits['secdiff_filtered'],
                     profile, fits['age'],
                     only_secdiff=False, fit=False, individual_components1=False, individual_components2=False, individual_components3=True, individual_components4=False, individual_components_all=False,
                     f=fit_smooth_HeII_BCZ_H, params=best_params, residuals=residuals_components_dict)  
    
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_grid_components3.savefig(fig_smooth_HeII_BCZ_H_individual_components3) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_individual_components3)

        # PLOT4
        fig_smooth_HeII_BCZ_H_individual_components4 = plt.figure()
        ax_smooth_HeII_BCZ_H_individual_components_smooth, \
        ax_smooth_HeII_BCZ_H_individual_components_oscillating, \
        ax_smooth_HeII_BCZ_H_individual_components_residuals = \
        plot_secdiff(fig_smooth_HeII_BCZ_H_individual_components4,
                     fits['nu_secdiff_filtered'], fits['secdiff_filtered'],
                     profile, fits['age'],
                     only_secdiff=False, fit=False, individual_components1=False, individual_components2=False, individual_components3=False, individual_components4=True, individual_components_all=False,
                     f=fit_smooth_HeII_BCZ_H, params=best_params, residuals=residuals_components_dict)  
    
        # Save in a page of the PDF
        pdf_smooth_HeII_BCZ_H_grid_components4.savefig(fig_smooth_HeII_BCZ_H_individual_components4) 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_individual_components4)

        # PLOT ALL INDIVIDUAL COMPONENTS
        fig_smooth_HeII_BCZ_H_individual_components_all = plt.figure()
        ax_smooth_HeII_BCZ_H_individual_components_smooth, \
        ax_smooth_HeII_BCZ_H_individual_components_oscillating, \
        ax_smooth_HeII_BCZ_H_individual_components_residuals, \
        ax_tmp, \
        ax_tmp, \
        ax_tmp = \
        plot_secdiff(fig_smooth_HeII_BCZ_H_individual_components_all,
                     fits['nu_secdiff_filtered'], fits['secdiff_filtered'],
                     profile, fits['age'],
                     only_secdiff=False, fit=False, individual_components1=False, individual_components2=False, individual_components3=False, individual_components4=False, individual_components_all=True,
                     f=fit_smooth_HeII_BCZ_H, params=best_params, residuals=residuals_components_dict)  

        # Save in a page of the PDF
#        pdf_smooth_HeII_BCZ_H_grid_components_all.savefig(fig_smooth_HeII_BCZ_H_individual_components_all) 
        pdf_smooth_HeII_BCZ_H_grid_components_all.savefig(fig_smooth_HeII_BCZ_H_individual_components_all, bbox_inches='tight') 
        # Close the figure
        plt.close(fig_smooth_HeII_BCZ_H_individual_components_all)    
        
    #Close PDF
    if PLOT_SECDIFF: pdf_secdiff.close()
    pdf_smooth_fit.close()
    pdf_smooth_HeII_fit.close()
    pdf_smooth_HeII_BCZ_fit.close()
    pdf_BCZ_fit.close()
    pdf_smooth_HeII_BCZ_H_fit.close()
    pdf_smooth_HeII_BCZ_H_grid_fit.close()
    if PLOT_ALL_GRID: pdf_smooth_HeII_BCZ_H_grid_all_fit.close()
    pdf_smooth_HeII_BCZ_H_grid_all_fit_hist_initial_guesses.close()
    pdf_smooth_HeII_BCZ_H_grid_all_fit_hist_results.close()
    pdf_smooth_HeII_BCZ_H_repeated.close()
    pdf_smooth_HeII_BCZ_H_grid_components1.close()
    pdf_smooth_HeII_BCZ_H_grid_components2.close()
    pdf_smooth_HeII_BCZ_H_grid_components3.close()
    pdf_smooth_HeII_BCZ_H_grid_components4.close()
    pdf_smooth_HeII_BCZ_H_grid_components_all.close()

    time2 = time.time()

    print( 'Plots took {:.3} min'.format((time2-time1)/60) )

    print('Creating plots')
    if FILE_NOTICE:
        file_notice = open(file_notice_name,'a')
        file_notice.write('Plots took {:.3} min\n'.format((time2-time1)/60))
        file_notice.close()

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
sys.path.insert(0, '/home/stefano/python/other_libraries/mios')
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
import os
import subprocess
import tomso
from tomso import fgong
import pickle


# =============================================================================
# =============================================================================

def second_differences(nu):
    
    nu_second_differences = nu[1:-1]
    second_differences = nu[:-2] - 2*nu[1:-1] + nu[2:]
    
    return nu_second_differences, second_differences

# =============================================================================
# =============================================================================

# Directory where second_differences.py is:
wdir= '/media/stefano/Elements/master_thesis/final_models/y.0.28.z.0.02/'

# Masses to address:
#masses = ['0.8', '0.9', '1.1', '1.2'] # Solar masses
masses = ['0.8', '0.9', '1.0', '1.1', '1.2'] # Solar masses
#masses = ['1.0'] # Solar masses
#masses = ['0.8','1.2'] # Solar masses

for mass in masses:
    
    os.chdir(wdir)
    mass_folder =  mass + 'M/'
    freq_folder =  'frequencies/'
    
    # Go to mass directory
    os.chdir(wdir+mass_folder)

    # Here we will save the frequncies
    savefile = open('frequencies.bin','wb')

    # Go to frequency directory
    os.chdir(wdir+mass_folder+freq_folder)
    
    # Read the agsm.mesa.profile.txt files (ordered by number)    
    agsm_file, agsm_file_number, ind = utilities.find_ordered_files(wdir+mass_folder+freq_folder+'agsm.mesa.profile*.txt')
    nagsm_file = len(agsm_file)

    # Dictionary where we will store the results
    profiles = dict()
    
    # For each of the agsm txt file, convert them to adipls binary type.
    for i in range(nagsm_file):
        # Read file
        l, n, nu = np.genfromtxt(agsm_file[i], usecols=(0,1,2), unpack=True)
    
        # Not negative n value (Filtering out g-modes?)
        l = np.delete(l,np.where(n < 1))
        nu = np.delete(nu,np.where(n < 1))
        n = np.delete(n,np.where(n < 1))
        
        # Calculate the second differencese for each dregee l       
        nu_second_differences_l0, second_differences_l0 = second_differences(nu[l==0])
        nu_second_differences_l1, second_differences_l1 = second_differences(nu[l==1])
        nu_second_differences_l2, second_differences_l2 = second_differences(nu[l==2])
        nu_second_differences_l3, second_differences_l3 = second_differences(nu[l==3]) 
 
        # Correct extremas
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
        frequencies=np.empty(len(nu),dtype=types)
        frequencies['nu']=nu
        frequencies['n']=n
        frequencies['l']=l
        frequencies['secdiff']=np.concatenate([second_differences_l0,second_differences_l1,second_differences_l2,second_differences_l3])        

        profile_name = 'profile'+str(agsm_file_number[i])
        
        profiles[profile_name] = frequencies

    # Save the frequencies to the file        
    pickle.dump(profiles, savefile)
    
    savefile.close()    

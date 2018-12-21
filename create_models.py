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

MESA = True
ADIPLS = True

# Directory where create_models.py is:
wdir= '/media/stefano/Elements/master_thesis/final_models/y.0.28.z.0.02/'

# Masses to create:
#masses = ['0.8', '0.9', '1.1', '1.2'] # Solar masses
masses = ['0.8', '0.9', '1.0', '1.1', '1.2'] # Solar masses
#masses = ['1.0'] # Solar masses
#masses = ['0.8','1.2'] # Solar masses
#masses = ['0.9','1.1'] # Solar masses

for mass in masses:
    
    os.chdir(wdir)
    mass_folder =  mass + 'M/'
    freq_folder =  'frequencies/'
    
    # ==========================================================================
    # ==========================================================================

    if MESA:
        # Create mass directory
        command = 'mkdir ' + mass_folder
        return_code = subprocess.call(command, shell=True)
    
        # Go to mass directory
        os.chdir(wdir+mass_folder) 
    
        # Copy MESA work directory
        command = 'cp -R /home/stefano/MESA/mesa/star/work '+wdir+mass_folder
        return_code = subprocess.call(command, shell=True)

        # Go to the work directory inside the mass directory
        os.chdir(wdir+mass_folder+'work') 
    
        # Copy the templates:
        #  extra_history_columns_copy
        #  extra_profile_columns_copy
        #  inlist_copy
        #  inlist_pgstar_copy
        #  inlist_project_copy
        command = 'cp '+wdir+'inlist_project_copy '+wdir+mass_folder+'work/inlist_project' 
        return_code = subprocess.call(command, shell=True)
        command = 'cp '+wdir+'extra_profile_columns_copy '+wdir+mass_folder+'work/extra_profile_columns' 
        return_code = subprocess.call(command, shell=True)
        command = 'cp '+wdir+'inlist_copy '+wdir+mass_folder+'work/inlist' 
        return_code = subprocess.call(command, shell=True)
        command = 'cp '+wdir+'inlist_pgstar_copy '+wdir+mass_folder+'work/inlist_pgstar' 
        return_code = subprocess.call(command, shell=True)
        command = 'cp '+wdir+'extra_history_columns_copy '+wdir+mass_folder+'work/extra_history_columns' 
        return_code = subprocess.call(command, shell=True)
    
        # Update inlist_project
        utilities.simple_update_file({'initial_mass':mass},
                                     name='inlist_project',
                                     comment_character='!')
    
        # Run MESA
        command = './clean'
        return_code = subprocess.call(command, shell=True)
        command = './mk'
        return_code = subprocess.call(command, shell=True)
        command = './rn > log.txt'
        return_code = subprocess.call(command, shell=True)
    
    # ==========================================================================
    # ==========================================================================
    
    if ADIPLS:
        # Go back to the mass directory
        os.chdir(wdir+mass_folder) 
        
        # Create frequency directory
        command = 'mkdir ' + freq_folder
        return_code = subprocess.call(command, shell=True)
        
        # Go to frequency directory
        os.chdir(wdir+mass_folder+freq_folder)
        
        # Read the FGONG files (ordered by number)    
        FGONG_file, FGONG_file_number, ind = utilities.find_ordered_files(wdir+mass_folder+'work/LOGS/*.FGONG')
        nFGONG_file = len(FGONG_file)

        # For each of the FGONG files, convert them to adipls binary type.
        for i in range(nFGONG_file):
            # Read FGONF file
            fgong_glob,fgong_var = tomso.fgong.load_fgong(FGONG_file[i]) 
            # Convert to amdl format
            amdl = tomso.fgong.fgong_to_amdl(fgong_glob,fgong_var)
            # Output name (for the adipls binary)
            output_name = 'amdl.mesa.profile' + str(FGONG_file_number[i])        
            # Save
            tomso.adipls.save_amdl(output_name,amdl[0],amdl[1])
            # Status
            print('Mas '+mass+'M: '+output_name+' created.')
    
        # Copy the templates:
        #  adipls.c.in_copy
        command = 'cp '+wdir+'adipls.c.in_copy '+wdir+mass_folder+freq_folder+'adipls.c.in' 
        return_code = subprocess.call(command, shell=True)

        # Read the amld files (ordered by number)    
        amdl_file, amdl_file_number, ind = utilities.find_ordered_files(wdir+mass_folder+freq_folder+'amdl.mesa.profile*')
        namdl_file = len(amdl_file)

        # For each of the amld files, run ADIPLS.
        # Then convert the agsm frequencies to text
        for i in range(namdl_file):
            # updated the adipls.c.in file
            amde_name = 'amde.mesa.profile' + str(amdl_file_number[i])
            agsm_name = 'agsm.mesa.profile' + str(amdl_file_number[i])
            utilities.update_ADIPLS_infile({'2':amdl_file[i], 
                                            '4':amde_name, 
                                            '11':agsm_name})

            # Run ADIPLS
            command = 'adipls.c.d adipls.c.in'
            return_code = subprocess.call(command, shell=True) 
            
            # Convert the fequencies in agsm files to text file
            command = 'set-obs.d 4 '+agsm_name+' '+agsm_name+'.txt'
            return_code = subprocess.call(command, shell=True)

            # Status
            print('Mas '+mass+'M: Frequencies '+agsm_name+' created.')     
    

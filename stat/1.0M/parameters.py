import numpy as np

'''
Here we must explain to where the paths below point to
'''
class Paths:
    MESA_history_file = 'work/LOGS/history.data'
    MESA_index_profiles_file = 'work/LOGS/profiles.index'
'''
The constants below have self explanatory names
'''
class Constants:
    Sun_age = 4.6e9 # yr
    Sun_radius = 6.95700e10 # cm

'''
Here we must explain what do the 'Scales' below do
'''
class Scales:
    time = 1e9 # 1Gyr in yr

'''
Here we must explain what do the 'Flags' below do
'''
class Flags:
    INTERACTIVE_PLOTING = False
    PLOT_SECDIFF = True
    PLOT_SMOOTH_FIT = True
    PLOT_ALL_GRID = False
    MAKE_PLOTS = False
    FILE_NOTICE = True
    IMPOSE_TAU_HEII_TO_BE_WITHIN_THE_SEARCHING_RANGE = False

'''
Here we must explain what do the name of the files below correspond to
'''
class File_names:
    file_notice = 'file_notice.txt'
    frequencies_file = 'frequencies.bin'

'''
Here we must explain what do 'Others' are
'''
class Others:
    min_delta_age = 0.1 # Gyr

'''
Here we define variables used to store results. We must define each of them
'''
class profile_results:
    l = np.nan # angular degree of the eigenfrequency
    age = np.nan # age stellar model from which the frequencies are obtained
    secdiff = np.nan # second differences computed from the eigenfrequencies
    nu_secdiff = np.nan # nu_{n} from nu_{n+1} - 2*nu_{n} + nu_{n-1}
    secdiff_no_statistical_errors = np.nan
    nu_secdiff_no_statistical_errors = np.nan
    secdiff_filtered = np.nan
    nu_secdiff_filtered = np.nan


import astropy
import dill as pickle
from astropy import units
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.nddata import NDDataRef
import astropy.units as u
from astropy.modeling import models
import m2fs_process as m2fs
import os
from os import path
import mycode
import specutils
from specutils import Spectrum1D
import scipy
from astropy.nddata import StdDevUncertainty
from astropy.visualization import quantity_support
import numpy as np
from specutils.fitting import fit_lines
from specutils.analysis import centroid
from specutils.analysis import fwhm
from specutils.analysis import line_flux
from specutils.analysis import equivalent_width
from specutils import SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models,fitting
from specutils.fitting import estimate_line_parameters
from ccdproc import Combiner
from scipy import interpolate
#matplotlib.use('pdf')
matplotlib.use('TkAgg')

display_image=False
initialize=False
find=False
trace_all=False
trace_edit=False
make_image=False
extract1d_flat=False
extract1d_thar=False
extract1d_sci=False
id_lines=True
reid_lines=False

linelist_file='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs_config1b_thar_list'
threshold_factor=25.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
n_lines=20#columns to combine when scanning across rows to identify apertures (as 'emission lines')
continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
lsf_rejection_iterations=10#number of iterations of outlier rejection for fitting lsf sigma
lsf_nsample=50#number of points along spectral- (x-) axis at which to measure LSF sigma before performing fit of sigma(x)
lsf_order=4#order of polynomial used to fit lsf sigma as function of pixel along dispersion direction
window=10#pixels, width of aperture window for fitting (gaussian) aperture profiles (perpendicular to spectral axis)
trace_step=n_lines#tracing step
trace_nlost_max=2
trace_shift_max=1.5
trace_order=4
trace_rejection_iterations=10
trace_rejection_sigma=3.#largest rms deviation to accept in fit to aperture trace
trace_rejection_iterations=10
id_lines_continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
id_lines_threshold_factor=10.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
id_lines_window=5.#pixels, width of aperture window for fitting (gaussian) line profiles in arc spectra
id_lines_order=5#order of wavelength solution
id_lines_rejection_iterations=10#rejection iterations to run in determining wavelength solution
id_lines_rejection_sigma=3.#multiple of continuum residual rms threshold to impose for wavelength solution
directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'

#m2fsrun='nov18'
m2fsrun='may19'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2019/'
#m2fsrun='jul15'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
scifile=[]
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(int(p[3]))
        tharfile.append(p[4])
        scifile.append(p[6])
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
scifile=np.array(scifile)

tharfile0=[]
scifile0=[]
for i in range(0,len(tharfile)):
    tharfile0.append(tharfile[i].split('-'))
    scifile0.append(scifile[i].split('-'))
tharfile0=np.array(tharfile0)
scifile0=np.array(scifile0)

for i in range(0,len(utdate)):
    for ccd in ('b','r'):
        
        root=datadir+utdate[i]+'/'+ccd+str(flatfile[i]).zfill(4)
#        root2=datadir+utdate[i]+'/'+ccd+str(tharfile[i]).zfill(4)

        data_file=root+'_stitched.fits'
#        data_file2=root2+'_stitched.fits'
        image_boundary_file=root+'_image_boundary.pickle'
        columnspec_array_file=root+'_columnspec_array.pickle'
        apertures_profile_middle_file=root+'_apertures_profile_middle.pickle'
        aperture_array_file=root+'_aperture_array.pickle'
        image_file=root+'_apertures.pdf'
        extract1d_array_file=root+'_extract1d_array.pickle'

        image_boundary_exists=path.exists(image_boundary_file)
        columnspec_array_exists=path.exists(columnspec_array_file)
        apertures_profile_middle_exists=path.exists(apertures_profile_middle_file)
        aperture_array_exists=path.exists(aperture_array_file)
        extract1d_array_exists=path.exists(extract1d_array_file)

        if display_image:
            print('displaying '+root)
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            
            image_boundary=m2fs.get_image_boundary(data)
            pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file

        if initialize:#find apertures in each column stack and save to pickle files
            print('initializing '+root)
            if columnspec_array_exists:
                print('will overwrite existing '+columnspec_array_file)
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            columnspec_array=m2fs.get_columnspec(data,trace_step,n_lines,continuum_rejection_iterations,threshold_factor,window)
            '''
            columnspec_array is array of columnspec objects.
            columnspec object contains: 
            columns: numbers of columns in original data frame that are stacked
            spec1d: 'spectrum' across stacked column, where each aperture appears as an 'emission line'
            pixel: value of pixel across stacked column 'spectrum', artificially given units of AA in order to comply with specutils requirements for fitting spectra
            continuum: parameters of 1dGaussian continuum fit to stacked column 'spectrum'
            rms: rms residuals around continuum fit (using only regions between fiber bundles)
            apertures_initial: initial aperture centers returned by specutils find_lines_derivative
            apertures_profile: contains parameters of fits to apertures detected in apertures_initial
            '''
            pickle.dump(columnspec_array,open(columnspec_array_file,'wb'))#save pickle to file

        if find:
            print('finding apertures for '+root)
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            
            '''
            Make initial array of more precise aperture location fits in middle column stack
            '''
            
            if apertures_profile_middle_exists:
                print('loading existing '+apertures_profile_middle_file+', will overwrite')
                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            else:
                middle_column=np.long(len(columnspec_array)/2)
                apertures_profile_middle=columnspec_array[middle_column].apertures_profile
                
            '''
            apertures_profile_middle object contains: 
            fit: parameters of fit to cross-aperture profile ('line' profile) *for middle column*
            subregion: pixel range of cross-aperture 'spectrum' used for fit *for middle column*
            initial: value of True means aperture was found in automatic procedure, value of False means it was inserted by hand
            realvirtual: value of True means aperture corresponds to real spectrum, value of False means virtual aperture used as place-holder (due to bad or unplugged fiber)
            '''

            '''
            display apertures initially identified in central column stack and allow user to add/delete apertures with screen input
            '''
            apertures_profile_middle=m2fs.fiddle_apertures(columnspec_array,middle_column,window,apertures_profile_middle)
            pickle.dump([apertures_profile_middle,middle_column],open(apertures_profile_middle_file,'wb'))#save pickle to file

        if trace_all:
            print('tracing all apertures for '+root)
            if aperture_array_exists:
                print('will overwrite existing '+aperture_array_file)
            image_boundary=pickle.load(open(image_boundary_file,'rb'))
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            
            aperture_array=[]
            for j in range(0,len(apertures_profile_middle.fit)):
                if apertures_profile_middle.realvirtual[j]:
                    aperture_array.append(m2fs.get_aperture(j,columnspec_array,apertures_profile_middle,middle_column,trace_order,trace_rejection_sigma,trace_rejection_iterations,image_boundary,trace_shift_max,trace_nlost_max,lsf_rejection_iterations,lsf_nsample,lsf_order,window))

            pickle.dump(aperture_array,open(aperture_array_file,'wb'))

        if trace_edit:
            image_boundary=pickle.load(open(image_boundary_file,'rb'))
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))

            command=input('enter number of aperture to edit (integer)')
            j=np.long(command)-1
            aperture_array[j]=m2fs.get_aperture(j,columnspec_array,apertures_profile_middle,middle_column,trace_order,trace_rejection_sigma,trace_rejection_iterations,image_boundary,trace_shift_max,trace_nlost_max,lsf_rejection_iterations,lsf_nsample,lsf_order,window)

            pickle.dump(aperture_array,open(aperture_array_file,'wb'))

        if make_image:
            print('writing '+image_file)
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))
            
            vmin=np.quantile(data.data.flatten(),0.05)
            vmax=np.quantile(data.data.flatten(),0.95)
            plt.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)
            for j in range(0,len(aperture_array)):
                if aperture_array[j].trace_npoints>40:
                    x=np.linspace(aperture_array[j].trace_pixel_min,aperture_array[j].trace_pixel_max,100)
                    y=aperture_array[j].trace_func(x)
                    plt.plot(x,y,color='r',alpha=1,lw=0.2)
            plt.savefig(image_file,dpi=300)
            plt.show()
            plt.close()

        if extract1d_flat:

            print('extracting to '+extract1d_array_file)
            if extract1d_array_exists:
                print('will overwrite existing version')

            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))

            aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
            pix=np.arange(len(data.data[0]))

            extract1d_array=[]
            for j in range(0,len(aperture_array)):
                print('   extracting aperture ',aperture_array[j].trace_aperture,' of ',len(aperture_array))
                extract1d_array.append(m2fs.get_extract1d(j,data,apertures_profile_middle,aperture_array,aperture_peak,pix))
                
            pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))


        if extract1d_thar:
            
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))

            aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]

            for j in range(0,len(tharfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                extract1d_array_file=root0+'_extract1d_array.pickle'
                extract1d_array_exists=path.exists(extract1d_array_file)
                print('extracting to '+extract1d_array_file)
                if extract1d_array_exists:
                    print('will overwrite existing version')

                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows

                pix=np.arange(len(data.data[0]))

                extract1d_array=[]
                for j in range(0,len(aperture_array)):
                    print('   extracting aperture ',aperture_array[j].trace_aperture,' of ',len(aperture_array))
                    extract1d_array.append(m2fs.get_extract1d(j,data,apertures_profile_middle,aperture_array,aperture_peak,pix))
                
                pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))

        if extract1d_sci:
            
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))

            aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]

            for j in range(0,len(scifile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                extract1d_array_file=root0+'_extract1d_array.pickle'
                extract1d_array_exists=path.exists(extract1d_array_file)
                print('extracting to '+extract1d_array_file)
                if extract1d_array_exists:
                    print('will overwrite existing version')

                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows

                pix=np.arange(len(data.data[0]))

                extract1d_array=[]
                for j in range(0,len(aperture_array)):
                    print('   extracting aperture ',aperture_array[j].trace_aperture,' of ',len(aperture_array))
                    extract1d_array.append(m2fs.get_extract1d(j,data,apertures_profile_middle,aperture_array,aperture_peak,pix))
                
                pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))

        if id_lines:
            
            with open(linelist_file) as f:
                data=f.readlines()
            linelist_wavelength=[]
            linelist_species=[]
            for line in data:
                p=line.split()
                linelist_wavelength.append(float(p[0]))
                linelist_species.append(p[1])
            linelist_wavelength=np.array(linelist_wavelength)
            linelist_species=np.array(linelist_species)
            linelist=m2fs.linelist(wavelength=linelist_wavelength,species=linelist_species)

            root=datadir+utdate[i]+'/'+ccd+str(tharfile0[0][0]).zfill(4)#use first ThAr from first field in run
            extract1d_array_file=root+'_extract1d_array.pickle'
            id_lines_file=root+'_id_lines.pickle'
            id_lines_exist=path.exists(id_lines_file)

            extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
            print('identify lines for '+root)
            if extract1d_array_exists:
                print('will overwrite existing version')

            for j in range(0,len(extract1d_array)):
                id_lines_array=m2fs.get_id_lines(extract1d_array,linelist,id_lines_continuum_rejection_iterations,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_rejection_iterations,id_lines_rejection_sigma)

'''In this module I dump the classes which I try to construct'''

#### IMPORTS ####
import astropy
from astropy.nddata import CCDData
from . import m2fs_process as m2fs
import os
import pickle
from os import path
import numpy as np
from astropy import time, coordinates as coord, units as u
from astropy.nddata import StdDevUncertainty
from astropy.io import fits
from . import normalizationTools as norm_tools

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from . import function_dump as fdump
from . import interactive_plot as intacplot

# test datafile: b0034_stitched.fits

'''The first thing we need to do is to initialize.
Trim and initialize are independant
Find needs initialize
'''

'''
TODO: 
-   When checking if file exsists, there is a lot of code that is simply 
    repeated. Need a clean up.
-   Implement multiple filter handling in id_lines_translate? See docstring for the function
-   Some files may be corrupted and therefore unreadable. Add a check in check_data to list
    the corrupted files, remove them from the initial arrays, and check taht we have at least
    one science, one thar, one led, one bias and one dark.
'''


class ReduceM2FS:
    """This class is the main analysis object, containing the
    routines for data reduction.

    NB: For now, I initialize the img_bounds boundary of the image on
    construction of the object, which means that it cannot start from stored
    data. In the future this could be updated.

    Attributes:
    - data      : astropy CCDData object containing data to be reduced

    """
    def __init__(self):
        '''There are a few things in this code that will be used
        by several functions. It makes very little sense to go on
        with IO at every function. Instead, will load everything
        in the initialization step.
        If I believe there's enough things to store to justify an object
        I will. Else they will be attributes.
        '''

        # self.set_datapath("/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-master/m2fs/polo/ut20191116/")
        # self.set_filename(datafile)
        # self.read_data()
        # self.img_bounds = m2fs.image_boundary(0, 1) ## contains the data on the bounds to use on the image

        ## Initialize some stuff?
        # self.data # contains the data from the FLAT file

        ## Let's store here the result of initialize:
        self.columnspec_array = []
        self.throughputcorr_bool = False ## Indicator for whether data was throughput correted.
        self.crreject_bool = False
        self.flat_field_bool = False
        self.normalize_sky = False
        
        ## Let us define some of the fundamental variables for the analysis
        self.trace_step = 20 # Nb columns combined when scanning across rows to identify apertures [default was 20]
        self.n_lines = self.trace_step # tracing steps - Always the same?
        self.columnspec_continuum_rejection_low = -5. # Low limit rejection of points?
        self.columnspec_continuum_rejection_high = 1. # Higher limit?
        self.columnspec_continuum_rejection_iterations = 10 #number of iterations of outlier rejection for fitting "continuum"
        self.columnspec_continuum_rejection_order = 10 # ???
        self.threshold_factor = 25 # multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
        self.window = 10 #pixels, width of aperture window for fitting (gaussian) aperture profiles (perpendicular to spectral axis)
        self.overwrite = True
        self.trace_order = 4 ## Order used to trace
        self.trace_rejection_sigma = 3. # largest rms deviation to accept in fit to aperture trace
        self.trace_rejection_iterations = 10
        self.trace_shift_max = 5. # PIC: Can be too restrictive? 
                                    # And what is that for anyway? Are we really afraid to fall
                                    # onto the next aperture if it's too large? 
        self.trace_nlost_max = 2.
        self.profile_rejection_iterations = 10
        self.profile_nsample = 50 #number of points along spectral- (x-) axis at which to measure profile amplitude and sigma before performing fit of amplitude(x) and sigma(x)
        self.profile_order = 4 #order of polynomial used to fit profile amplitude and sigma as functions of pixel along dispersion direction
        self.scatteredlightcorr_order = 4
        self.scatteredlightcorr_rejection_iterations = 10
        self.scatteredlightcorr_rejection_sigma = 3.
        self.extract1d_aperture_width = 3. # 3. #maximum (half-)width of aperture for extraction (too large and we get weird edge effects)
        self.resolution_order = 1
        self.resolution_rejection_iteration = 10
        self.id_lines_continuum_rejection_sigma = 3.
        self.id_lines_tol_angs = 0.02 # 0.005 # tolerance for finding new lines to add from linelist (Angstroms)
        self.resolution_rejection_iterations = 10
        self.id_lines_continuum_rejection_low = -5.
        self.id_lines_continuum_rejection_high = 1.
        self.id_lines_continuum_rejection_iterations = 10 #number of iterations of outlier rejection for fitting "continuum"
        self.id_lines_continuum_rejection_order = 10
        self.id_lines_threshold_factor = 50 # default was 10. Threshold to id lines based on continuum?
        self.id_lines_window = 7.
        self.id_lines_order = 5
        self.id_lines_tol_pix = 2. #tolerance for matching lines between template and new spectrum (pixels)
        self.id_lines_translate_add_lines_iterations = 5
        self.use_flat = True
        self.hires_exptime = 29.
        self.id_lines_minlines_hires = 4 #25 #mininum number of ID'd lines for acceptable wavelength solution (less than this, and cannot fit reliable throughput correction and beyond)
        self.throughputcorr_continuum_rejection_low = -1.
        self.throughputcorr_continuum_rejection_high = 3.
        self.throughputcorr_continuum_rejection_iterations = 5
        self.throughputcorr_continuum_rejection_order = 4


        ## For CR rejection
        self.cr_rejection_low = -2.
        self.cr_rejection_high = 3.
        self.cr_rejection_order = 4
        self.cr_rejection_iterations = 5
        self.cr_rejection_tol = 5. #multiple of rms residual above fit to flag as CR
        self.cr_rejection_collateral = 2 #number of pixels adjacent to CR-flagged pixel to mask

        ## Really, we don't need to have this multiple times...
        self.lco=coord.EarthLocation.from_geodetic(lon=-70.6919444*u.degree,lat=-29.0158333*u.degree,height=2380.*u.meter)

        self.twilight = False

        self.binning = [2,2] ## Default binning
        ## I set the default filter to None, meaning that if we do not set the correct filter
        ## the program will stop.
        self.filter = 'NONE' ## Will ignore - _ spaces and caps 
        # self.filter = 'dupreeblue' ## Will ignore - _ spaces and caps 
        
        self.corrdark = False ## trigger to correct the dark from the science frames
        self.corr_apflat = False
        #
        self.sky_subtracted = False ## Informs us on whether the data was sky subtracted.
        self.scatteredlight_corrected = False
        self.apflat_corrected = False

    ## Setters
    def set_datapath(self, datapath):
        if not os.path.isdir(datapath):
            raise Exception("Provided or default path does not exist")
        if datapath[-1]!="/": datapath+="/"
        self.datapath = datapath

    def set_filter(self, filtername):
        myfilter = filtername.lower()
        myfilter = myfilter.replace('-', '').replace('_', '')
        myfilter = myfilter.replace(' ', '').strip()
        self.filter = myfilter

    def set_ccd(self, ccd):
        fail = False
        if ('r' in ccd) & ('b' in ccd):
            fail=True
        elif 'r' in ccd:
            self.ccd = 'r'; print('Red ccd selected')
        elif 'b' in ccd:
            self.ccd = 'b'; print('Blue ccd selected')
        else:
            fail=True
        if fail:
            raise Exception('Could not understand the selected CCD')

    def set_filename(self, datafile):
        if datafile is not None:
            self.filename = self.datapath+datafile
            if not os.path.isfile(self.filename):
                raise Exception("Input file does not exist")
        else:
            raise IOError("Please provide a 'datafile' input file name.")

    def set_binning(self, dims):
        self.binning = dims

    def check_data(self):
        '''Function to check that the files input frames exist, are not corrupted, 
        and that their dimensions match. Also updates the binning attributes to that
        the program can automatically handle different binnings.'''
        inidims=[0,0]
        ccd = self.ccd
        for tile in range(1, 5):
            for fileoro in self.full_list:
                _fname = self.rawfiledict[(ccd, tile, fileoro)]
                if not os.path.isfile(_fname):
                    raise Exception('File not found: '+_fname)
                try:
                    data = fits.getdata(_fname)
                except:
                    raise Exception('Could not read {}'.format(_fname))
                datadims = np.shape(data)
                ndims = [int(round(datadims[0]/1000)), int(round(datadims[1]/1000))]
                if inidims==[0,0]: inidims = ndims
                #
                if ndims != inidims:
                    raise Exception('Binning inconsistent for file: '+_fname)
        #
        print("Binning detected to be: {}".format(inidims))
        self.set_binning(np.array(inidims))
        _fname = self.rawfiledict[(ccd, tile, self.sci_list[0])]
        filter = fits.getheader(_fname)['filter'].replace('_', '').lower()
        print("Filter detected to be: {}".format(filter))
        self.set_filter(filter)
    
    def read_data(self):
        if self.filename is None:
            raise IOError("No file. Please provide a 'datafile' input file name.")
        data = astropy.nddata.CCDData.read(self.filename)
        self.data = data

    def set_utdate(self, utdate):
        self.utdate = utdate

    def gen_filelists(self, filename):
        '''This function is meant to create the file lists based on one single 
        input text file.
        The file lists must be exaustive and will be stored in dictionaries.'''
        # utdate = self.utdate
        f = open(filename, 'r')
        list_dict = {}
        for line in f.readlines():
            # sci_list = []
            # arc_list = []
            # led_list = []
            # bias_list = []
            # dark_list = []
            if line[0]=="#": continue ## This is a comment
            if line.strip()=="": continue ## Skip empty lines
            sl = line.split()
            utdate = sl[0].strip()
            list_dict[utdate] = {}
            ## Get the science
            sciliststr = sl[1].split("-")
            sci_list = np.array([int(float(sciliststr[i])) for i in range(len(sciliststr))])
            list_dict[utdate]['sci_list'] = sci_list
            ## Get the ThAr
            arcliststr = sl[2].split("-")
            arc_list = np.array([int(float(arcliststr[i])) for i in range(len(arcliststr))])
            list_dict[utdate]['arc_list'] = arc_list
            ## Get the led
            ledliststr = sl[3].split("-")
            led_list = np.array([int(float(ledliststr[i])) for i in range(len(ledliststr))])
            list_dict[utdate]['led_list'] = led_list
            ## Get the bias
            # if "," in sl[4]:
            #     biasliststr = sl[4].split(",")
            #     bias_list = np.arange(int(float(biasliststr[0])), int(float(biasliststr[1])))
            # else:
            #     biasliststr = sl[4].split("-")
            #     bias_list = np.array([int(float(biasliststr[i])) for i in range(len(biasliststr))])
            #     list_dict[utdate]['bias_list'] = bias_list
            # list_dict[utdate]['bias_list'] = bias_list
            if "-" in sl[4]:
                biasliststr = sl[4].split("-")
                strlist = []
                for substr in biasliststr:
                    if ',' in substr:
                        subbiasliststr = substr.split(",")
                        subarr = np.arange(int(float(subbiasliststr[0])), int(float(subbiasliststr[1])))  
                    else:
                        subarr = np.array([int(float(substr))])
                    strlist.append(subarr)
                ## Concatenate the sublists
                bias_list = np.concatenate(strlist)
            elif "," in sl[4]:
                biasliststr = sl[4].split(",")
                bias_list = np.arange(int(float(biasliststr[0])), int(float(biasliststr[1]))+1)
            else:
                biasliststr = sl[4].split("-")
                bias_list = np.array([int(float(biasliststr[i])) for i in range(len(biasliststr))])
            list_dict[utdate]['bias_list'] = bias_list
            ## Get the dark
            if "-" in sl[5]:
                darkliststr = sl[5].split("-")
                strlist = []
                for substr in darkliststr:
                    if ',' in substr:
                        subdarkliststr = substr.split(",")
                        subarr = np.arange(int(float(subdarkliststr[0])), int(float(subdarkliststr[1]))+1)  
                    else:
                        subarr = np.array([int(float(substr))])
                    strlist.append(subarr)
                ## Concatenate the sublists
                dark_list = np.concatenate(strlist)
            elif "," in sl[5]:
                darkliststr = sl[5].split(",")
                dark_list = np.arange(int(float(darkliststr[0])), int(float(darkliststr[1]))+1)
            else:
                darkliststr = sl[5].split("-")
                dark_list = np.array([int(float(darkliststr[i])) for i in range(len(darkliststr))])
                # list_dict[utdate]['dark_list'] = dark_list
            list_dict[utdate]['dark_list'] = dark_list
            ## if given, grab the reference LED
            if len(sl)>6:
                led_ref = int(float(sl[6]))
            else:
                led_ref = led_list[0]
            ## if given, grab the reference ARC
            if len(sl)>7:
                arc_ref = int(float(sl[7]))
            else:
                arc_ref = arc_list[0]
            list_dict[utdate]['led_ref'] = led_ref
            list_dict[utdate]['arc_ref'] = arc_ref
        f.close()

        if self.utdate is None:
            self.utdate = list_dict.keys()[0] ## Default is the first one
        utdate = self.utdate # Safeguard because my implementation is crappy.
        led_ref = list_dict[self.utdate]['led_ref']
        arc_ref = list_dict[self.utdate]['arc_ref']
        self.sci_list = list_dict[self.utdate]['sci_list']
        self.arc_list = list_dict[self.utdate]['arc_list']
        self.led_list = list_dict[self.utdate]['led_list']
        self.bias_list = list_dict[self.utdate]['bias_list']
        self.dark_list = list_dict[self.utdate]['dark_list']
        self.all_list = np.concatenate([self.led_list, self.sci_list, self.arc_list], 0)
        self.full_list  = np.concatenate([self.sci_list, self.arc_list, self.led_list, self.bias_list, self.dark_list], 0)
        self.led_ref = led_ref
        self.arc_ref = arc_ref

        self.rawfiledict = {} ## Contains all the path to all files, calibration and other.
        for ccd in ['r', 'b']:
            for tile in range(1, 5):
                for fileoro in self.all_list: ## Not the biases!
                    self.rawfiledict[(ccd, tile, fileoro)] = self.inpath \
                                    + "{}{:04d}c{}.fits".format(ccd, fileoro, tile)
                for fileoro in self.dark_list: ## Not the biases!
                    self.rawfiledict[(ccd, tile, fileoro)] = self.darkpath \
                                    + "{}{:04d}c{}.fits".format(ccd, fileoro, tile)
                for fileoro in self.bias_list: ## Not the biases!
                    self.rawfiledict[(ccd, tile, fileoro)] = self.biaspath \
                                    + "{}{:04d}c{}.fits".format(ccd, fileoro, tile)

        self.masterbiasframes = {}
        for ccd in ['r', 'b']:
            for tile in range(1, 5):
                self.masterbiasframes[(ccd, tile)] = self.tmppath \
                                                 + "{}-{}-master-bias.fits".format(ccd, tile)
        self.masterdarkframes = {}
        for ccd in ['r', 'b']:
            for tile in range(1, 5):
                self.masterdarkframes[(ccd, tile)] = self.tmppath \
                                                 + "{}-{}-master-dark.fits".format(ccd, tile)
        self.filedict = {}
        for ccd in ['r', 'b']:
            for file_id in self.all_list:
                self.filedict[(ccd, file_id)] = self.tmppath \
                                                   + "{}{:04d}-stitched.fits".format(ccd, file_id)

        self.apflatcorr_files = {}
        self.scatteredlightcorr_files = {}
        self.extract1d_array_files = {}
        self.extract1d_flatfield_array_files = {}
        self.wavcal_array_files = {}
        for ccd in ['r', 'b']:    
            for file_id in self.all_list:
                self.apflatcorr_files[(ccd, file_id)] = self.outpath \
                                    + "{}{:04d}-apflatcorr.pickle".format(ccd, file_id)
                self.scatteredlightcorr_files[(ccd, file_id)] = self.outpath \
                        + "{}{:04d}-scatteredlightcorr.pickle".format(ccd, file_id)
                self.extract1d_array_files[(ccd, file_id)] =  self.outpath \
                    + "{}{:04d}-extract1d-array.pickle".format(ccd, file_id)
                self.extract1d_flatfield_array_files[(ccd, file_id)] = self.outpath \
                        + "{}{:04d}-extract1d-flatfield-array.pickle".format(ccd, file_id)
                self.wavcal_array_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-wavcal-array.pickle".format(ccd, file_id)


        self.id_lines_template_files = {}
        self.id_lines_array_files = {}
        for ccd in ['r', 'b']:    
            for file_id in self.arc_list:
                self.id_lines_template_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-id-lines-template.pickle".format(ccd, file_id)
                self.id_lines_array_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-id-lines-array.pickle".format(ccd, file_id)

        self.cr_reject_array_files = {}
        self.sky_array_files = {}
        self.skysubtract_array_files = {}
        self.continuum_array_files = {}
        for ccd in ['r', 'b']:    
            for file_id in self.sci_list:
                self.cr_reject_array_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-cr-reject-array.pickle".format(ccd, file_id)
                self.sky_array_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-sky-array.pickle".format(ccd, file_id)
                self.skysubtract_array_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-skysubtract-array.pickle".format(ccd, file_id)
                self.continuum_array_files[(ccd, file_id)] = self.outpath \
                    + "{}{:04d}-continuum-array.pickle".format(ccd, file_id)
       
        self.stack_array_files = {}
        self.stack_wavcal_array_files = {}
        self.stack_array_files_nosky = {}
        self.stack_array_files_flat_nosky = {}
        self.ref_array_files_flat = {}
        self.stack_wavcal_array_files_nosky = {}
        for ccd in ['r', 'b']:    
            self.stack_array_files[ccd] = self.outpath + "{}-stack-array-skysubtract.pickle".format(ccd)
            self.stack_array_files_nosky[ccd] = self.outpath + "{}-stack-array.pickle".format(ccd)
            self.stack_array_files_flat_nosky[ccd] = self.outpath + "{}-stack-flat-array.pickle".format(ccd)
            self.ref_array_files_flat[ccd] = self.outpath +  "{}-flat-array.pickle".format(ccd)
            self.stack_wavcal_array_files[ccd] = self.outpath \
                + "{}-stack-wavcal-array-skysubtract.pickle".format(ccd)
            self.stack_wavcal_array_files_nosky[ccd] = self.outpath \
                + "{}-stack-wavcal-array.pickle".format(ccd)

        ## Generate the output paths
        self.plugmap_file = self.tmppath \
                            + "{}{:04d}-plugmap.pickle".format(self.ccd, self.led_ref)
        self.plugmap_txtfile = self.tmppath \
                            + "{}-plugmap.txt".format(self.ccd)
        self.image_boundary_file = self.outpath \
                            + "{}{:04d}-image-boundary.pickle".format(self.ccd, self.led_ref)
        self.columnspec_array_file = self.tmppath \
                            + "{}{:04d}-columnspec-array.pickle".format(self.ccd, self.led_ref)
        self.apertures_profile_middle_file = self.outpath \
                + "{}{:04d}-apertures-profile-middle.pickle".format(self.ccd, self.led_ref)
        self.aperture_array_file =  self.outpath \
                + "{}{:04d}-aperture-array.pickle".format(self.ccd, self.led_ref)
        self.apmask_file =  self.outpath \
                + "{}{:04d}-apmask.pickle".format(self.ccd, self.led_ref)
        self.apmask2_file =  self.outpath \
                + "{}{:04d}-apmask2.pickle".format(self.ccd, self.led_ref)
        self.apflat_file =  self.outpath \
                + "{}{:04d}-apflat.pickle".format(self.ccd, self.led_ref)
        self.apflat_residual_file =  self.outpath \
                + "{}{:04d}-apflat-residual.pickle".format(self.ccd, self.led_ref)
        ## And now the results fits filenames
        self.fits_files = {}
        self.fits_stack = {}
        self.extract1d_fits_files = {}
        self.extract1d_fits_stack = {}
        for ccd in ['r', 'b']:
            for file_id in self.sci_list:
                self.fits_files[(self.utdate, ccd, file_id)] = self.outpath \
                    + "{}-{}{:04d}-results-skysubtract.fits".format(self.utdate, ccd, file_id)
                self.extract1d_fits_files[(self.utdate, ccd, file_id)] = self.outpath \
                    + "{}-{}{:04d}-results.fits".format(self.utdate, ccd, file_id)
            for file_id in [self.led_ref]: ## So we can store the flat for correction
                self.extract1d_fits_files[(self.utdate, ccd, file_id)] = self.outpath \
                    + "{}-{}-flat-results.fits".format(self.utdate, ccd)
            self.fits_stack[(self.utdate, ccd)] = self.outpath \
                    + "{}-{}-results-skysubtract.fits".format(self.utdate, ccd)
            self.extract1d_fits_stack[(self.utdate, ccd)] = self.outpath \
                    + "{}-{}-results.fits".format(self.utdate, ccd)
        ## Now generate some files meant to accelerate the program
        self.idlinestemplate_datafile = self.outpath + 'tmp-idlinesdatafile.pickle'

    ########################################
    #### PRE REDUCTION (FRAME STICHING) ####
    ## --                              -- ##
    def zero_corr(self):
        for tile in range(1, 5):
            fdump.zero_corr(self.rawfiledict, self.ccd, tile, self.bias_list, 
                            self.masterbiasframes[(self.ccd, tile)], self.binning)
    def dark_corr(self):
        for tile in range(1, 5):
            fdump.dark_corr(self.rawfiledict, self.ccd, tile, self.dark_list,
                            self.masterbiasframes[(self.ccd, tile)], 
                            self.masterdarkframes[(self.ccd, tile)], self.binning, self.corrbias)  
    def stitch_frames(self):
        ## For each file name, and for the global ccd, we combine the tiles.
        for file_id in self.all_list:
            fdump.stitch_frames(self.ccd, file_id, self.rawfiledict, self.masterbiasframes, 
                                self.masterdarkframes, self.filedict, self.binning, self.corrdark, self.corrbias) 
    def stitch_frames_nocorr(self):
        ## For each file name, and for the global ccd, we combine the tiles.
        for file_id in self.all_list:
            fdump.stitch_frames_nocorr(self.ccd, file_id, self.rawfiledict, self.masterbiasframes, 
                                self.masterdarkframes, self.filedict, self.binning)    
    ## --                              -- ##
    ########################################
    ########################################

    def gen_plugmap(self, filename=None, overwrite=False):
        '''Function to read the fits header a build a plug map.
        The function will check all the fits headers for inconsistencies.
        For now we only check the ThAr, LED, and Science frames.
        '''
        if filename is not None:
            filename = filename.strip()
        nbfibers = 128 # this is the number of fibers we should have. TODO make this an attribute
        # statuses = []
        # headers = []
        # objtype = []
        # for file_id in self.all_list:
        #     filename = self.filedict[(self.ccd, file_id)]
        #     _statuses, _headers, _objtype = fdump.gen_plugmap(filename)
        #     if len(_statuses)!=nbfibers:
        #         raise Exception("gen_plugmap: Cannot ID 128 fibers?")
        #     statuses.append(_statuses)
        #     headers.append(_headers)
        #     objtype.append(_objtype)
        # ## Check that we have the same thing for all files:
        # for i in range(nbfibers):
        #     _stat = statuses[0][i] ## First file status
        #     _head = headers[0][i]
        #     _obj = objtype[0][i]
        #     for j in range(1, len(self.all_list)):
        #         if statuses[j][i]!=_stat:
        #             raise Exception('gen_plugmap: inconsistency in status headers')
        #         if headers[j][i]!=_head:
        #             raise Exception('gen_plugmap: inconsistency in status headers')
        #         if objtype[j][i]!=_obj:
        #             raise Exception('gen_plugmap: inconsistency in status headers')
        #
        fdump.check_headers(nbfibers, self.ccd, self.all_list, self.filedict)
        # identifiers, headers, objtype = fdump.gen_plugmap(self.filedict[(self.ccd, self.all_list[0])])
        if filename is not None:
            plugmapdic = fdump.gen_plugmap_from_file(filename, self.ccd)
        else:
            plugmapdic = fdump.gen_plugmap(self.filedict[(self.ccd, self.sci_list[0])])

        apertures, identifiers, objtypes, fibers = fdump.order_fibers(plugmapdic, self.ccd, self.filter)

        objcol = fits.Column(name='OBJTYPE',format='A6',array=objtypes)
        apcol = fits.Column(name='APERTURE',format='I',array=apertures)
        fibcol = fits.Column(name='FIBERS',format='A8',array=fibers)
        idcol = fits.Column(name='IDENTIFIERS',format='A10',array=identifiers)
        cols=fits.ColDefs([objcol, apcol, fibcol, idcol])
        plugmap_table_hdu=fits.FITS_rec.from_columns(cols)

        # ## We then take the first file headers: 
        # objtype
        # identifiers = statuses[0]
        # from IPython import embed
        # embed()

        # objtype = np.array(objtype)[::-1]
        # identifiers = np.array(identifiers)[::-1]
        # nobjtype = np.copy(objtype)
        # for i in range(1, len(objtype)):
        #     if (objtype[i]=='unused') & (objtype[i-1]!='unused'):
        #         nobjtype[i] = objtype[i-1]
        # objtype = nobjtype
        # objcol = fits.Column(name='OBJTYPE',format='A6',array=objtype)
        # apcol = fits.Column(name='APERTURE',format='I',array=np.arange(nbfibers)+1)
        # expid = fits.Column(name='EXPID',format='A100',array=identifiers)
        # cols=fits.ColDefs([expid, objcol, apcol])
        # plugmap_table_hdu=fits.FITS_rec.from_columns(cols)

        pickle.dump(plugmap_table_hdu, open(self.plugmap_file,'wb'))

        writefile = True
        if path.exists(self.plugmap_txtfile):
            #
            print(self.plugmap_txtfile + ' already exsists')
            val = input("Do you really want to overwrite it? [y/n]")
            if val.lower()=='y':
                ## Make a copy of the file
                import shutil
                shutil.copyfile(self.plugmap_txtfile, self.plugmap_txtfile+'_bckp')
                writefile = True
            else:
                writefile = False
        
        ## Write the file
        if writefile:
            plugstring = ""
            for i in range(len(objtypes)):
                plugstring += "{:2} {} {:10} {:10}\n".format(i+1, fibers[i], objtypes[i], identifiers[i])
            f = open(self.plugmap_txtfile, 'w')
            f.write(plugstring)
            f.close()

        return plugmap_table_hdu

    def gen_plugmap2(self):
        '''This function was mainly to test the gen_plugmap function 
        and check that the results are the same'''
        fibermap_file = "/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-data/nov-2019/Dupree_MCClusters_2019B-NGC_330-46.fibermap"
        for file_id in self.all_list:
            data_file = self.filedict[(self.ccd, file_id)]
        # if (('twilight' not in field_name[i])&('standard' not in field_name[i])):
            # for j in range(0,len(scifile0[i])):
            cr_reject_array_file = self.tmppath + "{}{:04d}-cr-reject-array.pickle".format(self.ccd, file_id)
            throughput_array_file = self.tmppath + "{}{:04d}-throughput-array.pickle".format(self.ccd, file_id)
            throughputcorr_array_file = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)
            plugmap_file = self.tmppath + "{}{:04d}-plugmap-array.pickle".format(self.ccd, file_id)
            id_lines_array_file = self.tmppath + "{}{:04d}-id-lines-array.pickle".format(self.ccd, file_id)

            plugmap_exists=path.exists(plugmap_file)
            if (not(plugmap_exists))|(plugmap_exists & self.overwrite):
                if self.overwrite:
                    print('will overwrite existing version')
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                throughputcorr_array=pickle.load(open(throughputcorr_array_file,'rb'))
                # throughput_array=pickle.load(open(throughput_array_file,'rb'))
                # cr_reject_array=pickle.load(open(cr_reject_array_file,'rb'))
                with open(fibermap_file) as f:
                    fibermap=f.readlines()

                plugmap0=m2fs.get_plugmap(data.header,throughputcorr_array,fibermap,self.ccd,"none")
                pickle.dump(plugmap0,open(plugmap_file,'wb'))
                # id_lines_array=pickle.load(open(id_lines_array_file,'rb'))

                skies=np.where(((plugmap0['objtype']=='SKY')|(plugmap0['objtype']=='unused')))[0]
                targets=np.where(plugmap0['objtype']=='TARGET')[0]
            # from IPython import embed
            # embed()
        
    def check_plugmap(self):
        
        plugmap = pickle.load(open(self.plugmap_file, 'rb'))
        apertures = plugmap['APERTURE']
        objtypes = plugmap['OBJTYPE']
        ids = plugmap['IDENTIFIERS']
        fibers = plugmap['FIBERS']

        # options = [fibers[i] for i in range(len(fibers))]
        myccd = self.ccd.upper()
        options = []
        for cassette in range(1, 9):
            for fiber in range(1, 17):                    
                options.append("FIBER{}{:02d}".format(cassette, fiber))
        texts = {}
        for ii, fiber in enumerate(fibers):
            texts[fiber] = fiber + " - " + ids[ii].upper()
        for option in options:
            if option not in fibers:
                texts[option] = option + " - UNPLUG"
        import tkinter as tk

        def show_selection():
            selected_options = [option for option, value in checkboxes.items() if value.get()]
            num_checked_boxes.set(f"Number selected fibers: {len(selected_options)}\n"
                                  + "Filter: {}".format(self.filter))
            print("Selected options:", selected_options)
        
        def continue_execution():
            # Add your desired code here to continue the execution after breaking the main loop
            window.destroy()
            print("Continuing the code execution...")

        # Create the main window
        window = tk.Tk()

        # Set the title of the window
        window.title("Fibers selection")

        # Create a dictionary to store the checkbutton variables
        checkboxes = {}

        # Create the multiple choice checkbuttons
        # options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5", "Option 6", "Option 7", "Option 8", "Option 9", "Option 10"]
        j=0
        counter = 0
        offset = 0
        for i, option_text in enumerate(options):
            checkboxes[option_text] = tk.BooleanVar(value=False)  # Set default value to False
            checkbox = tk.Checkbutton(window, text=texts[option_text], variable=checkboxes[option_text])
            # checkbox.pack()
            # checkbox.grid(row=i // 5, column=0, sticky="w")
            if counter>len(options)//4: j+=1; offset+=counter; counter=0; print(j)
            counter+=1
            checkbox.grid(row=i-offset, column=j, sticky="w")

        # Set options 4, 7, and 2 to be checked by default
        for fiber in fibers:
            checkboxes[fiber].set(True)
        # checkboxes["Option 4"].set(True)
        # checkboxes["Option 7"].set(True)
        # checkboxes["Option 2"].set(True)

        # Create a button to show the selected options
        button = tk.Button(window, text="Show Selection", command=show_selection)
        # button.grid(row=(len(options) // 5) + 1, column=0, columnspan=5)
        button.grid(row=len(options) + 1, column=0, columnspan=5)
        # button.pack()

        # Create a button to break the main loop and continue the code execution
        button_continue_execution = tk.Button(window, text="Continue Execution", command=continue_execution)
        button_continue_execution.grid(row=len(options) + 2, column=0, columnspan=5)


        ## Add a label showing the total number of selected fibers
        num_checked_boxes = tk.StringVar()
        label = tk.Label(window, textvariable=num_checked_boxes)
        label.grid(row=len(options) + 3, column=0, columnspan=5)

        # Start the main event loop
        window.mainloop()


    def auto_trim(self, led_id=None):
        '''This function will launch an interactive window to trim
        the image so that the user can choose the region to analyze.
        Input: thar_ref'''
        
        if led_id is None: led_id = self.led_ref
        data_file = self.filedict[(self.ccd, led_id)]
        data=astropy.nddata.CCDData.read(data_file)
        ## work file
        image_boundary_file = self.image_boundary_file
        image_boundary_exists=path.exists(image_boundary_file) # Exists?
        if image_boundary_exists:
            print('The image boundary file already exists. Are you sure you want to overwrite?')
            cond = input('overwrite? [Yes/n]')
            if cond.lower().strip()!='yes':
                print('NOT overwriting.')
                return 1
        # #
        # if ((image_boundary_exists)&(self.overwrite)):
        #     image_boundary0=pickle.load(open(image_boundary_file,'rb'))
        #     image_boundary_fiddle=True
        #     image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
        #     print("go")
        #     pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file
        # elif (not(image_boundary_exists)):
        #     image_boundary0=m2fs.image_boundary()
        #     image_boundary_fiddle=False
        #     image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
        #     pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file

        binning = self.binning
        revbin = self.binning
        # revbin = np.abs(np.array(binning[::-1])-2)+1 ## Converts binning in factor for immage dimensions

        ## Let's hardcode some boundaries:
        x1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + 10
        x2 = np.array([2048, 2048, 2048, 2048, 2048, 
                       2048, 2048, 2048, 2048, 2048]) - 10
        x2 = x2 * revbin[1]
        y1 = np.array([0, 250, 500, 750, 1000, 
                       1250, 1520, 1780, 2056, 2056]) * revbin[0]
        y2 = np.array([0, 51,  290, 360, 790, 
                       1040, 1290, 1550, 1820, 2056]) * revbin[0]

        ## Let's hardcode some new boundaries:
        x1 = np.array([0, 0]) + 10
        x2 = np.array([2048, 2048]) - 10
        x2 = x2 * revbin[1]
        y1 = np.array([0, 2056]) * revbin[0]
        y2 = np.array([0, 2056]) * revbin[0]

        image_boundary0=m2fs.image_boundary()
        image_boundary_fiddle=False
        image_boundary = m2fs.image_boundary(lower=[(x1[q],y1[q]) 
                              for q in range(0,len(x1))],
                       upper=[(x2[q],y2[q]) 
                              for q in range(0,len(x2))])
        pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file
        return 1

    def trim(self, led_id=None):
        '''This function will launch an interactive window to trim
        the image so that the user can choose the region to analyze.
        Input: thar_ref'''
        if led_id is None: led_id = self.led_ref
        data_file = self.filedict[(self.ccd, led_id)]
        data=astropy.nddata.CCDData.read(data_file)
        ## work file
        image_boundary_file = self.image_boundary_file
        image_boundary_exists=path.exists(image_boundary_file) # Exists?
        print('{} exists? : {}'.format(image_boundary_file, image_boundary_exists))
        #
        if ((image_boundary_exists)&(self.overwrite)):
            image_boundary0=pickle.load(open(image_boundary_file,'rb'))
            image_boundary_fiddle=True
            image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
            print("go")
            pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file
        elif (not(image_boundary_exists)):
            image_boundary0=m2fs.image_boundary()
            image_boundary_fiddle=False
            image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
            pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file
        return 1

    def initialize(self):
        '''This function initializes an object called columnspec
        It is meant to perform a fit of some sort.
        columnspec creates a list of columnspec objects.
        These objects contain for each column of the image, with an applied
        normalization of the column...
        This obviously should only be applied to the FLAT exposure'''
        ## work file
        ## The columnspec_array_file should be the same for every ccd, 
        ## so long as we haven't changed the triming. It make sense to just use the file
        ## and store it in a first directory.

        led_id = self.led_ref ## Reference LED frame
        columnspec_array_file = self.columnspec_array_file
        columnspec_array_exists=path.exists(columnspec_array_file) # Exists?
        #
        if (not(columnspec_array_exists))|(columnspec_array_exists & self.overwrite):
            if columnspec_array_exists:
                print('will overwrite existing '+columnspec_array_file)
            data_file = self.filedict[(self.ccd, led_id)]
            data=astropy.nddata.CCDData.read(data_file)
            #
            columnspec_array = fdump.get_columnspec(data, self.trace_step, self.n_lines, 
                                                self.columnspec_continuum_rejection_low,
                                                self.columnspec_continuum_rejection_high,
                                                self.columnspec_continuum_rejection_iterations,
                                                self.columnspec_continuum_rejection_order, 
                                                self.threshold_factor, self.window)
            pickle.dump(columnspec_array,open(columnspec_array_file,'wb'))#save pickle to file

    def find(self, reset=False):
        '''dummy.txt is placed when a file was unused by function be required.'''

        apertures_profile_middle_file = self.apertures_profile_middle_file
        ## Load the column spec array
        columnspec_array=pickle.load(open(self.columnspec_array_file,'rb'))
        apertures_profile_middle_exists=path.exists(apertures_profile_middle_file) # Exists?

        if((apertures_profile_middle_exists)&(self.overwrite)):
            print('loading existing '+apertures_profile_middle_file+', will overwrite')
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
        elif(not(apertures_profile_middle_exists)):
            middle_column=np.long(len(columnspec_array)/2)
            apertures_profile_middle=columnspec_array[middle_column].apertures_profile
        else: 
            return 1
        
        ## This is completely dumb, but for now I have to bypass the code before in order to 
        ## set the middle column to something else. This means that I have to restart the find
        ## every time I run this function.
        # middle_column=np.long(len(columnspec_array)/2)
        print('We are working on the middle_column: {}'.format(middle_column*self.trace_step))
        # apertures_profile_middle=columnspec_array[middle_column].apertures_profile

        middle_column_arr = [middle_column]
        apertures_profile_middle=fdump.fiddle_apertures(columnspec_array, middle_column_arr,
                                                        self.window, apertures_profile_middle,
                                                        'dummy.txt')

        middle_column = middle_column_arr[0]
        # print('Now middle_column is: {}'.format(middle_column))

        if((not apertures_profile_middle_exists)|(self.overwrite)):
            print('find() -> Overwriting {}'.format(apertures_profile_middle_file))
            pickle.dump([apertures_profile_middle,middle_column],open(apertures_profile_middle_file,'wb'))#save pickle to file
                
            # from IPython import embed
            # embed()
            realvirtualarray = np.array(apertures_profile_middle.realvirtual)
            f = open(self.outpath + 'real-virtual-array.txt', 'w')
            for i in range(len(realvirtualarray)):
                f.write("{}\n".format(int(float(realvirtualarray[i]))))
            f.close()
        else:
            print('find() -> No overwriting.')

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

        return 0

    def trace_all(self):
        aperture_array_file = self.aperture_array_file
        image_boundary_file = self.image_boundary_file
        columnspec_array_file = self.columnspec_array_file
        apertures_profile_middle_file = self.apertures_profile_middle_file
        aperture_array_exists=path.exists(aperture_array_file) # Exists?

        print('tracing all apertures for '+str(self.led_ref))
        if (not(aperture_array_exists))|(aperture_array_exists & self.overwrite):
            if aperture_array_exists:
                print('will overwrite existing '+aperture_array_file)
            image_boundary=pickle.load(open(image_boundary_file,'rb'))
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
        
            aperture_array=[]
            for j in range(0,len(apertures_profile_middle.fit)):
#            if apertures_profile_middle.realvirtual[j]:
                aperture_array.append(fdump.get_aperture_fast(j,columnspec_array,
                                                        apertures_profile_middle,
                                                        middle_column,
                                                        self.trace_order,
                                                        self.trace_rejection_sigma,
                                                        self.trace_rejection_iterations,
                                                        image_boundary,
                                                        self.trace_shift_max,self.trace_nlost_max,
                                                        self.profile_rejection_iterations,self.profile_nsample,
                                                        self.profile_order,self.window))

            pickle.dump(aperture_array,open(aperture_array_file,'wb'))

    def check_trace(self):
        aperture_array_file = self.aperture_array_file
        image_boundary_file = self.image_boundary_file
        columnspec_array_file = self.columnspec_array_file
        apertures_profile_middle_file = self.apertures_profile_middle_file
        aperture_array_exists=path.exists(aperture_array_file) # Exists?
        myframe = self.sci_list[0] # self.led_ref
        myframe = self.led_list[0] # self.led_ref
        flat_file = self.filedict[(self.ccd, myframe)]
        print("Plotting {}".format(flat_file))
        if aperture_array_exists:
            aperture_array = pickle.load(open(aperture_array_file,'rb'))
            image = fits.getdata(flat_file)
            aps = []
            for i in range(len(aperture_array)):
                aps.append(aperture_array[i].trace_func(np.arange(len(image[0]))))
            plt.figure()
            plt.imshow(image, vmin=0, vmax=np.nanpercentile(image, 90))
            for i in range(len(aps)):
                plt.plot(aps[i], color='red', linewidth=0.3)
            plt.show()

    def apmask(self):
        apmask_file = self.apmask_file
        apmask2_file = self.apmask2_file
        apertures_profile_middle_file = self.apertures_profile_middle_file
        aperture_array_file = self.aperture_array_file
        image_boundary_file = self.image_boundary_file
        data_file = self.filedict[(self.ccd, self.led_ref)]

        apmask_exists = path.exists(apmask_file)
        print('making aperture mask for ',str(self.led_ref))
        if (not(apmask_exists))|(apmask_exists & self.overwrite):
            if self.overwrite:
                print('will overwrite existing version')
                
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))
            aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
            image_boundary=pickle.load(open(image_boundary_file,'rb'))
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows

            apmask0, apmask2=fdump.get_apmask(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary)
            # apmask2=fdump.get_apmask2(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary)
            # apmask0[:] = False
            # from IPython import embed
            # embed()    
            plt.figure()
            yyy = apmask0
            plt.imshow(yyy)#, vmin=np.nanpercentile(yyy, 10), vmax=np.nanpercentile(yyy, 90))
            plt.show()

            pickle.dump(apmask0,open(apmask_file,'wb'))
            pickle.dump(apmask2,open(apmask2_file,'wb'))
        print('apmask done')
    def apflat(self):
        '''Hardcode the name of the field as dummy-led for now.
        In the code the comments checks whether the name is twilight. 
        If it is, it changes a little bit, saying "this doesn't work if 
        it's own twilight". Would that be a specific case where we chose the
        LED to be same as the twilight? In the current setup, I do not see how that
        would be possible here. '''

        apflat_file = self.apflat_file
        apflat_residual_file = self.apflat_residual_file
        apertures_profile_middle_file = self.apertures_profile_middle_file
        aperture_array_file = self.aperture_array_file
        image_boundary_file = self.image_boundary_file
        apmask_file = self.apmask_file
        data_file = self.filedict[(self.ccd, self.led_ref)]

        if self.twilight:
            ## The program apparently checks whther the twilight is its own flat field. 
            ## I simply assume that this is not the case.
            ## TODO: Update this more inteligently
            pass

        apflat_exists = path.exists(apflat_file)
        if (not(apflat_exists))|(apflat_exists & self.overwrite):
            if self.overwrite:
                print('will overwrite existing version')
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))
            aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
            image_boundary=pickle.load(open(image_boundary_file,'rb'))
            apmask0=pickle.load(open(apmask_file,'rb'))
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows

            # from IPython import embed
            # embed()    
            # plt.figure()
            # yyy = apflat_residual.data
            # plt.imshow(yyy)#, vmin=np.nanpercentile(yyy, 10), vmax=np.nanpercentile(yyy, 90))
            # plt.show()

            apflat0,apflat_residual=m2fs.get_apflat(data,aperture_array,apertures_profile_middle,
                                                    aperture_peak,image_boundary,apmask0,"dummy")
            pickle.dump(apflat0,open(apflat_file,'wb'))            
            pickle.dump(apflat_residual,open(apflat_residual_file,'wb'))

    def apflatcorr(self):
        '''Hardcode the name of the field as dummy-led for now.
        Here we may have to provide a list of names associated to the frames
        to both treat the twilight frames but avoid what should be avoided here.
        '''
        apflat_file = self.apflat_file

        allfile0 = self.all_list
        for file_id in allfile0:
            data_file = self.filedict[(self.ccd, file_id)]
            ## Outputfile?
            apflatcorr_file = self.apflatcorr_files[(self.ccd, file_id)]
            # apflatcorr_file =  self.tmppath + "{}{:04d}-apflatcorr.pickle".format(self.ccd, file_id)

            apflatcorr_exists = path.exists(apflatcorr_file)
            print('creating aperture-flat-corrected frame: \n'+apflatcorr_file)
            if (not(apflatcorr_exists))|(apflatcorr_exists & self.overwrite):
                if self.overwrite:
                    print('will overwrite existing version')
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                apflat0=pickle.load(open(apflat_file,'rb'))
                apflat0_noflat=apflat0.divide(apflat0) ## That makes 1
                # apflat0_noflat.uncertainty=None

                # if 'twilight' in "dummy-led":
                if self.twilight:
                    apflatcorr0=data.divide(apflat0_noflat)#can't flat-field twilights when we didn't take an LED, so just don't flat-field any twilights
                else:
                    if self.corr_apflat:
                        apflatcorr0=data.divide(apflat0)
                    else:
                        data_noflat = data.divide(data)
                        apflatcorr0=data.divide(data_noflat)


                # from IPython import embed
                # embed()            

                plt.figure()
                # plt.imshow(data, vmin=np.nanpercentile(data, 10), vmax=np.nanpercentile(data, 90))
                yyy = apflatcorr0.data#-
                yyy[np.isinf(yyy)] = np.nan
                # yyy = oscan_subtracted
                plt.imshow(yyy, vmin=np.nanpercentile(yyy, 10), vmax=np.nanpercentile(yyy, 90))
                # plt.imshow(oscan_subtracted)
                plt.show()


                plt.figure()
                # plt.imshow(data, vmin=np.nanpercentile(data, 10), vmax=np.nanpercentile(data, 90))
                yyy = data.data - apflatcorr0.data
                yyy[np.isinf(yyy)] = np.nan
                # yyy = oscan_subtracted
                plt.imshow(yyy, vmin=np.nanpercentile(yyy, 10), vmax=np.nanpercentile(yyy, 90))
                # plt.imshow(oscan_subtracted)
                plt.show()

                print(data_file,apflat_file,apflatcorr_file)
                pickle.dump(apflatcorr0,open(apflatcorr_file,'wb'))
            ## Now set a flag if the file now exists:
            if path.exists(apflatcorr_file):
                self.apflat_corrected = True

    def scatteredlightcorr(self):
        apmask_file = self.apmask2_file

        allfile0 = self.all_list
        for file_id in allfile0:
        # for file_id in [36]:
            print('scattered light correction for {}'.format(file_id))
            data_file = self.filedict[(self.ccd, file_id)]
            apflatcorr_file = self.apflatcorr_files[(self.ccd, file_id)]
            scatteredlightcorr_file = self.scatteredlightcorr_files[(self.ccd, file_id)]
            # apflatcorr_file =  self.tmppath + "{}{:04d}-apflatcorr.pickle".format(self.ccd, file_id)
            # scatteredlightcorr_file =  self.tmppath + "{}{:04d}-scatteredlightcorr.pickle".format(self.ccd, file_id)
            scatteredlightcorr_exists=path.exists(scatteredlightcorr_file)

            if (not(scatteredlightcorr_exists))|(scatteredlightcorr_exists & self.overwrite):
                if self.overwrite:
                    print('will overwrite existing version')
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                apmask0=pickle.load(open(apmask_file,'rb'))
                if self.apflat_corrected:
                    apflatcorr0=pickle.load(open(apflatcorr_file,'rb'))
                else:
                    apflatcorr0=astropy.nddata.CCDData.read(data_file)
                scatteredlightfunc=fdump.get_scatteredlightfunc(data,apmask0,
                                                               self.scatteredlightcorr_order,
                                                               self.scatteredlightcorr_rejection_iterations,
                                                               self.scatteredlightcorr_rejection_sigma)
                y,x=np.mgrid[:len(data.data),:len(data.data[0])]
                scattered_model=CCDData(scatteredlightfunc.func(x,y)*u.electron,
                                        mask=np.full((len(data.data),len(data.data[0])),False,dtype=bool),
                                        uncertainty=StdDevUncertainty(np.full((len(data.data),len(data.data[0])),
                                                                              scatteredlightfunc.rms,dtype='float')))
                scatteredlightcorr0=apflatcorr0.subtract(scattered_model)
                # scatteredlightcorr0=apflatcorr0

                # from IPython import embed
                # embed()    
                # plt.figure()
                # yyy = scattered_model.data
                # # yyy = data.data
                # plt.imshow(yyy, vmin=np.nanpercentile(yyy, 10), vmax=np.nanpercentile(yyy, 90))
                # plt.axhline(978, color='red')
                # plt.axvline(2000, color='red')
                # plt.colorbar()
                # plt.show()

                # plt.figure()
                # plt.plot(data.data[1500, :])
                # plt.plot(scattered_model.data[1500, :])
                # plt.show()


                # plt.figure()
                # plt.plot(data.data[978, :])
                # plt.plot(scattered_model.data[978, :])
                # plt.show()

                pickle.dump(scatteredlightcorr0,open(scatteredlightcorr_file,'wb'))  
                pickle.dump(scattered_model,open(self.outpath+'{}{}-dumpscatterlight.pickle'.format(self.ccd, file_id),'wb'))  
        self.scatteredlight_corrected = True


    def extract_1d(self, type):
        
        apertures_profile_middle_file = self.apertures_profile_middle_file
        aperture_array_file = self.aperture_array_file

        if 'sci' in type: arrlist = self.sci_list
        elif 'arc' in type: arrlist = self.arc_list
        elif 'led' in type: arrlist = self.led_list

        for file_id in arrlist:
            data_file = self.filedict[(self.ccd, file_id)]
            scatteredlightcorr_file = self.scatteredlightcorr_files[(self.ccd, file_id)]
            extract1d_array_file = self.extract1d_array_files[(self.ccd, file_id)]
            # scatteredlightcorr_file =  self.tmppath + "{}{:04d}-scatteredlightcorr.pickle".format(self.ccd, file_id)
            # extract1d_array_file =  self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)

            extract1d_array_exists = path.exists(extract1d_array_file)
            print('extracting to '+extract1d_array_file)
            if (not(extract1d_array_exists))|(extract1d_array_exists & self.overwrite):
                if self.overwrite:
                    print('will overwrite existing version')

                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                aperture_array=pickle.load(open(aperture_array_file,'rb'))
                aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                if self.scatteredlight_corrected:
                    scatteredlightcorr0=pickle.load(open(scatteredlightcorr_file,'rb'))
                elif self.apflat_corrected:
                    scatteredlightcorr0=pickle.load(open(self.apflatcorr_files[(self.ccd, file_id)],'rb'))
                else:
                    scatteredlightcorr0=astropy.nddata.CCDData.read(data_file)
                pix=np.arange(len(scatteredlightcorr0.data[0]))

                import time
                itime = time.time()
                extract1d_array=[]
                extract1d_array_old = []
                for k in range(0,len(aperture_array)):
                    # print('   extracting aperture ',aperture_array[k].trace_aperture,' of ',len(aperture_array))
                    extract=m2fs.extract1d(aperture=aperture_array[k].trace_aperture,spec1d_pixel=pix,
                                           spec1d_flux=np.full(len(pix),0.,dtype='float')*data.unit,
                                           spec1d_uncertainty=StdDevUncertainty(np.full(len(pix),999.,dtype='float')),
                                           spec1d_mask=np.full(len(pix),True,dtype='bool'))
                    extract_old = extract
                    if apertures_profile_middle.realvirtual[k]:
                        extract1d0=fdump.get_extract1d(k,scatteredlightcorr0,apertures_profile_middle,
                                                      aperture_array,aperture_peak,pix,
                                                      self.extract1d_aperture_width)
                        # from IPython import embed
                        # embed()
                        # plt.figure()
                        # plt.plot(extract1d0.spec1d_flux)
                        # plt.show()
                        ## PIC: The following was again to test that my new implementation of the extraction gives the
                        ## same results as the old function.
                        ## The new function using numba is about 30 to 80 times faster than the old one.
                        #
                        # extract1d0_old=fdump.get_extract1d_old(k,scatteredlightcorr0,apertures_profile_middle,
                        #         aperture_array,aperture_peak,pix,
                        #         self.extract1d_aperture_width)
                        
                        # pause = False
                        # if np.std((extract1d0.spec1d_flux.value/extract1d0_old.spec1d_flux.value-1))>0.001:
                        #     print('extract1d0.spec1d_flux mismatch')
                        #     pause = True
                        # if np.std((extract1d0.spec1d_pixel/extract1d0_old.spec1d_pixel-1))>0.001:
                        #     print('extract1d0.spec1d_pixel mismatch')
                        #     pause = True
                        # if np.std((extract1d0.spec1d_mask/extract1d0_old.spec1d_mask-1))>0.001:
                        #     print('extract1d0.spec1d_mask mismatch')
                        #     pause = True
                        # if np.std((extract1d0.spec1d_uncertainty.array/extract1d0_old.spec1d_uncertainty.array-1))>0.001:
                        #     print('extract1d0.spec1d_uncertainty.array mismatch')     
                        #     pause = True

                        # if pause:
                        #     from IPython import embed
                        #     embed()

                        if len(np.where(extract1d0.spec1d_mask==False)[0])>0:
                            if np.max(extract1d0.spec1d_flux)>0.:
                                extract=extract1d0
                                # extract_old = extract1d0_old
                    extract1d_array.append(extract)
                    extract1d_array_old.append(extract_old)                    
                etime = time.time()
                print("Time for extraction: {:0.2f} s".format(etime-itime))
                # from IPython import embed
                # embed()
                # if(len(extract1d_array)!=128):
                #     print('problem with extract1d_flat')
                    # np.pause()
                pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))

    def id_lines_template(self):

        '''Here again, perhaps there should be a filter name associated to each of the frame.
        For now I simply assume there is a unique line list file.'''

        linelist_file = self.linelist_file

        for file_id in [self.arc_list[0]]:
            extract1d_array_file = self.extract1d_array_files[(self.ccd, file_id)]
            id_lines_template_file = self.id_lines_template_files[(self.ccd, file_id)]
            # extract1d_array_file = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)
            # id_lines_template_file = self.tmppath + "{}{:04d}-id-lines-template.pickle".format(self.ccd, file_id)
            id_lines_template_exist=path.exists(id_lines_template_file)

            extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
            print('identify line template for '+str(file_id))

            linelist=fdump.get_linelist(linelist_file)
            if id_lines_template_exist:
                id_lines_template0=pickle.load(open(id_lines_template_file,'rb'))
                id_lines_template_fiddle=True#says that we already have a template, and will let us fiddle with that
                print('will overwrite existing version')
            else:
                id_lines_template_fiddle=False#start template from scratch
                id_lines_template0=m2fs.id_lines()#initialize values to zero-arrays
            ## TODO: Issue here due to the fact that we use the first extract1d_array, but the first could be 
            ## a virtual aperture... (or a masked one), so the mask is True everywhere.
            realvirtualarray = np.loadtxt(self.outpath+ 'real-virtual-array.txt')
            ## Let's take the middle aperture
            realap_pos = np.where(realvirtualarray>0.5)[0]
            mid = len(realap_pos)//2
            firstreal = realap_pos[mid] ## 1xN_apertures
            # print("We use the first real aperture, which is number: {}".format(firstreal))
            id_lines_template0=intacplot.get_id_lines_template(extract1d_array[firstreal],linelist,
                                                          self.id_lines_continuum_rejection_low,
                                                          self.id_lines_continuum_rejection_high,
                                                          self.id_lines_continuum_rejection_iterations,
                                                          self.id_lines_continuum_rejection_order,
                                                          self.id_lines_threshold_factor,
                                                          self.id_lines_window,self.id_lines_order,
                                                          self.id_lines_continuum_rejection_iterations,
                                                          self.id_lines_continuum_rejection_sigma,self.id_lines_tol_angs,
                                                          id_lines_template_fiddle,id_lines_template0,
                                                          self.resolution_order,self.resolution_rejection_iterations,
                                                          self.idlinestemplate_datafile)
            pickle.dump(id_lines_template0,open(id_lines_template_file,'wb'))

    def id_lines_translate(self):
        '''Here again, perhaps there should be a filter name associated to each of the frame.
        For now I simply assume there is a unique filtername "dummy-filter".
        In the pipeline by Walker, the code here checks for multiple filters and picks the "arcfilename"
        that corresponds to that filter. Now I am not 100% sure we'll need that in the future. For now,
        I just ignore it, but it should go in the questions and the TODO list.
        For now we assume we have one single template, and that is the first ThAr frame
        TODO: The program still writes and read a hardcoded file 'real-virtual-array'. This should be modified.
        '''

        linelist_file = self.linelist_file

        ## This is to identify the first real aperture.
        realvirtualarray = np.loadtxt(self.outpath+ 'real-virtual-array.txt')
        ## Let's take the middle aperture
        realap_pos = np.where(realvirtualarray>0.5)[0]
        mid = len(realap_pos)//2
        firstreal = realap_pos[mid] ## 1xN_apertures
    
        for file_id in self.arc_list:
            extract1d_array_file = self.extract1d_array_files[(self.ccd, file_id)] ## output file
            id_lines_array_file = self.id_lines_array_files[(self.ccd, file_id)] ## input file (previous id)
            # data_file = self.filedict[(self.ccd, file_id)] ## raw data file ## unused
            # arcfiltername = "dummy-filter"

            id_lines_array_exist=path.exists(id_lines_array_file)
            if (not(id_lines_array_exist))|(id_lines_array_exist & self.overwrite):
                # data=astropy.nddata.CCDData.read(data_file) ## PIC: unused
                extract1d_array=pickle.load(open(extract1d_array_file,'rb'))

                ## PIC: We just ignore all that for now. We just say "I have chosen a reference frame,
                # filtername=data.header['FILTER'] ## Filter in the data
                ## and I am using this one."
                # if filtername=='Mgb_Rev2':
                #     filtername='Mgb_HiRes'
                # arc=np.where(arcfiltername==filtername)[0][0]
                # template_root=str(arcfilename[arc])
                ## --

                ## Grab the default template file (aperture used to ID lines)
                extract1d_array_template_file = \
                    self.extract1d_array_files[(self.ccd, self.arc_list[0])]
                id_lines_template_file = self.id_lines_template_files[(self.ccd, self.arc_list[0])]
                ## Load the template data
                extract1d_array_template=pickle.load(open(extract1d_array_template_file,'rb'))
                id_lines_template0=pickle.load(open(id_lines_template_file,'rb'))
                
                print('Translating line IDs and wavelength solution from ', self.arc_list[0],' to ', file_id)
                if self.overwrite:
                    print('will overwrite existing version')

                ## PIC Here again, I assume there is a unique line list.
                linelist=fdump.get_linelist(linelist_file) ## Read the line list
                
                id_lines_array=[]
                for j in range(0,len(extract1d_array)): ## iterate through apertures.
                # for j in range(40,len(extract1d_array)): ## iterate through apertures.
                    print('working on aperture ',j)
                    id_lines_array.append(fdump.get_id_lines_translate(extract1d_array_template[firstreal],
                                                                      id_lines_template0,extract1d_array[j], linelist,
                                                                      self.id_lines_continuum_rejection_low,
                                                                      self.id_lines_continuum_rejection_high,
                                                                      self.id_lines_continuum_rejection_iterations,
                                                                      self.id_lines_continuum_rejection_order,
                                                                      self.id_lines_threshold_factor,
                                                                      self.id_lines_window,self.id_lines_order,
                                                                      self.id_lines_continuum_rejection_iterations,
                                                                      self.id_lines_continuum_rejection_sigma,
                                                                      self.id_lines_tol_angs,self.id_lines_tol_pix,
                                                                      self.resolution_order,
                                                                      self.resolution_rejection_iterations,
                                                                      self.id_lines_translate_add_lines_iterations,
                                                                      self.idlinestemplate_datafile))
                # j = 0
                # plt.figure()
                # plt.plot(extract1d_array[j].spec1d_flux/np.max(extract1d_array[j].spec1d_flux))
                # plt.plot(extract1d_array[j+2].spec1d_flux/np.max(extract1d_array[j+2].spec1d_flux))
                # plt.plot(extract1d_array[j].spec1d_mask)
                # plt.show()
                # from IPython import embed
                # embed()
                pickle.dump(id_lines_array,open(id_lines_array_file,'wb'))

    def wavcal(self):
        '''Wavelength calibration function
        Now in the origincal code, this block calls a function that actually reads
        the directory to the stiched frames and all. That is a bit uggly and impractical
        for debugging and to adapt to other paths configurations and all.
        So I will rewrite this function in the function_dump.py module.
        Here again, filters stuff that I don't really understand and therefore bypass.'''


        thar,thar_lines,thar_temperature,thar_exptime,thar_mjd=fdump.get_thar(self.filedict,
                                                                              self.ccd, self.arc_list,
                                                                              self.lco, self.use_flat,
                                                                              self.hires_exptime, 
                                                                              self.id_lines_array_files)
        for file_id in self.all_list:
            if file_id not in self.arc_list: ## We don't do this for the ThAr
                data_file = self.filedict[(self.ccd, file_id)]
                extract1d_array_file = self.extract1d_array_files[(self.ccd, file_id)]
                wavcal_array_file = self.wavcal_array_files[(self.ccd, file_id)]
                # extract1d_array_file = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)
                # wavcal_array_file = self.tmppath + "{}{:04d}-wavcal-array.pickle".format(self.ccd, file_id)

                wavcal_array_exists=path.exists(wavcal_array_file)
                print('creating wavcal_array: \n'+str(file_id))
                if (not(wavcal_array_exists))|(wavcal_array_exists & self.overwrite):
                    if self.overwrite:
                        print('will overwrite existing version')
                    
                    extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    if 'T' in data.header['DATE-OBS']:
                        time0=[data.header['DATE-OBS'],data.header['DATE-OBS'].replace(data.header['UT-TIME'], data.header['UT-END'])]
                    else:
                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                    times=time.Time(time0,location=self.lco, precision=6)
                    mjd=np.mean(times.mjd)
                    filtername=data.header['FILTER']
                    ## PIC: bypassing
                    # if filtername=='Mgb_Rev2':
                    #     filtername='Mgb_HiRes'
                    # if filtername=='Mgb_HiRes':
                    #     id_lines_minlines=id_lines_minlines_hires
                    # if filtername=='Mgb_MedRes':
                    #     id_lines_minlines=id_lines_minlines_medres
                    ## --
                    id_lines_minlines = self.id_lines_minlines_hires
                    wavcal_array=[]
                    for j in range(0,len(extract1d_array)):
                        print('working on aperture',j+1,' of ',len(extract1d_array))
                        wavcal_array.append(m2fs.get_wav(j,thar,extract1d_array,thar_mjd,mjd,id_lines_minlines))
                    pickle.dump(wavcal_array,open(wavcal_array_file,'wb'))

    def cr_reject(self):
            for file_id in self.sci_list:
                extract1d_array_file = self.extract1d_array_files[(self.ccd, file_id)]
                wavcal_array_file = self.wavcal_array_files[(self.ccd, file_id)]
                cr_reject_array_file = self.cr_reject_array_files[(self.ccd, file_id)]

                # extract1d_array_file = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)
                # wavcal_array_file = self.tmppath + "{}{:04d}-wavcal-array.pickle".format(self.ccd, file_id)
                # cr_reject_array_file = self.tmppath + "{}{:04d}-cr-reject-array.pickle".format(self.ccd, file_id)

                cr_reject_array_exists=path.exists(cr_reject_array_file)
                print('CR rejection for '+str(file_id))
                if (not(cr_reject_array_exists))|(cr_reject_array_exists & self.overwrite):
                    if self.overwrite:
                        print('will overwrite existing version')
                    extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                    cr_reject_array=[]
                    for k in range(0,len(extract1d_array)):
                        cr_reject_array.append(m2fs.get_cr_reject(extract1d_array[k],self.cr_rejection_low,
                                                                  self.cr_rejection_high,self.cr_rejection_order,
                                                                  self.cr_rejection_iterations, self.cr_rejection_tol,
                                                                  self.cr_rejection_collateral))
                    pickle.dump(cr_reject_array,open(cr_reject_array_file,'wb'))


    def stack_twilight(self):
        
        twilightstack_array_exists = path.exists(self.twilightstack_array_file)

        if (not(twilightstack_array_exists))|(twilightstack_array_exists & self.overwrite):
            if self.overwrite:
                print('will overwrite existing version')

            thar,thar_lines,thar_temperature,thar_exptime,thar_mjd=fdump.get_thar(self.filedict,
                                                                            self.ccd, self.arc_list,
                                                                            self.lco, self.use_flat,
                                                                            self.hires_exptime,
                                                                            self.id_lines_array_files)
                
            
            temperature=[]
            stack0=[]
            stack=[]
            mjd_mid=[]
            var_med=[]
            ## For each twilight frame (not science frame !):
            for file_id in self.twi_list:
                data_file = self.filedict[(self.ccd, file_id)]
                cr_reject_array_file = self.cr_reject_array_files[(self.ccd, file_id)]
                wavcal_array_file = self.wavcal_array_files[(self.ccd, file_id)]

                # cr_reject_array_file = self.tmppath + "{}{:04d}-cr-reject-array.pickle".format(self.ccd, file_id)
                # cr_reject_array_file = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)
                # wavcal_array_file = self.tmppath + "{}{:04d}-wavcal-array.pickle".format(self.ccd, file_id)

                wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                cr_reject_array=pickle.load(open(cr_reject_array_file,'rb'))
                data=astropy.nddata.CCDData.read(data_file)
                stack0.append(cr_reject_array)
                temperature.append(data.header['T-DOME'])
                filtername=data.header['FILTER']
                if filtername=='Mgb_Rev2':
                    filtername='Mgb_HiRes'
                if filtername=='Mgb_HiRes':
                    id_lines_minlines=id_lines_minlines_hires
                if filtername=='Mgb_MedRes':
                    id_lines_minlines=id_lines_minlines_medres
                else: ## PIC: bypassing
                    id_lines_minlines=self.id_lines_minlines_hires

                time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                times=time.Time(time0,location=self.lco,precision=6)
                mjd_mid.append(np.mean(times.mjd))
                var_med0=[]
                for k in range(0,len(cr_reject_array)):
                    var_med0.append(np.median(cr_reject_array[k].spec1d_uncertainty.quantity.value[cr_reject_array[k].spec1d_mask==False]**2))
                var_med0=np.array(var_med0)
                var_med.append(np.median(var_med0[var_med0==var_med0]))
            mjd_mid=np.array(mjd_mid)
            var_med=np.array(var_med)
            var_med[var_med!=var_med]=1.e+30
            mjd_weightedmean=np.sum(mjd_mid/var_med)/np.sum(1./var_med)
            print('stacking ',data_file,' dome temperature=',temperature[len(temperature)-1],' deg C')
            temperature=np.array(temperature)

            twilightstack_wavcal_array=[]
            for j in range(0,len(cr_reject_array)):
#                        print('twilight stackwavcal working on aperture',j+1,' of ',len(cr_reject_array))
                twilightstack_wavcal_array.append(m2fs.get_wav(j,thar,cr_reject_array,thar_mjd,mjd_weightedmean,id_lines_minlines))
#                        print(twilightstack_wavcal_array[len(twilightstack_wavcal_array)-1].wav)
#                        shite00=len(twilightstack_wavcal_array[len(twilightstack_wavcal_array)-1].wav)
#                        print(shite00)
#                        if shite==0:
#                            print('asdf')
            pickle.dump(twilightstack_wavcal_array,open(self.twilightstack_wavcal_array_file,'wb'))
            
            twilightstack_array=[]
            for j in range(0,len(stack0[0])):
                twilightstack_array.append(m2fs.get_stack(stack0,j))
            pickle.dump(twilightstack_array, open(self.twilightstack_array_file,'wb'))#save pickle to file



    def throughputcorr(self):

        ## PIC: for now I simply ignore the twilight frames
        ## The following is looking at all the twilight NIGHTS
        ## provided, and append the STACKED arrays along with the 
        ## filternames to list.
        ## For now I working only with one stacked twilight, so let's
        ## by pass that.
        ## TODO: add support to multiple twilights?
        ## Also it does something a bit dumb here: searches for the stacked
        ## twilight names, but read from stiched frames to find filtername.
        ## TODO: grab filtername earlier on and store it with the stacked twilight?
        twi_name=[]
        twi_wavcal_name=[]
        twi_mjd=[]
        twi_filtername=[]
        # for qqq in range(0,len(utdate)):
        #     if 'twilight' in field_name[qqq]:
        #         root4=datadir+utdate[qqq]+'/'+ccd+'_'+field_name[qqq]+'_'+m2fsrun
        #         wavcal0=pickle.load(open(root4+'_twilightstack_wavcal_array.pickle','rb'))
        #         data_file=datadir+utdate[qqq]+'/'+ccd+str(scifile0[qqq][0]).zfill(4)+'_stitched.fits'
        #         data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
        #         filtername=data.header['FILTER']
        #         if filtername=='Mgb_Rev2':
        #             filtername='Mgb_HiRes'
        #         if len(wavcal0)>0:
        #             twi_wavcal_name.append(root4+'_twilightstack_wavcal_array.pickle')
        #             twi_name.append(root4+'_twilightstack_array.pickle')
        #             twi_mjd.append(wavcal0[0].mjd)
        #             twi_filtername.append(filtername)

        twi_name.append(self.twilightstack_array_file)
        twi_wavcal_name.append(self.twilightstack_wavcal_array_file)
        wavcal0=pickle.load(open(twi_wavcal_name[0],'rb'))
        if len(wavcal0)>0: twi_mjd.append(wavcal0[0].mjd)
        twi_filtername.append("Halpha_Li")
        
        twi_name=np.array(twi_name)
        twi_wavcal_name=np.array(twi_wavcal_name)
        twi_mjd=np.array(twi_mjd)
        twi_filtername=np.array(twi_filtername)


        for file_id in self.sci_list: # Code was on all but ThAr and flat... That is science.                
            throughput_array_file = self.tmppath + "{}{:04d}-throughput-array.pickle".format(self.ccd, file_id)
            throughputcorr_array_file = self.tmppath + "{}{:04d}-throughputcorr-array.pickle".format(self.ccd, file_id)
            wavcal_array_file = self.tmppath + "{}{:04d}-wavcal-array.pickle".format(self.ccd, file_id)
            cr_reject_array_file = self.tmppath + "{}{:04d}-cr-reject-array.pickle".format(self.ccd, file_id)
            cr_reject_array_file = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id)

            data_file = self.filedict[(self.ccd, file_id)]
            data=astropy.nddata.CCDData.read(data_file)
            filtername=data.header['FILTER']
            if filtername=='Mgb_Rev2':
                filtername='Mgb_HiRes'
            time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
            times=time.Time(time0,location=self.lco,precision=6)
            mjd=np.mean(times.mjd)
            throughput_array_exists=path.exists(throughput_array_file)
            throughputcorr_array_exists=path.exists(throughputcorr_array_file)

            #### Need to turn these into mjd-weighted averages!!!  
            #### Also, in get_throughputcorr, need to change the penalty for not having len(wav)>100 
            #### from masking all pixels to simply not applying throughput correction
            print('creating throughputcorr_array: \n'+throughputcorr_array_file)
            if (not(throughputcorr_array_exists))|(throughputcorr_array_exists & self.overwrite):
                if self.overwrite:
                    print('will overwrite existing version')

                cr_reject_array=pickle.load(open(cr_reject_array_file,'rb'))
                wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                if len(cr_reject_array)!=len(wavcal_array):
                    print('mismatched array lengths!!!')
                    raise Exception('throughputcorr: Mismatch in array lengths')
                throughputcorr_array=cr_reject_array#initialize throughputcorr to be original cr_reject
                throughput_array=[]
                for k in range(0,len(throughputcorr_array)):
                    throughput_array.append(np.full(len(throughputcorr_array[k].spec1d_pixel),1.,dtype='float'))

                keep=np.where(twi_filtername==filtername)[0]
###########################################
                if len(keep)>0:
                    twilightstack_wavcal_array_array=[]
                    twilightstack_array_array=[]
                    twilightstack_array_array_mjd=[]
                    for k in range(0,len(keep)):
                        twilightstack_wavcal_array=pickle.load(open(twi_wavcal_name[keep[k]],'rb'))
                        twilightstack_array=pickle.load(open(twi_name[keep[k]],'rb'))
                        twilightstack_wavcal_array_array.append(twilightstack_wavcal_array)
                        twilightstack_array_array.append(twilightstack_array)
                        twilightstack_array_array_mjd.append(twi_mjd[keep[k]])

                    twilightstack_array0=[]
                    for k in range(0,len(cr_reject_array)): ## PIC: Initialization?
                        twilightstack_array0.append(
                            m2fs.extract1d(aperture=cr_reject_array[k].aperture,
                                           spec1d_pixel=cr_reject_array[k].spec1d_pixel,
                                           spec1d_flux=np.full(len(cr_reject_array[k].spec1d_pixel),0.,dtype='float'),
                                           spec1d_uncertainty=StdDevUncertainty(
                                                np.full(len(cr_reject_array[k].spec1d_pixel),999.,dtype='float')),
                                           spec1d_mask=np.full(len(cr_reject_array[k].spec1d_pixel),
                                                                True,dtype='bool')))

                    for k in range(0,len(cr_reject_array)): # For each aperture
                        ## The following line of code makes absoluetely no sense... 
                        ## It may be the source of our trouble... misplaced parentheses?
                        # if ((len(np.where(cr_reject_array[k].spec1d_mask==False)[0]>0))&(len(wavcal_array[k].wav)>0)):
                        ## I replace with:
                        # if ((len(np.where(cr_reject_array[k].spec1d_mask==False)[0])>0)&(len(wavcal_array[k].wav)>0)):
                        if 1==1:
                            # print("coucou aperture {}".format(k))
                            this=[]
                            for z in range(0,len(twilightstack_array_array)):
                                this0=np.where(np.array([twilightstack_array_array[z][zz].aperture for zz in range(0,len(twilightstack_array_array[z]))])==cr_reject_array[k].aperture)[0]
                                this.append(this0)
                            
                            for q in range(0,len(cr_reject_array[k].spec1d_flux)):
                                if cr_reject_array[k].spec1d_mask[q]==False:
                                    sum1=0.
                                    sum2=0.
                                    use=False
                                    for z in range(0,len(twilightstack_array_array)):
                                        if len(this[z])>0:
                                            if ((len(twilightstack_wavcal_array_array[z][this[z][0]].wav)>0)&(len(np.where(twilightstack_array_array[z][this[z][0]].spec1d_mask==False)[0])>0)):
                                                weight=1.
#                                                        weight=1./np.abs(twilightstack_array_array_mjd[z]-mjd_mid)
                                                # print("coucou len(this): {}".format(len(this)))
                                                if len(this)>1: ## If we have multiple twilights?
                                                    weight=np.exp(-0.5*(twilightstack_array_array_mjd[z]-mjd_mid)**2/(0.0001+np.max(twilightstack_array_array_mjd)-np.min(twilightstack_array_array_mjd))**2)#the 0.001 is there to make sure we don't divide by zero.
                                                sum1+=weight*np.interp(wavcal_array[k].wav[q],twilightstack_wavcal_array_array[z][this[z][0]].wav[twilightstack_array_array[z][this[z][0]].spec1d_mask==False],twilightstack_array_array[z][this[z][0]].spec1d_flux.value[twilightstack_array_array[z][this[z][0]].spec1d_mask==False])
                                                sum2+=weight
                                                use=True
                                            else:
                                                print('what happened here?') ## PIC: never reached
                                        else:
                                            print('what happened here?') ## PIC: never reached
                                    if use:
                                        if sum2!=1.0:
                                            print("coucou sum2: {}".format(sum2))
                                        twilightstack_array0[k].spec1d_flux[q]=sum1/sum2
                                        twilightstack_array0[k].spec1d_mask[q]=False
                                    else:
                                        print("Use set to false at this point?")
#                                                print(sum1,sum2,weight,twilightstack_array_array_mjd[z],mjd_mid)
#                                plt.plot(wavcal_array[k].wav[twilightstack_array0[k].spec1d_mask==False],twilightstack_array0[k].spec1d_flux[twilightstack_array0[k].spec1d_mask==False],lw=0.3)
#                                print(k,len(np.where(twilightstack_array0[k].spec1d_mask==False)[0]),len(wavcal_array[k].wav),'asdfasdfasdfasdf')
#                                plt.plot(wavcal_array[k].wav[twilightstack_array0[k].spec1d_mask==False],twilightstack_array0[k].spec1d_flux[twilightstack_array0[k].spec1d_mask==False])
#                                plt.show()


                    ## PIC: if I understand, with only twilight stack array here,
                    ## twilightstack_array0 should contain the same data as 
                    ## twilightstack_array
                    ## Let's check
                    ## So apparently not exactly. Now even if I bypass the above weird function
                    ## I still get a flat throughput pretty much. Is that coming from the the
                    ## Continuum adjustment?

                    twilightstack_continuum_array,throughput_continuum=\
                        fdump.get_throughput_continuum(twilightstack_array0,wavcal_array,
                                                    self.throughputcorr_continuum_rejection_low,
                                                    self.throughputcorr_continuum_rejection_high,
                                                    self.throughputcorr_continuum_rejection_iterations,
                                                    self.throughputcorr_continuum_rejection_order)
                    
#                            for k in range(0,len(twilightstack_continuum_array)):
#                                plt.plot(wavcal_array[k].wav,twilightstack_continuum_array[k](wavcal_array[k].wav),lw=1)
#                            plt.plot(wavcal_array[0].wav,throughput_continuum(wavcal_array[0].wav),lw=5,color='orange')
#                            plt.xlim([5120,5190])
#                            plt.ylim([0,35000])
#                            plt.show()
#                            plt.close()
                    # print('OKOKOKOK')
                    # from IPython import embed
                    # embed()
                    # exit()
                    # plt.figure()
                    # # plt.plot(spec_0846[1].spec1d_flux)
                    # apnb = 1
                    # plt.plot(wavcal_array[apnb].wav, twilightstack_continuum_array[apnb](wavcal_array[apnb].wav))
                    # plt.plot(wavcal_array[apnb].wav, throughput_continuum(wavcal_array[apnb].wav))
                    # # plt.plot(wavcal_array[apnb].wav, twilightstack_array0[apnb].spec1d_flux)
                    # # plt.plot(wavcal_array[apnb].wav, twilightstack_array0[apnb].spec1d_mask*15000, color='black')
                    # # plt.plot(spec_wavstack[11].wav, spec_stack[11].spec1d_flux)
                    # # plt.plot(spec_wavstack[11].wav, spec_stack[11].spec1d_mask)
                    # plt.show()

                    throughput_array=[]
                    throughputcorr_array=[]
                    for k in range(0,len(cr_reject_array)): ## For each aperture
                        print('working on ',k+1,' of ',len(cr_reject_array))
                        if len(np.where(twilightstack_array0[k].spec1d_mask==False)[0])>0:
                            # print ("wavelength length: {}".format(len(wavcal_array[k].wav)))
                            print ("twilight mask length: {}".format(len(np.where(twilightstack_array0[k].spec1d_mask==False)[0])))
                        if ((len(np.where(twilightstack_array0[k].spec1d_mask==False)[0])>0)&(len(wavcal_array[k].wav)>0)):
                            throughput,throughputcorr=m2fs.get_throughputcorr(k,throughput_continuum,cr_reject_array,twilightstack_array0,twilightstack_continuum_array,wavcal_array)
                        else:
                            throughput=np.full(len(cr_reject_array[k].spec1d_pixel),1.,dtype='float')
                            throughputcorr=cr_reject_array[k]
                        throughput_array.append(throughput)
                        throughputcorr_array.append(throughputcorr)


#                            for k in range(0,len(throughputcorr_array)):
#                                print(k,len(np.where(cr_reject_array[k].spec1d_mask==False)[0]),len(np.where(throughputcorr_array[k].spec1d_mask==False)[0]))
#                                if ((len(wavcal_array[k].wav)>0)&(len(np.where(throughputcorr_array[k].spec1d_mask==False)[0]))):
#                                    plt.plot(wavcal_array[k].wav[throughputcorr_array[k].spec1d_mask==False],throughputcorr_array[k].spec1d_flux[throughputcorr_array[k].spec1d_mask==False],lw=0.1)
#                            plt.ylim([0,35000])
#                            plt.show()
#                            plt.close()
                    pickle.dump(throughput_array,open(throughput_array_file,'wb'))
                    pickle.dump(throughputcorr_array,open(throughputcorr_array_file,'wb'))
            
                else:
                    print('WARNING: no suitable twilight for throughput correction for \n'+root0)
                    print('so not performing throughput correction')
                    cr_reject_array=pickle.load(open(cr_reject_array_file,'rb'))
                    throughputcorr_array=cr_reject_array
                    throughput_array=[]
                    
                    pickle.dump(throughput_array,open(throughput_array_file,'wb'))
                    pickle.dump(throughputcorr_array,open(throughputcorr_array_file,'wb'))

    def fix_sky(meansky):
        '''This function is meant to fix the sky (i.e.) replace part of the sky to on2, in order to
        remove features that we believe to be something else than sky (e.g. overflow from other fibers)'''
        bounds = [6565.08, 6568.47]

    def skysubtract(self):
        '''Clearly, the subtraction need to be performed after some sort of normalization, otherwise, 
        it will induce weird things. The subtract method is merely a subtraction. I believe the 
        normalization step is implemented in the throughput correction. For some reason, it looks
        like no further step is taken here. However, it could be useful to have one since apparently
        it is not necessary for us to perform the throughput correction.'''

        for file_id in self.sci_list:
            data_file = self.filedict[(self.ccd, file_id)]
            if self.throughputcorr_bool:
                infile = self.tmppath + "{}{:04d}-throughputcorr-array.pickle".format(self.ccd, file_id)
            else:
                if self.flat_field_bool:
                    infile = self.extract1d_flatfield_array_files[(self.ccd, file_id)]
                else:
                    infile = self.extract1d_array_files[(self.ccd, file_id)]
                # infile = self.tmppath + "{}{:04d}-cr-reject-array.pickle".format(self.ccd, file_id)
                # infile = self.tmppath + "{}{:04d}-extract1d-array.pickle".format(self.ccd, file_id) ## PIC I test with this cause I think the problem comes from the rejection of
                                                                                                    ## the CR whih simply also marks the lines are cosmic rays.
                

            # id_lines_array_file = self.tmppath + "{}{:04d}-id-lines-array.pickle".format(self.ccd, file_id)
            plugmap_file = self.plugmap_file
            wavcal_array_file = self.wavcal_array_files[(self.ccd, file_id)]
            # wavcal_array_file = self.tmppath + "{}{:04d}-wavcal-array.pickle".format(self.ccd, file_id)
            sky_array_file = self.sky_array_files[(self.ccd, file_id)]
            skysubtract_array_file = self.skysubtract_array_files[(self.ccd, file_id)]
            continuum_array_file = self.continuum_array_files[(self.ccd, file_id)]
            # sky_array_file = self.tmppath + "{}{:04d}-sky-array.pickle".format(self.ccd, file_id)
            # skysubtract_array_file = self.tmppath + "{}{:04d}-skysubtract-array.pickle".format(self.ccd, file_id)
            # continuum_array_file = self.tmppath + "{}{:04d}-continuum-array.pickle".format(self.ccd, file_id)

            skysubtract_array_exists=path.exists(skysubtract_array_file)

            print('creating sky-subtracted frame: \n'+str(file_id))
            if (not(skysubtract_array_exists))|(skysubtract_array_exists & self.overwrite):
                if self.overwrite:
                    print('will overwrite existing version')

                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                times=time.Time(time0,location=self.lco,precision=6)
                mjd=np.mean(times.mjd)
                filtername=data.header['FILTER']
                if filtername=='Mgb_Rev2':
                    filtername='Mgb_HiRes'
                if filtername=='Mgb_HiRes':
                    id_lines_minlines=id_lines_minlines_hires
                if filtername=='Mgb_MedRes':
                    id_lines_minlines=id_lines_minlines_medres
                else: ## PIC: That's me adapting
                    id_lines_minlines=self.id_lines_minlines_hires

                thar,thar_lines,thar_temperature,thar_exptime,thar_mjd=fdump.get_thar(self.filedict,
                                                                              self.ccd, self.arc_list,
                                                                              self.lco, self.use_flat,
                                                                              self.hires_exptime,
                                                                              self.id_lines_array_files)
                throughputcorr_array=pickle.load(open(infile,'rb')) ## PIC: if no throughputcorr this is actually the cr-reject

                plugmap0=pickle.load(open(plugmap_file,'rb'))
                wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                for i in range(1, len(plugmap0['objtype'])):
                    if plugmap0['objtype'][i]=='unused':
                        plugmap0['objtype'][i] = plugmap0['objtype'][i-1]
                # plugmap0['objtype'] = plugmap0['objtype'][::-1] 
                id_nlines=np.array([len(np.where(np.array(thar[0][q].wav)>0.)[0]) for q in range(0,len(thar[0]))],dtype='int')
                ## PIC: I believe there is an attempt to check whether for a given apertures we had enough lines to 
                ## identify the wavelength solution or not. However, id_nlines will containt information for each file
                ## not for each parameter, thus we have an inconsistency in shape...
                ## For now I bypass that by ignoring the threshold.
                # skies=np.where(((plugmap0['objtype']=='SKY')|(plugmap0['objtype']=='unused'))&(id_nlines>=id_lines_minlines))[0]
                skies=np.where(((plugmap0['objtype']=='SKY')|(plugmap0['objtype']=='unused')))[0]
                targets=np.where(plugmap0['objtype']=='TARGET')[0]

                ## PIC: Trying to only label the skies that are actually used
                # skies = skies[1::2] ## one out of 2

                ## PIC: Ok, so this block of code is just to have a default a prevent error
                ## when saving file if there are not skies.
                sky_array=[]
                for k in range(0,len(throughputcorr_array)):
                    zero=throughputcorr_array[k].spec1d_flux-throughputcorr_array[k].spec1d_flux
                    sky_array.append(m2fs.extract1d(spec1d_pixel=throughputcorr_array[k].spec1d_pixel*u.AA,spec1d_flux=zero,spec1d_mask=throughputcorr_array[k].spec1d_mask))
                skysubtract_array=throughputcorr_array#default to throughput-corrected frame, to use if there are no skies

                if len(skies)>0:
                    
                    ## PIC: Should we normalize all this?
                    # conts = []
                    # specs = []
                    # for i in range(len(throughputcorr_array)):
                    #     print(i)
                    #     from specutils.spectra import Spectrum1D
                    #     spec0=Spectrum1D(spectral_axis=throughputcorr_array[i].spec1d_pixel*u.AA,
                    #                      flux=throughputcorr_array[i].spec1d_flux,
                    #                      uncertainty=throughputcorr_array[i].spec1d_uncertainty,
                    #                      mask=throughputcorr_array[i].spec1d_mask)
                    #     continuum0,rms0=m2fs.get_continuum(spec0,-5,1,10,10)
                    #     conts.append(continuum0)
                    #     specs.append(spec0)
                    
                    # plt.figure()
                    # plt.plot(specs[1].flux)
                    # plt.plot(throughputcorr_array[1].spec1d_flux)
                    # plt.plot(conts[1](np.arange(len(specs[1].flux))))
                    # plt.show()

                    from copy import deepcopy
                    norm_throughputcorr_array = []
                    continuum = []
                    for i in range(len(throughputcorr_array)):
                        norm_throughputcorr_array.append(deepcopy(throughputcorr_array[i]))
                        c = np.ones(len(norm_throughputcorr_array[i].spec1d_flux.value))
                        continuum.append(c)
                    if self.normalize_sky:
                        continuum = []
                        for i in range(len(throughputcorr_array)):
                            ## TODO: The following is a quickfix and should be replaced ideally
                            ##       by a check that we only perform the normalization on the
                            ##       apertures that were not masked.
                            if len(wavcal_array[i].wav)==0:
                                c = 1
                            else:
                                _x, _y, c, ps = fdump.normalize(wavcal_array[i].wav, 
                                                                norm_throughputcorr_array[i].spec1d_flux.value)
                                # print(len(norm_throughputcorr_array[i].spec1d_flux), len(c))
                                norm_throughputcorr_array[i].spec1d_flux = norm_throughputcorr_array[i].spec1d_flux/c
                            continuum.append(c)
                        continuum = np.array(continuum)
                    meansky=fdump.get_meansky(norm_throughputcorr_array,wavcal_array,plugmap0)#thar,id_lines_minlines,thar_mjd,mjd,plugmap0)

                    # from IPython import embed
                    # embed()

                    # x = wavcal_array[1].wav
                    # y = throughputcorr_array[1].spec1d_flux.value
                    # yerr = throughputcorr_array[1].spec1d_flux
                    
                    # _x, _y, c, ps = fdump.normalize(x, y)

                    # plt.figure()
                    # plt.plot(x, y/c)
                    # # plt.plot(_x, _y)
                    # plt.plot(x, c)
                    # plt.plot(ps[0], ps[1])
                    # plt.show()
                    # plt.figure()
                    # for si in skies:
                    #     print(si)
                    #     if si%2==1:
                    #         plt.plot(wavcal_array[i].wav, throughputcorr_array[i].spec1d_flux/np.max(throughputcorr_array[i].spec1d_flux))
                    #         plt.plot(wavcal_array[i].wav, throughputcorr_array[i].spec1d_mask)
                    # plt.show()

                    # plt.figure()
                    # # plt.plot(meansky.flux)
                    # plt.plot(wavcal_array[15].wav, throughputcorr_array[15].spec1d_flux)
                    # plt.plot(wavcal_array[27].wav, throughputcorr_array[27].spec1d_flux)
                    # plt.plot(wavcal_array[17].wav, throughputcorr_array[17].spec1d_flux, ':')
                    # plt.plot(wavcal_array[97].wav, throughputcorr_array[97].spec1d_flux, ':')
                    # plt.plot(wavcal_array[127].wav, throughputcorr_array[127].spec1d_flux, ':')
                    # plt.show()


                    # from IPython import embed
                    # embed()
                    import warnings, sys, os

                    sky_array=[]
                    skysubtract_array=[]
                    for k in range(0,len(throughputcorr_array)):
                        ## PIC /!\ CAUTION : TESTING TO NORMALIZE EVERYTHING
                        # with warnings.catch_warnings():
                        sky,skysubtract=fdump.get_skysubtract(meansky,k,throughputcorr_array,wavcal_array)
#                            sky,skysubtract=m2fs.get_skysubtract(meansky,k,cr_reject_array,wavcal_array)
                        sky_array.append(sky)
                        skysubtract_array.append(skysubtract)

                else:
                    print('no sky fibers -- so-called sky-subtracted frame is merely the throughput-corrected frame')
                pickle.dump(sky_array,open(sky_array_file,'wb'))
                pickle.dump(skysubtract_array,open(skysubtract_array_file,'wb'))
                pickle.dump(continuum, open(continuum_array_file, 'wb'))
                self.sky_subtracted = True
                print('skysubstract: Done')

    def _stack_frames(self, kind='skycorr'):
        stack_array_file = self.stack_array_files[self.ccd]
        stack_array_exists=path.exists(stack_array_file)

        print('stacking subexposures_array \n')

        if (not(stack_array_exists))|(stack_array_exists & self.overwrite):
            if self.overwrite:
                print('will overwrite existing version')
        
            thar,thar_lines,thar_temperature,thar_exptime,thar_mjd=fdump.get_thar(self.filedict,
                                                                            self.ccd, self.arc_list,
                                                                            self.lco, self.use_flat,
                                                                            self.hires_exptime, 
                                                                            self.id_lines_array_files)
            
            temperature=[]
            stack0=[]
            stack=[]
            mjd_mid=[]
            var_med=[]
            for file_id in self.sci_list:
                data_file = self.filedict[(self.ccd, file_id)]
                skysubtract_array_file = self.skysubtract_array_files[(self.ccd, file_id)]
                # skysubtract_array_file = self.tmppath + "{}{:04d}-skysubtract-array.pickle".format(self.ccd, file_id)
                if self.throughputcorr_bool:
                    infile = self.tmppath + "{}{:04d}-throughputcorr-array.pickle".format(self.ccd, file_id)
                elif self.crreject_bool:
                    infile = self.cr_reject_array_files[(self.ccd, file_id)]
                else:
                    infile_flat = self.extract1d_flatfield_array_files[(self.ccd, file_id)]
                    infile_noflat = self.extract1d_array_files[(self.ccd, file_id)]
                    # infile = self.tmppath + "{}{:04d}-cr-reject-array.pickle".format(self.ccd, file_id)
                wavcal_array_file = self.wavcal_array_files[(self.ccd, file_id)]
                # wavcal_array_file = self.tmppath + "{}{:04d}-wavcal-array.pickle".format(self.ccd, file_id)

                wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                if 'skycorr' in kind:
                    skysubtract_array=pickle.load(open(skysubtract_array_file,'rb'))
                    stack_wavcal_array_file = self.stack_wavcal_array_files[self.ccd]
                    stack_array_file = self.stack_array_files[self.ccd]
                if 'raw' in kind:
                    skysubtract_array=pickle.load(open(infile_noflat,'rb'))
                    stack_wavcal_array_file = self.stack_wavcal_array_files_nosky[self.ccd]
                    stack_array_file = self.stack_array_files_nosky[self.ccd]
                if 'flat' in kind:
                    skysubtract_array=pickle.load(open(infile_flat,'rb'))
                    stack_wavcal_array_file = self.stack_wavcal_array_files_nosky[self.ccd]
                    stack_array_file = self.stack_array_files_flat_nosky[self.ccd]
                # throughputcorr_array=pickle.load(open(infile,'rb'))
#                    wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
#                    id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)+'_id_lines_array.pickle'
#                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                stack0.append(skysubtract_array)
                temperature.append(data.header['T-DOME'])
                filtername=data.header['FILTER']
                if filtername=='Mgb_Rev2':
                    filtername='Mgb_HiRes'
                if filtername=='Mgb_HiRes':
                    id_lines_minlines=self.id_lines_minlines_hires
                if filtername=='Mgb_MedRes':
                    id_lines_minlines=self.id_lines_minlines_medres
                else: ## PIC: this is me bypassing
                    id_lines_minlines=self.id_lines_minlines_hires
                if 'T' in data.header['DATE-OBS']:
                    time0=[data.header['DATE-OBS'],data.header['DATE-OBS'].replace(data.header['UT-TIME'], data.header['UT-END'])]
                else:
                    time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                times=time.Time(time0,location=self.lco,precision=6)
                mjd_mid.append(np.mean(times.mjd))
                var_med0=[]
                for k in range(0,len(skysubtract_array)):
                    var_med0.append(np.median(skysubtract_array[k].spec1d_uncertainty.quantity.value[skysubtract_array[k].spec1d_mask==False]**2))
                var_med0=np.array(var_med0)
                var_med.append(np.median(var_med0[var_med0==var_med0]))
            mjd_mid=np.array(mjd_mid)
            var_med=np.array(var_med)
            var_med[var_med!=var_med]=1.e+30
            mjd_weightedmean=np.sum(mjd_mid/var_med)/np.sum(1./var_med)
            print('stacking ',data_file,' dome temperature=',temperature[len(temperature)-1],' deg C')
            temperature=np.array(temperature)

            # stack_wavcal_array=[]
            # for j in range(0,len(skysubtract_array)):
            #     # print('stackwavcal working on aperture',j+1,' of ',len(skysubtract_array))
            #     stack_wavcal_array.append(m2fs.get_wav(j,thar,skysubtract_array,thar_mjd,mjd_weightedmean,id_lines_minlines))

            ## wavcal_array Should be the same as stack_wavcal_array
            stack_wavcal_array = wavcal_array
            pickle.dump(stack_wavcal_array, open(stack_wavcal_array_file,'wb'))

            stack_array=[]
            for j in range(0,len(stack0[0])):
                # stack_array.append(m2fs.get_stack(stack0,j))
                stack_array.append(fdump.get_stack(stack0,j))
            pickle.dump(stack_array, open(stack_array_file,'wb'))#save pickle to file

    def stack_frames(self):
        if self.sky_subtracted is True:
            self._stack_frames('skycorr')
        self._stack_frames('raw')
        self._stack_frames('flat')
    
    def _writefits(self, kind='sky'):
        '''Function to write the results in fits files.
        I choose to name the files based on the frame, color utdate.'''
        # if (('twilight' not in field_name[i])&('standard' not in field_name[i])):
        thar,thar_lines,thar_temperature,thar_exptime,thar_mjd=fdump.get_thar(self.filedict,
                                                                            self.ccd, self.arc_list,
                                                                            self.lco, self.use_flat,
                                                                            self.hires_exptime, 
                                                                            self.id_lines_array_files)
        ## Get the stacked files
        if 'sky' in kind:
            stack_array_file = self.stack_array_files[self.ccd]
            stack_wavcal_array_file = self.stack_wavcal_array_files[self.ccd]
        if 'raw' in kind:
            stack_array_file = self.stack_array_files_nosky[self.ccd]
            stack_wavcal_array_file = self.stack_wavcal_array_files_nosky[self.ccd]
        stack_array=pickle.load(open(stack_array_file,'rb'))
        stack_wavcal_array=pickle.load(open(stack_wavcal_array_file,'rb'))

        ## PIC: I find that the stacking can be problematic. Since it consists in
        ## taking some sort of median of the spectra, I propose to bypass this here,
        ## and rather take the median of the spectra in a post-processing step.

        ## PIC: I add this cause apparently we need it for the get_hdul function
        aperture_array_file = self.aperture_array_file
        aperture_array = pickle.load(open(aperture_array_file,'rb'))
        linelist_file = self.linelist_file
        linelist = fdump.get_linelist(linelist_file)

        temperature=[]
        for file_id in self.sci_list:
            data_file = self.filedict[(self.ccd, file_id)]
            wavcal_array_file = self.wavcal_array_files[(self.ccd, file_id)]
            if 'sky' in kind:
                skysubtract_array_file = self.skysubtract_array_files[(self.ccd, file_id)]
            if 'raw' in kind:
                if self.flat_field_bool:
                    skysubtract_array_file = self.extract1d_flatfield_array_files[(self.ccd, file_id)]
                else:
                    skysubtract_array_file = self.extract1d_array_files[(self.ccd, file_id)]
            sky_array_file = self.sky_array_files[(self.ccd, file_id)]
            plugmap_file = self.plugmap_file
            #
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            temperature.append(str(round(data.header['T-DOME'],1)))
            # wavcal_array_file=root0+'_wavcal_array.pickle'
            # skysubtract_array_file=root0+'_skysubtract_array.pickle'
            # sky_array_file=root0+'_sky_array.pickle'
            # plugmap_file=root0+'_plugmap.pickle'

            filtername=data.header['FILTER']
            if filtername=='Mgb_Rev2':
                filtername='Mgb_HiRes'

            wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
            skysubtract_array=pickle.load(open(skysubtract_array_file,'rb'))
            if 'sky' in kind:
                sky_array=pickle.load(open(sky_array_file,'rb'))
            else:
                sky_array = [skysubtract_array[i] for i in range(len(skysubtract_array))]
            plugmap0=pickle.load(open(plugmap_file,'rb'))
            m2fsrun = 'dummy'
            field_name = "dummy"
            new_hdul=fdump.get_hdul(data,skysubtract_array,sky_array,wavcal_array,
                                  plugmap0,m2fsrun,field_name,thar,
                                  [temperature[len(temperature)-1]],
                                  aperture_array, linelist)
            new_hdul[0].header['filtername']=filtername
            new_hdul[0].header['m2fsrun']=m2fsrun
            new_hdul[0].header['field_name']=field_name

            if len(skysubtract_array)>0:
                new_hdul[0].header['weighted_mjd']=str(wavcal_array[0].mjd)
                # print('writing individual frame .fits for: \n'+root0)
#                    hjdstring=str(round(new_hdul[5].data['hjd'][0],2))
                if 'sky' in kind:
                    fits_file = self.fits_files[(self.utdate, self.ccd, file_id)]
                if 'raw' in kind:
                    fits_file = self.extract1d_fits_files[(self.utdate, self.ccd, file_id)]
                # fits_file=root0+'_'+new_hdul[0].header['ut-date']+'_'+new_hdul[0].header['ut-time']+'_skysubtract.fits'
                new_hdul.writeto(fits_file,overwrite=True)

        temperature=np.array(temperature,dtype='str')

        new_hdul=fdump.get_hdul(data,stack_array,sky_array,stack_wavcal_array,
                               plugmap0,m2fsrun,field_name,thar,temperature,
                               aperture_array, linelist)
#                hjdstring=str(round(new_hdul[5].data['hjd'][0],2))
        # stack_fits_file=root2+'_'+new_hdul[0].header['ut-date']+'_'+new_hdul[0].header['ut-time']+'_stackskysub.fits'
        # stack_fits_file2=root2+'_'+new_hdul[0].header['ut-date']+'_'+new_hdul[0].header['ut-time']+'_stackskysub_file'
        if 'sky' in kind:
            stack_fits_file = self.fits_stack[(self.utdate, self.ccd)]
        if 'raw' in kind:
            stack_fits_file = self.extract1d_fits_stack[(self.utdate, self.ccd)]
        # g0=open(stack_fits_file2,'w')
        # g0.write(stack_fits_file+' '+obj[i]+' \n')
        # g0.close()
        stack_fits_exists=path.exists(stack_fits_file)
        
        if (not(stack_fits_exists))|(stack_fits_exists & self.overwrite):
            if self.overwrite:
                print('will overwrite existing version')                

            new_hdul[0].header['filtername']=filtername
            new_hdul[0].header['m2fsrun']=m2fsrun
            new_hdul[0].header['field_name']='dummy'
            if len(stack_array)>0:
                new_hdul[0].header['weighted_mjd']=str(stack_wavcal_array[0].mjd)
            ref_frame = self.arc_list[0]
            new_hdul[0].header['continuum_lamp_frame']=str(ref_frame)
            tharframe0=self.arc_list
            for q in range(0,len(tharframe0)):
                keyword='comparison_arc_frame_'+str(q)
                new_hdul[0].header[keyword]=tharframe0[q]
#                    new_hdul[0].header['comparison_arc_frame']=tharfile[i]
            # sciframe0=scifile[i].split('-')
            sciframe0 = [str(self.sci_list[i]) for i in range(len(self.sci_list))]

            keyword='sub_frames'
            s=','
            string=s.join(sciframe0)
            new_hdul[0].header[keyword]=string

            keyword='dome_temp'
            s=','
            string=s.join(temperature)
            new_hdul[0].header[keyword]=string

            # print('writing stack fits for: \n'+root2)
            print(len(new_hdul[0].data),len(new_hdul[5].data))
            if len(new_hdul[0].data)!=len(new_hdul[5].data):
                np.pause()
            new_hdul.writeto(stack_fits_file,overwrite=True)

    def _write_all(self):
        '''This function writes the output fits file for the reduction.
        The goal is to put everything in a single file for the stacked frames.
        Right now it does support nor save the sky corrected spectra.
        NB: For now I assume that there was only one flat and one calibration frame used,
        and therefore assume that they were not stacked.
        '''

        # from IPython import embed
        # embed()

        data_file = self.filedict[(self.ccd, self.sci_list[0])]
        headers = fits.getheader(data_file)

        plugmap0=pickle.load(open(self.plugmap_file,'rb'))


        #############################
        ## For the stacked spectra ##
        stack_wavcal_array_file = self.stack_wavcal_array_files_nosky[self.ccd] ## Wavelength file common to all
        stack_array_file = self.stack_array_files_nosky[self.ccd] ## File with NO sky correctoin
        stack_array_file_flat_nosky = self.stack_array_files_flat_nosky[self.ccd] ## File with NO sky AFTER flat correction
        ref_array_file_flat = self.ref_array_files_flat[self.ccd] ## File with NO sky AFTER flat correction
        arc_array_file_flat = self.extract1d_flatfield_array_files[(self.ccd, self.arc_ref)] ## Ref calibration AFTER flat correction
        arc_array_file_noflat = self.extract1d_array_files[(self.ccd, self.arc_ref)] ## Ref calibration BEFORE flat correction
        stack_wavcal_array=pickle.load(open(stack_wavcal_array_file,'rb'))
        stack_array=pickle.load(open(stack_array_file,'rb'))
        stack_array_flat=pickle.load(open(stack_array_file_flat_nosky,'rb'))
        ref_array_flat=pickle.load(open(ref_array_file_flat,'rb'))
        arc_array_flat=pickle.load(open(arc_array_file_flat,'rb'))
        arc_array_noflat=pickle.load(open(arc_array_file_noflat,'rb'))
        # Converting to numpy arrays
        naps = len(stack_array)
        npoints = len(stack_array[0].spec1d_flux.value)
        mywvl = np.empty((naps, npoints))
        myspec = np.empty((naps, npoints))
        myerror = np.empty((naps, npoints))
        myspecflat = np.empty((naps, npoints))
        myflat = np.empty((naps, npoints))
        myarc = np.empty((naps, npoints))
        myarcflat = np.empty((naps, npoints))
        for ap in range(naps):
            _wvl = stack_wavcal_array[ap].wav
            if len(_wvl)==0: _wvl = np.arange(npoints) ## To avoid shape mismatches
            mywvl[ap] = _wvl
            myspec[ap] = stack_array[ap].spec1d_flux.value
            myspecflat[ap] = stack_array_flat[ap].spec1d_flux.value
            myflat[ap] = ref_array_flat[ap]
            myerror[ap] = stack_array[ap].spec1d_uncertainty.array
            myarcflat[ap] = arc_array_flat[ap].spec1d_flux.value
            myarc[ap] = arc_array_noflat[ap].spec1d_flux.value
        #
        hdu = fits.PrimaryHDU()
        hdu.header['OBJECT'] = (headers['OBJECT'], 'object observed')
        hdu.header['UTDATE'] = (self.utdate, 'utdate directory used')
        hdu.header['FILTER'] = (self.filter, 'Filter used for observations')
        hdu.header['RWFILTER'] = (headers['FILTER'], 'raw filter name as appearing in the raw data')
        #
        hdutable = fits.BinTableHDU.from_columns([
            fits.Column(name='APERTURE', format='I', array=plugmap0['APERTURE']),
            fits.Column(name='OBJTYPE', format='A6', array=plugmap0['OBJTYPE'])
            ])
        hdutable.name = 'BINTABLE'
        #
        hdu1 = fits.ImageHDU(data=mywvl, name='WVL')
        hdu2 = fits.ImageHDU(data=myspec, name='SPEC')
        hdu3 = fits.ImageHDU(data=myspecflat, name='SPECFLAT')
        hdu4 = fits.ImageHDU(data=myflat, name='FLAT')
        hdu5 = fits.ImageHDU(data=myerror, name='ERROR')
        hdu6 = fits.ImageHDU(data=myarc, name='ARC')
        hdu7 = fits.ImageHDU(data=myarcflat, name='ARCFLAT')
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdutable])
        hdul.writeto(self.extract1d_fits_stack[(self.utdate, self.ccd)], overwrite=True)
        #############################
        
    def _writefits_flat(self, kind='flat'):
        '''Function to write the results in fits files.
        I choose to name the files based on the frame, color utdate.'''
        
        # for j in range(0,len(scifile0[i])):
        for file_id in [self.led_ref]:
            data_file = self.filedict[(self.ccd, file_id)]
            if 'flat' in kind:
                flat_array_file = self.extract1d_array_files[(self.ccd, file_id)]
            flat_array=pickle.load(open(flat_array_file,'rb'))

            outputarray = np.empty((len(flat_array), len(flat_array[0].spec1d_flux.value)))
            for ap in range(len(flat_array)):
                outputarray[ap] = flat_array[ap].spec1d_flux.value

            fits_file = self.extract1d_fits_files[(self.utdate, self.ccd, file_id)]
            
            hdu = fits.PrimaryHDU(outputarray)
            hdu.writeto(fits_file, overwrite=True)


    def writefits(self):
        # if self.sky_subtracted is True:
        #     self._writefits('sky')
        # self._writefits('raw')
        # self._writefits_flat()
        self._write_all()

    def flat_field(self):
        '''Function to correct spectra by flat spectra'''

        ## Load the apertures for the default ThAr observation
        if self.throughputcorr_bool:
            reffile = self.tmppath + "{}{:04d}-throughputcorr-array.pickle".format(self.ccd, self.led_ref)
        else:
            reffile = self.extract1d_array_files[(self.ccd, self.led_ref)]     
        led_array = pickle.load(open(reffile, 'rb'))


        # from IPython import embed
        # embed()


        # plt.figure()
        # plt.plot(led_array[3].spec1d_flux)
        # plt.show()

        # plt.figure()
        ## Get the continuum of the flats
        continua = []
        for i in range(len(led_array)):
            meanflatspec = np.nanmax(led_array[i].spec1d_flux)
            flatspec = led_array[i].spec1d_flux / meanflatspec
            smoothflatspec, c = norm_tools.moving_median(flatspec, 10, btd=0)
            c[c<0.0001] = np.nan
            continua.append(c)
        #     plt.plot(flatspec-i)
        #     plt.plot(c-i)
        # plt.show()

        # from IPython import embed
        # embed()
        pickle.dump(continua, open(self.ref_array_files_flat[self.ccd], 'wb'))

        ## Then normalize the spectra by the flat.
        mylist = [self.sci_list[i] for i in range(len(self.sci_list))]
        mylist.append(self.arc_ref)
        for file_id in mylist:
            if self.throughputcorr_bool:
                infile = self.tmppath + "{}{:04d}-throughputcorr-array.pickle".format(self.ccd, file_id)
                outfile = self.extract1d_flatfield_array_files[(self.ccd, file_id)]            
            else:
                infile = self.extract1d_array_files[(self.ccd, file_id)]            
                outfile = self.extract1d_flatfield_array_files[(self.ccd, file_id)]            

            if (not path.exists(outfile)) | (path.exists(outfile) & self.overwrite ):
                
                if (path.exists(outfile) & self.overwrite ):
                    print('will overwrite {}'.format(outfile))

                sci_array = pickle.load(open(infile, 'rb'))
                for i in range(len(sci_array)):
                    sci_array[i].spec1d_flux = sci_array[i].spec1d_flux / continua[i]
                pickle.dump(sci_array, open(outfile, 'wb'))

        self.flat_field_bool = True

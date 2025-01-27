import astropy
import astropy.units as u ## This really is a bad habit...
from astropy.nddata import NDData
from astropy.nddata import CCDData
from astropy.modeling import models
import numpy as np
import ccdproc
from ccdproc import Combiner
import warnings
from astropy import log
from copy import deepcopy ## ?
from astropy.nddata import StdDevUncertainty

import matplotlib.pyplot as plt


def zero_corr(filedict, color, ccd, bias_list, outfile, binning):
    '''This function performs the initial correction of the data and writes the new data
    to fits files. It is a revamped version of the m2fs_zero_jan20.py script.
    There are still a lot of hardcoded stuff that should really not be.'''

    # revbin = np.abs(np.array(binning[::-1])-2)+1 ## Converts binning in factor for immage dimensions
    revbin = binning
    obs_readnoise=[]
    master_processed=[]
    sig_master_processed=[]
    processed=[]
    sig_processed=[]
    ntot = len(bias_list)
    log.setLevel('WARNING') ## Only display warnings of errors
    for i, bias_id in enumerate(bias_list):
        filename = filedict[(color, ccd, bias_id)]
        ## Load the data contained in the fits file
        data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
        # print(filename,data.header['object'],data.header['binning'])

        oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,revbin[1]*1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
        trimmed1=ccdproc.trim_image(oscan_subtracted[:,:revbin[1]*1024],add_keyword={'trim1':'Done'})
        trimmed2=ccdproc.trim_image(trimmed1[:revbin[0]*1028,:revbin[1]*1024],add_keyword={'trim2':'Done'})
        array1d=trimmed2.data.flatten()
        gain=float(trimmed2.header['egain'])
        keep=np.where(np.abs(array1d)<100.)[0] #remove crazy outliers
        obs_readnoise.append(np.std(array1d[keep]*gain))
#                data_with_deviation=ccdproc.create_deviation(trimmed2,gain=data.meta['egain']*u.electron/u.adu,readnoise=data.meta['enoise']*u.electron)
#                gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                cr_cleaned=ccdproc.cosmicray_lacosmic(trimmed2,sigclip=5,gain_apply=False,gain=0.68,readnoise=2.7)
#                sig_cr_cleaned=cr_cleaned.uncertainty._array

#                data_with_deviation=ccdproc.create_deviation(data,gain=data.meta['egain']*u.electron/u.adu,readnoise=data.meta['enoise']*u.electron)
#                gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=5)
#                oscan_subtracted=ccdproc.subtract_overscan(cr_cleaned,overscan=cr_cleaned[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
#                trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
#                trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})
#                trimmed.append(trimmed2)
#                sig_trimmed.append(trimmed2.uncertainty._array)
        processed.append(trimmed2)
#                sig_processed.append(sig_cr_cleaned)
        master_processed.append(trimmed2)
#                sig_master_processed.append(sig_cr_cleaned)
#            processed=np.array(processed)
#            sig_processed=np.array(sig_processed)

#            c=Combiner(processed)
#            c.clip_extrema(nlow=1,nhigh=1)
#            old_n_masked=0
#            new_n_masked=c.data_arr.mask.sum()
#            while (new_n_masked > old_n_masked):
#                c.sigma_clipping(low_thresh=3,high_thresh=3,func=np.ma.median)
#                old_n_masked=new_n_masked
#                new_n_masked=c.data_arr.mask.sum()
#            ccdall=c.average_combine()
#            ccdall.write(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias'+str(i+1)+'.fits',overwrite=True)
        print("Preprocessing bias frames ... {:0.2f}%".format(i/ntot*100), end='\r')
    print("Preprocessing bias frames ... {:0.2f}%".format(i/ntot*100))

    obs_readnoise=np.array(obs_readnoise)
    c=Combiner(master_processed)
    c.clip_extrema(nlow=1,nhigh=1)
    old_n_masked=0
    new_n_masked=c.data_arr.mask.sum()
    print("Sigma clipping ...")
    while (new_n_masked > old_n_masked):
#            c.sigma_clipping(func=np.ma.median)
        c.sigma_clipping(low_thresh=3,high_thresh=3,func=np.ma.median)
        old_n_masked=new_n_masked
        new_n_masked=c.data_arr.mask.sum()
#        c.clip_extrema(nlow=5,nhigh=5)
#        c.weights=1./sig_master_bias**2
#        ccdall=c.average_combine(uncertainty_func=mycode.stdmean)
    print("zero_corr: Saving to {}".format(outfile.split('/')[-1]))
    ccdall=c.average_combine()
    ccdall[0].header['obs_rdnoise']=str(np.median(obs_readnoise))
    ccdall[0].header['egain']=str(gain)
    ccdall.write(outfile, overwrite=True)

def dark_corr(filedict, color, ccd, dark_list, masterbiasfile, outfile, binning, corrbias):
    '''This function performs the initial correction of the data and writes the new data
    to fits files. It is a revamped version of the m2fs_zero_jan20.py script.
    There are still a lot of hardcoded stuff that should really not be.'''

    # revbin = np.abs(np.array(binning[::-1])-2)+1 ## Converts binning in factor for immage dimensions
    revbin = binning

    if corrbias:
        master_bias=astropy.nddata.CCDData.read(masterbiasfile)
    else:
        master_bias=0

    exptime=[]
    master_debiased=[]
    master_exptime=[]
    ntot = len(dark_list)
    log.setLevel('WARNING') ## Only display warnings of errors
    for i, dark_id in enumerate(dark_list):        
        filename = filedict[(color, ccd, dark_id)]
        data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
        oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,revbin[1]*1024:],overscan_axis=1,model=models.Polynomial1D(2),add_keyword={'oscan_corr':'Done'})
        # oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[revbin[0]*1024:, :],overscan_axis=0,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})

        trimmed1=ccdproc.trim_image(oscan_subtracted[:,:revbin[1]*1024],add_keyword={'trim1':'Done'})
        trimmed2=ccdproc.trim_image(trimmed1[:revbin[0]*1028,:revbin[1]*1024],add_keyword={'trim2':'Done'})
        exptime.append(data.header['exptime'])
        master_exptime.append(data.header['exptime'])
        print('EXPOSURE TIME')
        print(data.header['exptime'])


        if corrbias:
            debiased0=ccdproc.subtract_bias(trimmed2, master_bias)
        else:
            debiased0=trimmed2

        master_debiased.append(debiased0)
        print("Preprocessing dark frames ... {:0.2f}%".format(i/ntot*100), end='\r')
    print("Preprocessing dark frames ... {:0.2f}%".format(i/ntot*100))

    master_exptime=np.array(master_exptime)
    if np.std(master_exptime)>0.:
        print('WARNING: subexposures for darks have different exposure times!!!!')
        print("{} | {}".format('file id', 'exp time'))
        for i in range(len(dark_list)):
            print("{} | {}".format(dark_list[i], master_exptime[i]))
        raise Exception('WARNING: subexposures for darks have different exposure times!!!!')
    
    c=Combiner(master_debiased)
    c.clip_extrema(nlow=1,nhigh=1)
    old_n_masked=0
    new_n_masked=c.data_arr.mask.sum()
    while (new_n_masked > old_n_masked):
        c.sigma_clipping(low_thresh=3,high_thresh=3,func=np.ma.median)
        old_n_masked=new_n_masked
        new_n_masked=c.data_arr.mask.sum()

    print("dark_corr: Saving to {}".format(outfile.split('/')[-1]))
    ccdall=c.average_combine()
    ccdall.header['exptime']=np.mean(master_exptime)
    ccdall.write(outfile, overwrite=True)

def stitch_frames_nocorr(ccd, file_id, rawfiledict, masterbiasframes, masterdarkframes, filedict, binning):
    # for filename in framelist:
    # for file_id in all_list:
    # revbin = np.abs(np.array(binning[::-1])-2)+1 ## Converts binning in factor for immage dimensions
    revbin = binning
    for tile in np.arange(1, 5):
        filename = rawfiledict[(ccd, tile, file_id)]
        masterbiasfile = masterbiasframes[(ccd, tile)]
        masterdarkfile = masterdarkframes[(ccd, tile)]
        outfile = filedict[(ccd, file_id)]

        # master_bias=astropy.nddata.CCDData.read(masterbiasfile)
        # obs_readnoise=float(master_bias.header['obs_rdnoise'])
        # master_dark=astropy.nddata.CCDData.read(masterdarkfile)

        data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
        gain=float(data.header['egain'])

        oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,revbin[1]*1024:],overscan_axis=1,model=models.Polynomial1D(1),add_keyword={'oscan_corr':'Done'})
        trimmed1=ccdproc.trim_image(oscan_subtracted[:,:revbin[1]*1024],add_keyword={'trim1':'Done'})
        trimmed2=ccdproc.trim_image(trimmed1[:revbin[0]*1028,:revbin[1]*1024],add_keyword={'trim2':'Done'})
        
        # plt.figure()
        # plt.plot(np.mean(oscan_subtracted, axis=0))
        # # plt.plot(oscan_subtracted.data[10])
        # # plt.plot(np.median(data, axis=0))
        # plt.axhline(0)
        # plt.show()
        # from IPython import embed
        # embed()

#         debiased0=ccdproc.subtract_bias(trimmed2,master_bias)
#         dedark0=ccdproc.subtract_dark(debiased0,master_dark,exposure_time='exptime',exposure_unit=u.second,scale=True,add_keyword={'dark_corr':'Done'})

#         data_with_deviation=ccdproc.create_deviation(dedark0,gain=data.meta['egain']*u.electron/u.adu,readnoise=obs_readnoise*u.electron)

#         gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
# #                master_dark_gain_corrected=ccdproc.gain_correct(master_dark,master_dark.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
# #                master_bias_gain_corrected=ccdproc.gain_correct(master_bias,master_bias.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})

#         # from IPython import embed
#         # embed()
#         gain_corrected2=deepcopy(gain_corrected)
#         ## PIC: Just for fun I bypass the corrections here:
#         # gain_corrected2.data=trimmed2.data
#         exptime_ratio=float(data.header['exptime'])/float(master_dark.meta['exptime'])

#         ntot = len(gain_corrected2.data)
#         for k in range(0,len(gain_corrected2.data)):
#             for q in range(0,len(gain_corrected2.data[k])):
#                 gain_corrected2.uncertainty.quantity.value[k][q]=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
# #                        poop1=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
# #                        poop2=(np.max(np.array([gain_corrected2.data[k][q]+2.+obs_readnoise**2,0.6*(obs_readnoise**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
# #                        print(poop1/poop2)
# #                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=10)
#             print("Uncertainty computation {:0.0f}%".format(k/ntot*100), end='\r')
#         print("Uncertainty computation {:0.0f}%".format(k/ntot*100))
# #                bad=np.where(gain_corrected.data<0.)
# #                bad=np.where(gain_corrected._uncertainty.quantity.value!=gain_corrected._uncertainty.quantity.value)#bad variances due to negative counts after overscan/bias/dark correction
# #                gain_corrected.uncertainty.quantity.value[bad]=obs_readnoise
        if 'c1' in filename:
            # c1_reduce=gain_corrected2
            c1_reduce = trimmed2
        if 'c2' in filename:
            # c2_reduce=gain_corrected2
            c2_reduce = trimmed2
        if 'c3' in filename:
            # c3_reduce=gain_corrected2
            c3_reduce = trimmed2
        if 'c4' in filename:
            # c4_reduce=gain_corrected2
            c4_reduce = trimmed2
    # from IPython import embed
    # embed()

    left_data=np.concatenate((c1_reduce,np.flipud(c4_reduce)),axis=0)#left half of stitched image
    # left_uncertainty=np.concatenate((c1_reduce.uncertainty._array,np.flipud(c4_reduce.uncertainty._array)),axis=0)
    # left_mask=np.concatenate((c1_reduce.mask,np.flipud(c4_reduce.mask)),axis=0)
    right_data=np.concatenate((np.fliplr(c2_reduce),np.fliplr(np.flipud(c3_reduce))),axis=0)#right half of stitched image
    # right_uncertainty=np.concatenate((np.fliplr(c2_reduce.uncertainty._array),np.fliplr(np.flipud(c3_reduce.uncertainty._array))),axis=0)
    # right_mask=np.concatenate((np.fliplr(c2_reduce.mask),np.fliplr(np.flipud(c3_reduce.mask))),axis=0)

    stitched_data=np.concatenate((left_data,right_data),axis=1)
    # stitched_uncertainty=np.concatenate((left_uncertainty,right_uncertainty),axis=1)
    # stitched_mask=np.concatenate((left_mask,right_mask),axis=1)

    stitched=astropy.nddata.CCDData(stitched_data,unit=u.electron,uncertainty=StdDevUncertainty(np.ones(np.shape(stitched_data))),mask=np.zeros(np.shape(stitched_data)))

#            bad=np.where(stitched_uncertainty!=stitched_uncertainty)#bad variances due to negative counts after overscan/bias/dark correction
#            stitched_mask[bad]=True
#            stitched_uncertainty[bad]=1.e+10
#            stitched.uncertainty=stitched_uncertainty
#            stitched.mask=stitched_mask
#            stitched.mask[bad]=True
    stitched.header=c1_reduce.header
    stitched.write(outfile, overwrite=True)
    print("stitch_frames: Done")

from numba import jit
@jit(nopython=True)
def compute_flux_tile_fast(data, uncertainty, darkdata, darkuncertainty, biasuncertainty, corrdark, gain, exptime_ratio, obs_readnoise, ntot):
    for k in range(0,len(data)):
        for q in range(0,len(data[k])):
            if corrdark:
                uncertainty[k][q]=(
                    np.max(np.array([data[k][q]
                                    +darkdata[k][q]*gain*exptime_ratio
                                    +2.
                                    +obs_readnoise**2
                                    +(darkuncertainty[k][q]*gain*exptime_ratio)**2
                                    +(biasuncertainty[k][q]*gain)**2,
                                    0.6*(obs_readnoise**2+(darkuncertainty[k][q]*gain*exptime_ratio)**2
                                        +(biasuncertainty[k][q]*gain)**2)
                                    ])
                            ))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        poop1=(np.max(np.array([data[k][q]+darkdata[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(darkuncertainty[k][q]*gain*exptime_ratio)**2+(biasuncertainty[k][q]*gain)**2,0.6*(obs_readnoise**2+(darkuncertainty[k][q]*gain*exptime_ratio)**2+(biasuncertainty[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        poop2=(np.max(np.array([data[k][q]+2.+obs_readnoise**2,0.6*(obs_readnoise**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        print(poop1/poop2)
            else:
                uncertainty[k][q]=(
                    np.max(np.array([data[k][q]
                                    # +darkdata[k][q]*gain*exptime_ratio
                                    +2.
                                    +obs_readnoise**2
                                    # +(darkuncertainty[k][q]*gain*exptime_ratio)**2
                                    +(biasuncertainty[k][q]*gain)**2,
                                    0.6*(obs_readnoise**2
                                        #  +(darkuncertainty[k][q]*gain*exptime_ratio)**2
                                            +(biasuncertainty[k][q]*gain)**2)
                                    ])
                            ))**0.5
#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=10)
    #     print("Uncertainty computation {:0.0f}%".format(k/ntot*100), end='\r')
    # print("Uncertainty computation {:0.0f}%".format(k/ntot*100))
    return uncertainty

@jit(nopython=True)
def rescale_variances(gain_corr_data, gain_uncertainty_value, 
                      master_dark_data, master_dark_uncertainty, 
                      master_bias_uncertainty, 
                      gain, exptime_ratio, obs_readnoise):
    '''gain_corr_data is gain_corrected2
    gain_uncertainty_value is gain_corrected2.uncertainty.quantity.value
    master_dark_data is master_dark.data
    master_dark_uncertainty is master_dark.uncertainty.quantity.value
    master_bias_uncertainty is master_bias.uncertainty.quantity.value'''
    ## PIC I am not what the entire thing below was. It seems that the latest version
    ## of the code was doing something like that:
    for k in range(0,len(gain_corr_data)):
        for q in range(0,len(gain_corr_data[k])):
            ##rescale variances using empirically-determined fudges that hold
            ##when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
            gain_uncertainty_value[k][q]=(
                np.max(
                    np.array([gain_corr_data[k][q]
                              +master_dark_data[k][q]*gain*exptime_ratio
                              +2.+obs_readnoise**2
                              +(master_dark_uncertainty[k][q]*gain*exptime_ratio)**2
                              +(master_bias_uncertainty[k][q]*gain)**2,
                              0.6*(obs_readnoise**2
                                   +(master_dark_uncertainty[k][q]*gain*exptime_ratio)**2
                                   +(master_bias_uncertainty[k][q]*gain)**2)])))**0.5
    return gain_uncertainty_value

def stitch_frames(ccd, file_id, rawfiledict, masterbiasframes, masterdarkframes, filedict, binning, corrdark=False, corrbias=False):
    # for filename in framelist:
    # for file_id in all_list:
    # revbin = np.abs(np.array(binning[::-1])-2)+1 ## Converts binning in factor for immage dimensions
    revbin = binning
    for tile in np.arange(1, 5):
        filename = rawfiledict[(ccd, tile, file_id)]
        masterbiasfile = masterbiasframes[(ccd, tile)]
        masterdarkfile = masterdarkframes[(ccd, tile)]
        outfile = filedict[(ccd, file_id)]

        data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
        gain=float(data.header['egain'])

        if corrbias:
            master_bias=astropy.nddata.CCDData.read(masterbiasfile)
            if 'obs_rdnoise' in master_bias.header:
                obs_readnoise=float(master_bias.header['obs_rdnoise'])
            else:
                obs_readnoise=0
        else:
            master_bias=0
            obs_readnoise=0

        oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,revbin[1]*1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
        trimmed1=ccdproc.trim_image(oscan_subtracted[:,:revbin[1]*1024],add_keyword={'trim1':'Done'})
        trimmed2=ccdproc.trim_image(trimmed1[:revbin[0]*1028,:revbin[1]*1024],add_keyword={'trim2':'Done'})

        if corrbias:
            debiased0=ccdproc.subtract_bias(trimmed2,master_bias)
        else:
            print('Not correcting by bias')
            debiased0=trimmed2
        if corrdark:
            master_dark=astropy.nddata.CCDData.read(masterdarkfile)
            dedark0=ccdproc.subtract_dark(debiased0,master_dark,exposure_time='exptime',exposure_unit=u.second,scale=True,add_keyword={'dark_corr':'Done'})
        else:
            print('Not correcting by dark')
            dedark0 = debiased0
        
        # from IPython import embed
        # embed()


        # # _y = data_with_deviation.uncertainty.array
        # # _y = data_with_deviation.data
        # # _y = oscan_subtracted.data
        # _y = oscan_subtracted.data
        # plt.imshow(_y, vmin=np.nanpercentile(_y, 10), vmax=np.nanpercentile(_y, 90))
        # # plt.imshow(_y)#, vmin=np.nanpercentile(_y, 10), vmax=np.nanpercentile(_y, 90))
        # plt.colorbar()
        # plt.show()

        data_with_deviation=ccdproc.create_deviation(dedark0,gain=data.meta['egain']*u.electron/u.adu,readnoise=obs_readnoise*u.electron)
        gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                master_dark_gain_corrected=ccdproc.gain_correct(master_dark,master_dark.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                master_bias_gain_corrected=ccdproc.gain_correct(master_bias,master_bias.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})

        gain_corrected2=deepcopy(gain_corrected)
        if corrdark:
            exptime_ratio=float(data.header['exptime'])/float(master_dark.meta['exptime'])
        else:
            exptime_ratio=0.
        ntot = len(gain_corrected2.data)
        # from IPython import embed
        # embed()

        _data = np.array(np.copy(gain_corrected2.data), dtype=float)
        _uncertainty = np.array(np.copy(gain_corrected2.uncertainty.quantity.value), dtype=float)
        _datadark = np.array(np.copy(master_dark.data), dtype=float)
        _uncertaintydark = np.array(np.copy(master_dark.uncertainty.quantity.value), dtype=float)
        _uncertaintybias = np.array(np.copy(master_bias.uncertainty.quantity.value), dtype=float)
        uncertainty = compute_flux_tile_fast(_data, _uncertainty,
                               _datadark, _uncertaintydark, 
                               _uncertaintybias,
                               corrdark, gain, exptime_ratio, obs_readnoise, ntot)
        gain_corrected2.uncertainty = uncertainty

        ## PIC I am not what the entire thing below was. It seems that the latest version
        ## of the code was doing something like that:
        #
        # for k in range(0,len(gain_corrected2.data)):
        #     for q in range(0,len(gain_corrected2.data[k])):
        #         gain_corrected2.uncertainty.quantity.value[k][q]=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
        ## Doing the same but with a jit function

        '''gain_corr_data is gain_corrected2
        gain_uncertainty_value is gain_corrected2.uncertainty.quantity.value
        master_dark_data is master_dark.data
        master_dark_uncertainty is master_dark.uncertainty.quantity.value
        master_bias_uncertainty is master_bias.uncertainty.quantity.value'''
        gain_corr_data = np.array(gain_corrected2.data, dtype=float)
        gain_uncertainty_value = np.array(gain_corrected2.uncertainty.quantity.value.data, dtype=float)
        master_dark_data = np.array(master_dark.data, dtype=float)
        master_dark_uncertainty = np.array(master_dark.uncertainty.quantity.value, dtype=float)
        master_bias_uncertainty = np.array(master_bias.uncertainty.quantity.value, dtype=float)
        #
        gain_uncertainty_value = rescale_variances(gain_corr_data, gain_uncertainty_value, 
                      master_dark_data, master_dark_uncertainty, 
                      master_bias_uncertainty, 
                      gain, exptime_ratio, obs_readnoise)
        # #
        # ## PIC I also tried to set the uncertainty manually
        # gain_corrected2.uncertainty = np.sqrt(np.sqrt(gain_corrected2.data)**2+obs_readnoise**2)
        ## PIC But everything leads to larger noise here...
        ## Instead I bypass the uncertainty to take the average only
        gain_corrected2.uncertainty = np.sqrt(gain_corrected2.data)*0+1

        #gain_corrected2.uncertainty = gain_uncertainty_value*0+1
        # gain_corrected2.mask[np.isnan(gain_corrected2.uncertainty.array)] = True
#         for k in range(0,len(gain_corrected2.data)):
#             for q in range(0,len(gain_corrected2.data[k])):
#                 if corrdark:
#                     gain_corrected2.uncertainty.quantity.value[k][q]=(
#                         np.max(np.array([gain_corrected2.data[k][q]
#                                         +master_dark.data[k][q]*gain*exptime_ratio
#                                         +2.
#                                         +obs_readnoise**2
#                                         +(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2
#                                         +(master_bias.uncertainty.quantity.value[k][q]*gain)**2,
#                                         0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2
#                                             +(master_bias.uncertainty.quantity.value[k][q]*gain)**2)
#                                         ])
#                                 ))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
# #                        poop1=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
# #                        poop2=(np.max(np.array([gain_corrected2.data[k][q]+2.+obs_readnoise**2,0.6*(obs_readnoise**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
# #                        print(poop1/poop2)
#                 else:
#                     gain_corrected2.uncertainty.quantity.value[k][q]=(
#                         np.max(np.array([gain_corrected2.data[k][q]
#                                         # +master_dark.data[k][q]*gain*exptime_ratio
#                                         +2.
#                                         +obs_readnoise**2
#                                         # +(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2
#                                         +(master_bias.uncertainty.quantity.value[k][q]*gain)**2,
#                                         0.6*(obs_readnoise**2
#                                             #  +(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2
#                                              +(master_bias.uncertainty.quantity.value[k][q]*gain)**2)
#                                         ])
#                                 ))**0.5
# #                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=10)
#             print("Uncertainty computation {:0.0f}%".format(k/ntot*100), end='\r')
#         print("Uncertainty computation {:0.0f}%".format(k/ntot*100))
#                bad=np.where(gain_corrected.data<0.)
#                bad=np.where(gain_corrected._uncertainty.quantity.value!=gain_corrected._uncertainty.quantity.value)#bad variances due to negative counts after overscan/bias/dark correction
#                gain_corrected.uncertainty.quantity.value[bad]=obs_readnoise
        if 'c1' in filename:
            c1_reduce=gain_corrected2
        if 'c2' in filename:
            c2_reduce=gain_corrected2
        if 'c3' in filename:
            c3_reduce=gain_corrected2
        if 'c4' in filename:
            c4_reduce=gain_corrected2

    left_data=np.concatenate((c1_reduce,np.flipud(c4_reduce)),axis=0)#left half of stitched image
    left_uncertainty=np.concatenate((c1_reduce.uncertainty._array,np.flipud(c4_reduce.uncertainty._array)),axis=0)
    left_mask=np.concatenate((c1_reduce.mask,np.flipud(c4_reduce.mask)),axis=0)
    right_data=np.concatenate((np.fliplr(c2_reduce),np.fliplr(np.flipud(c3_reduce))),axis=0)#right half of stitched image
    right_uncertainty=np.concatenate((np.fliplr(c2_reduce.uncertainty._array),np.fliplr(np.flipud(c3_reduce.uncertainty._array))),axis=0)
    right_mask=np.concatenate((np.fliplr(c2_reduce.mask),np.fliplr(np.flipud(c3_reduce.mask))),axis=0)

    stitched_data=np.concatenate((left_data,right_data),axis=1)
    stitched_uncertainty=np.concatenate((left_uncertainty,right_uncertainty),axis=1)
    stitched_mask=np.concatenate((left_mask,right_mask),axis=1)

    stitched=astropy.nddata.CCDData(stitched_data,unit=u.electron,uncertainty=StdDevUncertainty(stitched_uncertainty),mask=stitched_mask)

#            bad=np.where(stitched_uncertainty!=stitched_uncertainty)#bad variances due to negative counts after overscan/bias/dark correction
#            stitched_mask[bad]=True
#            stitched_uncertainty[bad]=1.e+10
#            stitched.uncertainty=stitched_uncertainty
#            stitched.mask=stitched_mask
#            stitched.mask[bad]=True
    stitched.header=c1_reduce.header
    stitched.write(outfile, overwrite=True)
    print("stitch_frames: Done")

def stitch_frames_bad(framelist, masterbiasfile, masterdarkfile, outfile):
    for filename in framelist:
        master_bias=astropy.nddata.CCDData.read(masterbiasfile)
        obs_readnoise=float(master_bias.header['obs_rdnoise'])
        master_dark=astropy.nddata.CCDData.read(masterdarkfile)

        data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
        gain=float(data.header['egain'])

        oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
        trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
        trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})

        debiased0=ccdproc.subtract_bias(trimmed2,master_bias)
        dedark0=ccdproc.subtract_dark(debiased0,master_dark,exposure_time='exptime',exposure_unit=u.second,scale=True,add_keyword={'dark_corr':'Done'})

        data_with_deviation=ccdproc.create_deviation(dedark0,gain=data.meta['egain']*u.electron/u.adu,readnoise=obs_readnoise*u.electron)

        gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                master_dark_gain_corrected=ccdproc.gain_correct(master_dark,master_dark.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                master_bias_gain_corrected=ccdproc.gain_correct(master_bias,master_bias.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})

        gain_corrected2=deepcopy(gain_corrected)
        exptime_ratio=float(data.header['exptime'])/float(master_dark.meta['exptime'])

        ntot = len(gain_corrected2.data)
        for k in range(0,len(gain_corrected2.data)):
            for q in range(0,len(gain_corrected2.data[k])):
                gain_corrected2.uncertainty.quantity.value[k][q]=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        poop1=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        poop2=(np.max(np.array([gain_corrected2.data[k][q]+2.+obs_readnoise**2,0.6*(obs_readnoise**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        print(poop1/poop2)
#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=10)
            print("Uncertainty computation {:0.0f}%".format(k/ntot*100), end='\r')
        print("Uncertainty computation {:0.0f}%".format(k/ntot*100))
#                bad=np.where(gain_corrected.data<0.)
#                bad=np.where(gain_corrected._uncertainty.quantity.value!=gain_corrected._uncertainty.quantity.value)#bad variances due to negative counts after overscan/bias/dark correction
#                gain_corrected.uncertainty.quantity.value[bad]=obs_readnoise

        if 'c1' in filename:
            c1_reduce=gain_corrected2
        if 'c2' in filename:
            c2_reduce=gain_corrected2
        if 'c3' in filename:
            c3_reduce=gain_corrected2
        if 'c4' in filename:
            c4_reduce=gain_corrected2

    left_data=np.concatenate((c1_reduce,np.flipud(c4_reduce)),axis=0)#left half of stitched image
    left_uncertainty=np.concatenate((c1_reduce.uncertainty._array,np.flipud(c4_reduce.uncertainty._array)),axis=0)
    left_mask=np.concatenate((c1_reduce.mask,np.flipud(c4_reduce.mask)),axis=0)
    right_data=np.concatenate((np.fliplr(c2_reduce),np.fliplr(np.flipud(c3_reduce))),axis=0)#right half of stitched image
    right_uncertainty=np.concatenate((np.fliplr(c2_reduce.uncertainty._array),np.fliplr(np.flipud(c3_reduce.uncertainty._array))),axis=0)
    right_mask=np.concatenate((np.fliplr(c2_reduce.mask),np.fliplr(np.flipud(c3_reduce.mask))),axis=0)

    stitched_data=np.concatenate((left_data,right_data),axis=1)
    stitched_uncertainty=np.concatenate((left_uncertainty,right_uncertainty),axis=1)
    stitched_mask=np.concatenate((left_mask,right_mask),axis=1)

    stitched=astropy.nddata.CCDData(stitched_data,unit=u.electron,uncertainty=StdDevUncertainty(stitched_uncertainty),mask=stitched_mask)

#            bad=np.where(stitched_uncertainty!=stitched_uncertainty)#bad variances due to negative counts after overscan/bias/dark correction
#            stitched_mask[bad]=True
#            stitched_uncertainty[bad]=1.e+10
#            stitched.uncertainty=stitched_uncertainty
#            stitched.mask=stitched_mask
#            stitched.mask[bad]=True
    stitched.header=c1_reduce.header
    stitched.write(outfile, overwrite=True)
    print("stitch_frames: Done")

def stdmean(a,axis=None,dtype=None,out=None,ddof=0,keepdims=np._NoValue):
    std=np.ma.std(a)
#    print(len(a))
    return std/np.sqrt(len(a))

## PIC: The following functions are re-written or adapted versions of those found in m2fs_process.

# def get_thar(datadir,utdate,ccd,tharfile,hires_exptime,medres_exptime,field_name,use_flat):
def get_thar(filedict, ccd, id_list, lco, use_flat, exptime, id_lines_array_files):
    '''Here again, there seems to be something about filters.
    Filters are here used to select what to use for the "exptime" and to detect twilight.
    I am assuming that we are high resolution, and bypass the twilight for now.'''
    import astropy
    import dill as pickle
    from astropy import time, coordinates as coord, units as u
    from astropy.coordinates import SkyCoord, EarthLocation
    import numpy as np

    field_name = 'blabla'

    thar=[]
    lines=[]
    temperature=[]
    thar_exptime=[]
    thar_mjd=[]
    for file_id in id_list:
        data_file = filedict[(ccd, file_id)]
        if use_flat:
            # id_lines_array_file = tmppath + "{}{:04d}-id-lines-array.pickle".format(ccd, file_id)
            id_lines_array_file = id_lines_array_files[(ccd, file_id)]
        else:
            raise Exception("fdump.get_thar(): case not implemented. Set use_flat=True.")
            # id_lines_array_file = tmppath + "{}{:04d}-id-lines-array-noflat.pickle".format(ccd, file_id)
        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
        ## Sometimes the DATE-OBS key card contains the T and the UT-TIME... So here is a fix:
        if 'T' in data.header['DATE-OBS']:
            time0=[data.header['DATE-OBS'],data.header['DATE-OBS'].replace(data.header['UT-TIME'], data.header['UT-END'])]
        else:
            time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
        times=time.Time(time0,location=lco,precision=6)
        filtername=data.header['FILTER']
        ## PIC: Bypassing the following
#         if filtername=='Mgb_Rev2':
#             filtername='Mgb_HiRes'
#         if filtername=='Mgb_HiRes':
#             if ((float(data.header['exptime'])>=hires_exptime)|('twilight' in field_name)):
#                 thar_mjd.append(np.mean(times.mjd))
#                 thar.append(id_lines_array)
# #                lines.append([id_lines_array[q].wav[id_lines_array[q].wav.mask==False] for q in range(0,len(id_lines_array))])
#                 temperature.append(data.header['T-DOME'])
#                 thar_exptime.append(float(data.header['exptime']))
#         if filtername=='Mgb_MedRes':
#             if ((float(data.header['exptime'])>=medres_exptime)|('twilight' in field_name)):
#                 thar_mjd.append(np.mean(times.mjd))
#                 thar.append(id_lines_array)
# #                lines.append([id_lines_array[q].wav.data[id_lines_array[q].wav.mask==False] for q in range(0,len(id_lines_array))])
#                 temperature.append(data.header['T-DOME'])
#                 thar_exptime.append(float(data.header['exptime']))
        ## -- 
        ## And simply writing:
        ## For a twilight, bypass exptime threshold
        if ((float(data.header['exptime'])>=exptime)|('twilight' in field_name)): 
            thar_mjd.append(np.mean(times.mjd))
            thar.append(id_lines_array)
#                lines.append([id_lines_array[q].wav[id_lines_array[q].wav.mask==False] for q in range(0,len(id_lines_array))])
            temperature.append(data.header['T-DOME'])
            thar_exptime.append(float(data.header['exptime']))

    temperature=np.array(temperature)
    thar_exptime=np.array(thar_exptime)
    thar_mjd=np.array(thar_mjd)

    if len(thar)==0:
        print('ERROR: no qualifying ThArNe exposures for this resolution!!!!!')
    if np.std(thar_exptime)>0.:
        if filtername=='Mgb_MedRes':
            keep=np.where(thar_exptime==np.min(thar_exptime))[0]
            thar0=[]
            for qqq in keep:
                thar0.append(thar[qqq])
            thar=thar0
            thar_mjd=thar_mjd[keep]
            thar_exptime=thar_exptime[keep]

    return(thar,lines,temperature,thar_exptime,thar_mjd)

def get_linelist(linelist_file, species_name='default-name'):
    import numpy as np
    with open(linelist_file) as f:
        data=f.readlines()
    linelist_wavelength=[]
    linelist_species=[]
    for line in data:
        p=line.split()
        linelist_wavelength.append(float(p[0]))
        if len(p)>1:
            linelist_species.append(p[1])
        else:
            linelist_species.append(species_name)
    linelist_wavelength=np.array(linelist_wavelength)
    linelist_species=np.array(linelist_species)
    return m2fs.linelist(wavelength=linelist_wavelength,species=linelist_species)

from . import m2fs_process as m2fs

### TODELETE DEFINITION REPEATED
# def get_id_lines_template(extract1d,linelist,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order,threshold_factor,window,id_lines_order,id_lines_rejection_iterations,id_lines_rejection_sigma,id_lines_tol_angs,id_lines_template_fiddle,id_lines_template0,resolution_order,resolution_rejection_iterations):
#     import numpy as np
#     import astropy.units as u
#     from specutils.fitting import find_lines_threshold
#     from specutils.fitting import find_lines_derivative
#     from specutils.spectra import Spectrum1D
#     import matplotlib.pyplot as plt
#     from copy import deepcopy
#     from astropy.modeling import models,fitting

#     continuum0,spec_contsub,fit_lines=m2fs.get_fitlines(extract1d,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,threshold_factor,window,continuum_rejection_order)

#     line_centers=[fit_lines.fit[j].mean.value for j in range(0,len(fit_lines.fit))]
#     id_lines_pix=[]
#     id_lines_wav=[]
#     id_lines_species=[]
#     id_lines_used=[]
#     order=[id_lines_order]
#     rejection_iterations=[id_lines_rejection_iterations]
#     rejection_sigma=[id_lines_rejection_sigma]
#     func=[models.Legendre1D(degree=1)]
#     rms=[]
#     npoints=[]
#     if id_lines_template_fiddle:
#         id_lines_used=np.where(id_lines_template0.wav.mask==False)[0].tolist()
#         id_lines_pix=[id_lines_template0.fit_lines.fit[q].mean.value for q in id_lines_used]
#         id_lines_wav=id_lines_template0.wav[id_lines_used].tolist()
#         func=[id_lines_template0.func]
#         order=[id_lines_template0.func.degree]
#         rms=[id_lines_template0.rms]
#         npoints=[id_lines_template0.npoints]

#     fig=plt.figure(1)
#     ax1,ax2=m2fs.plot_id_lines(extract1d,continuum0,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func[len(func)-1],fig)
# #
#     print('press \'m\' to ID line nearest cursor \n')
#     print('press \'d\' to delete ID for line nearest cursor \n')
#     print('press \'o\' to change order of polynomial \n')
#     print('press \'r\' to change rejection sigma factor \n')
#     print('press \'t\' to change number of rejection iterations \n')
#     print('press \'g\' to re-fit wavelength solution \n')
#     print('press \'l\' to add lines from linelist according to fit \n')
#     print('press \'q\' to quit \n')
#     print('press \'.\' to print cursor position and position of nearest line \n')

#     cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_id_lines(event,[deepcopy(extract1d),continuum0,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints,id_lines_tol_angs,fig]))
#     plt.show()
#     plt.close()

#     wav=np.ma.masked_array(np.full((len(fit_lines.fit)),-999,dtype='float'),mask=np.full((len(fit_lines.fit)),True,dtype=bool))

#     for j in range(0,len(id_lines_pix)):
#         wav[id_lines_used[j]]=id_lines_wav[j]
#         wav.mask[id_lines_used[j]]==False

#     resolution,resolution_rms,resolution_npoints=m2fs.get_resolution(deepcopy(fit_lines),deepcopy(wav),resolution_order,resolution_rejection_iterations)

#     return m2fs.id_lines(aperture=extract1d.aperture,fit_lines=fit_lines,wav=wav,func=func[len(func)-1],rms=rms[len(rms)-1],npoints=npoints[len(npoints)-1],resolution=resolution,resolution_rms=resolution_rms,resolution_npoints=resolution_npoints)

def on_key_id_lines(event,args_list):
    import numpy as np
    from . import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils.spectra import Spectrum1D
    from astropy.modeling import models,fitting
    from copy import deepcopy

    #print('you pressed ',event.key)

    global id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints
    extract1d,continuum,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints,id_lines_tol_angs,fig=args_list

    if event.key=='.':
        x0=event.xdata
        y0=event.ydata
        print('cursor pixel = ',str(x0))
        dist=(line_centers-x0)**2
        best=np.where(dist==np.min(dist))[0][0]
        print('nearest line pixel = ',str(line_centers[best]))
        used=np.array(np.where(id_lines_pix==line_centers[best])[0])
        if len(used)>0:
            print('wavelength from line list =',str(np.array(id_lines_wav)[used][0]))
        else:
            print('line not yet identified')
        print('function wavelength =',str(func[len(func)-1](line_centers[best])))

    if event.key=='m':
        x0=event.xdata
        y0=event.ydata
        print('cursor pixel = ',str(x0))
        dist=(line_centers-x0)**2
        best=np.where(dist==np.min(dist))[0][0]
        print('nearest line pixel = ',str(line_centers[best]))
        command=input('enter wavelength in Angstroms \n')
        if command=='':
            print('no information entered')
        elif not command.replace('.', '').isnumeric():
            print('Please enter a float') 
            pass ## Avoid breaking the program if there was a alpha value in the command
        else:
            id_lines_pix.append(line_centers[best])
            id_lines_wav.append(float(command))
            id_lines_used.append(best)

    if event.key=='d':#delete nearest boundary point
        print('delete point nearest (',event.xdata,event.ydata,')')
        if len(id_lines_pix)>0:
            dist=(id_lines_pix-event.xdata)**2
            best=np.where(dist==np.min(dist))[0][0]
            del id_lines_pix[best]
            del id_lines_wav[best]
            del id_lines_used[best]
        else:
            print('no ID\'d lines to delete!')

    if event.key=='o':
        print('order of polynomial fit is ',order[len(order)-1])
        command=input('enter new order (must be integer): ')
        if command=='':
            print('keeping original value')
        else:
            order.append(int(command))
        func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
        func.append(func0)
        rms.append(rms0)
        npoints.append(npoints0)

    if event.key=='r':
        print('rejection sigma is ',rejection_sigma[len(rejection_sigma)-1])
        command=input('enter new rejection sigma (float): ')
        if command=='':
            print('keeping original value')
        else:
            rejection_sigma.append(float(command))
        func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
        func.append(func0)
        rms.append(rms0)
        npoints.append(npoints0)

    if event.key=='t':
        print('number of rejection iterations is ',rejection_iterations[len(rejection_iterations)-1])
        command=input('enter new rejection iterations = (integer): ')
        if command=='':
            print('keeping original value')
        else:
            rejection_iterations.append(int(command))
        func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
        func.append(func0)
        rms.append(rms0)
        npoints.append(npoints0)

    if event.key=='z':
        for q in range(1,len(id_lines_pix)):
            del(id_lines_pix[len(id_lines_pix)-1])
        for q in range(1,len(id_lines_wav)):
            del(id_lines_wav[len(id_lines_wav)-1])
        for q in range(1,len(id_lines_used)):
            del(id_lines_used[len(id_lines_used)-1])
        for q in range(1,len(order)):
            del(order[len(order)-1])
        for q in range(1,len(rejection_iterations)):
            del(rejection_iterations[len(rejection_iterations)-1])
        for q in range(1,len(rejection_sigma)):
            del(rejection_sigma[len(rejection_sigma)-1])
        for q in range(1,len(func)):
            del(func[len(func)-1])

    if event.key=='g':
        func0,rms0,npoints0,y=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
        func.append(func0)
        rms.append(rms0)
        npoints.append(npoints0)

    if event.key=='l':
        func0,rms0,npoints0,y=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
        new_pix,new_wav,new_used=m2fs.line_id_add_lines(deepcopy(linelist),deepcopy(line_centers),deepcopy(id_lines_used),deepcopy(func0),id_lines_tol_angs)

        for i in range(0,len(new_pix)):
            id_lines_pix.append(new_pix[i])
            id_lines_wav.append(new_wav[i])
            id_lines_used.append(new_used[i])
        func0,rms0,npoints0,y=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
        func.append(func0)
        rms.append(rms0)
        npoints.append(npoints0)

    ax1,ax2=m2fs.plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func[len(func)-1],fig)
    fig.canvas.draw_idle()
    return

def fiddle_apertures(columnspec_array,columnarr,window,apertures,find_apertures_file):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models

    column = columnarr[0]

    subregion,fit,realvirtual,initial=apertures.subregion,apertures.fit,apertures.realvirtual,apertures.initial
    subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,apertures.initial)

    fig=plt.figure(1)
    x=np.arange(0,len(columnspec_array[column].spec))
    ax1=fig.add_subplot(111)
    ax1.plot(columnspec_array[column].pixel,columnspec_array[column].spec,lw=0.3,color='k')
    ax1.plot(columnspec_array[column].pixel,columnspec_array[column].continuum(x),color='green',lw=1)
    ax1.set_xlim([0,np.max(x)])
    for j in range(0,len(fit)):
        use=np.where((columnspec_array[column].pixel.value>=subregion[j].lower.value)&(columnspec_array[column].pixel.value<=subregion[j].upper.value))[0]
        sub_spectrum_pixel=columnspec_array[column].pixel[use]
        y_fit=fit[j](sub_spectrum_pixel).value+columnspec_array[column].continuum(sub_spectrum_pixel.value)

        ax1.axvline(x=fit[j].mean.value,color='b',lw=0.3,alpha=0.7)
        ax1.plot(sub_spectrum_pixel.value,y_fit,color='r',lw=0.2,alpha=0.7)
        ax1.set_ylim([0,10000])
        ax1.text(fit[j].mean.value,0,str(j+1),fontsize=8)
        if not realvirtual[j]:
            ax1.axvline(x=fit[j].mean.value,color='k',lw=0.5,alpha=1,linestyle='--')

    thsarr = [0.05] ## default threshold
    print('\n')
    print('press \'r\' to reset all apertures fits \n')
    print('press \'g\' to iteratively fit all the apertures using maximum peak \n')
    print('press \'h\' to iteratively fit all the even apertures \n')
    print('press \'j\' to iteratively fit all the odd apertures \n')
    print('press \'p\' to iteratively mark all apertures as phantom \n')
    print('press \'k\' to iteratively fit the n apertures starting at i \n')
    print('press \'K\' same as key but with positions from initial guess \n')
    print('press \'d\' to delete aperture nearest cursor \n')
    print('press \'e\' to delete all apertures \n')
    print('press \'n\' to add new real aperture at cursor position \n')
    print('press \'a\' to add new phantom aperture at cursor position \n')
    print('press \'z\' to return to initial apertures \n')
    print('press \'q\' to quit \n')
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) # PIC This disable any key that was not set up.
    cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_find(event,[columnspec_array,
                                                                                  columnarr,subregion,
                                                                                  fit,realvirtual,
                                                                                  initial,window,fig,thsarr]))
    plt.show()
    # print('This is the column I get: {}'.format(column))
#    plt.savefig(find_apertures_file,dpi=200)
    subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
    return m2fs.aperture_profile(fit,subregion,realvirtual,initial)

def on_key_find(event,args_list):
    import numpy as np
    from . import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils.spectra import Spectrum1D

    print('you pressed ', event.key)
    keyoptions = ['z', 'd', 'e', 'n', 'a', 'q', 'r', 'g', 'h', ' H', 'j', 'p', 'c', 'k', 'K', 't']

    global columnspec_array,columnarr,subregion,fit,realvirtual,initial,window
    columnspec_array,columnarr,subregion,fit,realvirtual,initial,window,fig,thsarr=args_list


    ths = float(thsarr[0])
    # ths = 1/20 ## Threshold for peak detection
    # from IPython import embed
    # embed()

    ## Sort the apertures
    means = [fit[i].mean.value for i in range(len(fit))] ## means of the gaussian fits
    _idx_sort = np.argsort(means) ## sorted positions

    fit_sorted = [fit[i] for i in _idx_sort]
    subregion_sorted = [subregion[i] for i in _idx_sort]
    realvirtual_sorted = [realvirtual[i] for i in _idx_sort]
    initial_sorted = [initial[i] for i in _idx_sort]

    for i in range(len(fit)):
        fit[i] = fit_sorted[i]
        subregion[i] = subregion_sorted[i]
        realvirtual[i] = realvirtual_sorted[i]
        initial[i] = initial_sorted[i]

    column = columnarr[0]
    print('looking at column: {}'.format(column))

    if event.key in keyoptions:
        if event.key=='r':
            apertures = columnspec_array[column].apertures_profile
            subregion,fit,realvirtual,initial=apertures.subregion,apertures.fit,apertures.realvirtual,apertures.initial
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,apertures.initial)

        if event.key=='z':#revert to initial apertures
            to_delete=np.where(np.logical_not(initial))[0]
            j=0
            for i in range(0,len(to_delete)):
                del initial[to_delete[i]-j]
                del subregion[to_delete[i]-j]
                del fit[to_delete[i]-j]
                del realvirtual[to_delete[i]-j]
                j+=1
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)

        if event.key=='d':
            ## PIC fixed issue by adding '.value' in the line below
            aperture_centers=[fit[q].mean.value for q in range(0,len(fit))]
            x0=event.xdata
            y0=event.ydata
            print(aperture_centers)
            dist=(aperture_centers-event.xdata)**2
            best=np.where(dist==np.min(dist))[0][0]
            print('deleting aperture '+str(best+1)+', centered at '+str(aperture_centers[best]))
            del subregion[best]
            del fit[best]
            del realvirtual[best]
            del initial[best]
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)

        if event.key=='e':
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)

        if event.key=='n':
            new_center=float(event.xdata)
            x_center=new_center
            spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
            subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
            subregion.append(subregion0)
            fit.append(fit0)
            realvirtual.append(True)
            initial.append(False)
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)

        if event.key=='g':
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the thing we plot
            print(ths)
            idx, _ = peak_finder(specarray, ths)
            ntot = len(idx)
            for ieventxdata, eventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                new_center=float(eventxdata)
                x_center=new_center
                spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                subregion.append(subregion0)
                fit.append(fit0)
                realvirtual.append(True)
                initial.append(False)
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))

        if event.key=='t':
            ths = input('Enter new threshold [ths={:0.2f}]: '.format(ths))
            thsarr[0] = ths

        if event.key=='h':
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the think we plot
            idx, _ = peak_finder(specarray, ths)
            ntot = len(idx)
            for ieventxdata, eventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                if (ieventxdata%2)==0:
                    new_center=float(eventxdata)
                    x_center=new_center
                    spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                    subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                    subregion.append(subregion0)
                    fit.append(fit0)
                    realvirtual.append(True)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
                else:
                    new_center=float(eventxdata)
                    x_center=new_center
                    val1=x_center-window/2.
                    val2=x_center+window/2.
                    subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
                    aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
                    halfwindow=window/2.
                    fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
                    realvirtual.append(False)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
        if event.key=='H':
            ## Same as h but I keep the current position of the identified appertures
            initpos = [fit[i].mean.value for i in range(len(fit))]
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the think we plot
            # idx, _ = peak_finder(specarray, ths)
            idx = initpos
            ntot = len(idx)
            for ieventxdata, eventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                if (ieventxdata%2)==0:
                    new_center=float(eventxdata)
                    x_center=new_center
                    spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                    subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                    subregion.append(subregion0)
                    fit.append(fit0)
                    realvirtual.append(True)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
                else:
                    new_center=float(eventxdata)
                    x_center=new_center
                    val1=x_center-window/2.
                    val2=x_center+window/2.
                    subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
                    aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
                    halfwindow=window/2.
                    fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
                    realvirtual.append(False)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
        if event.key=='j':
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the think we plot
            idx, _ = peak_finder(specarray, ths)
            ntot = len(idx)
            for ieventxdata, enventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                if (ieventxdata%2)==1:
                    new_center=float(enventxdata)
                    x_center=new_center
                    spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                    subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                    subregion.append(subregion0)
                    fit.append(fit0)
                    realvirtual.append(True)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
                else:
                    new_center=float(enventxdata)
                    x_center=new_center
                    val1=x_center-window/2.
                    val2=x_center+window/2.
                    subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
                    aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
                    halfwindow=window/2.
                    fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
                    realvirtual.append(False)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 

        if event.key=='k':
            ini = input('Start at aperture i. Enter i: ')
            nap = input('fit every n apertures. Enter n: ')
            csa = input('how many consecutive apertures (default=1). Enter n: ')
            nap = int(float(nap))
            ini = int(float(ini)-1)
            if csa.isnumeric():
                csa = int(float(csa))
            else:
                csa = 1
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the think we plot
            idx, _ = peak_finder(specarray, ths)
            ntot = len(idx)
            allowed_apertures = []
            for iii in range(150):
                for _csa in range(1, csa+1):
                    allowed_apertures.append(ini+_csa-1+nap*iii)
            for ieventxdata, enventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                if ieventxdata in allowed_apertures:
                    new_center=float(enventxdata)
                    x_center=new_center
                    spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                    subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                    subregion.append(subregion0)
                    fit.append(fit0)
                    realvirtual.append(True)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
                else:
                    new_center=float(enventxdata)
                    x_center=new_center
                    val1=x_center-window/2.
                    val2=x_center+window/2.
                    subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
                    aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
                    halfwindow=window/2.
                    fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
                    realvirtual.append(False)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 

        if event.key=='K':
            ini = input('Start at aperture i. Enter i: ')
            nap = input('fit every n apertures. Enter n: ')
            csa = input('how many consecutive apertures (default=1). Enter n: ')
            nap = int(float(nap))
            ini = int(float(ini)-1)
            if csa.isnumeric():
                csa = int(float(csa))
            else:
                csa = 1
            initpos = [fit[i].mean.value for i in range(len(fit))]
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            mycenters = np.array([int(fit[q].mean.value) for q in range(0,len(fit))]) 
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the think we plot
            # idx, _ = peak_finder(specarray, ths, ths=0, med=0)
            idx = initpos
            # ## Take the closest to those initially found
            # mynewidx = np.copy(idx)
            # for ii in range(len(mycenters)):
            #     res = (mycenters[ii]-idx)**2
            #     position = np.where(res==np.min(res))[0][0]
            #     mynewidx[ii] = idx[position]
            # idx = np.copy(mynewidx)
            # idx = np.copy(mycenters)
            ntot = len(idx)
            allowed_apertures = []
            for iii in range(150):
                for _csa in range(1, csa+1):
                    allowed_apertures.append(ini+_csa-1+nap*iii)
            for ieventxdata, enventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                if ieventxdata in allowed_apertures:
                    new_center=float(enventxdata)
                    x_center=new_center
                    spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                    subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                    subregion.append(subregion0)
                    fit.append(fit0)
                    realvirtual.append(True)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
                else:
                    new_center=float(enventxdata)
                    x_center=new_center
                    val1=x_center-window/2.
                    val2=x_center+window/2.
                    subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
                    aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
                    halfwindow=window/2.
                    fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
                    realvirtual.append(False)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 


        if event.key=='p':
            ## PIC: First delete everything:
            aperture_centers=[fit[q].mean for q in range(0,len(fit))]
            print('deleting all apertures ')
            for q in range(0,len(subregion)):
                del(subregion[len(subregion)-1])
            for q in range(0,len(fit)):
                del(fit[len(fit)-1])
            for q in range(0,len(realvirtual)):
                del(realvirtual[len(realvirtual)-1])
            for q in range(0,len(initial)):
                del(initial[len(initial)-1])
            # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
            ## PIC: Iterate through everything:
            specarray = columnspec_array[column].spec ## That's the think we plot
            idx, _ = peak_finder(specarray, ths)
            ntot = len(idx)
            for ieventxdata, enventxdata in enumerate(idx):
                print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100), end='\r')
                if 2==1: ## Never run into this part of the loop. We want to assume all apertures to be phantom.
                    new_center=float(enventxdata)
                    x_center=new_center
                    spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
                    subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
                    subregion.append(subregion0)
                    fit.append(fit0)
                    realvirtual.append(True)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)
                else:
                    new_center=float(enventxdata)
                    x_center=new_center
                    val1=x_center-window/2.
                    val2=x_center+window/2.
                    subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
                    aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
                    halfwindow=window/2.
                    fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
                    realvirtual.append(False)
                    initial.append(False)
                    # subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 
            print("Fitting apertures... {:0.2f}%".format(ieventxdata/ntot*100))
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial) 


        if event.key=='a':
            new_center=float(event.xdata)
            x_center=new_center
            val1=x_center-window/2.
            val2=x_center+window/2.
            subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
            aaa=float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
            halfwindow=window/2.
            fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
            realvirtual.append(False)
            initial.append(False)
            subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)

        if event.key=='c':
            ## Update the column number
            newcolumn = input("Column nb [current:{}/{}]: ".format(column, len(columnspec_array)))
            diff = column - int(float(newcolumn))
            # from IPython import embed
            # embed()
            column = column - diff
            columnarr[0] = column

        if event.key=='q':
            plt.close(event.canvas.figure)
            # return
    
    x=np.arange(0,len(columnspec_array[column].spec))
    # ax1=fig.add_subplot(111)
    ax1 = fig.axes[0]
    ax1.cla()
    ax1.plot(columnspec_array[column].pixel,columnspec_array[column].spec,lw=0.3,color='k')
    # specarray = columnspec_array[column].spec ## That's the thing we plot
    # idx = peak_finder(specarray, ths)
    # for i in range(len(idx)):
    #     ax1.axvline(idx[i])
    ax1.plot(columnspec_array[column].pixel,columnspec_array[column].continuum(x),color='green',lw=1)
    ax1.set_xlim([0,np.max(x)])
    for j in range(0,len(fit)):
        use=np.where((columnspec_array[column].pixel.value>=subregion[j].lower.value)&(columnspec_array[column].pixel.value<=subregion[j].upper.value))[0]
        sub_spectrum_pixel=columnspec_array[column].pixel[use]
        y_fit=fit[j](sub_spectrum_pixel).value+columnspec_array[column].continuum(sub_spectrum_pixel.value)

        ax1.axvline(x=fit[j].mean.value,color='b',lw=0.3,alpha=0.7)
        ax1.plot(sub_spectrum_pixel.value,y_fit,color='r',lw=0.2,alpha=0.7)
        ax1.text(fit[j].mean.value,0,str(j+1),fontsize=8)
        ax1.set_ylim([0,10000])
        if not realvirtual[j]:
            ax1.axvline(x=fit[j].mean.value,color='k',lw=0.5,alpha=1,linestyle='--')
    fig.canvas.draw_idle()
    return

from scipy.interpolate import interp1d

# def generate_line_list_table(xvals):
#     from astropy.table.table import QTable
#     qtable = QTable()
#     qtable['line_center'] = [xvals[i] for i]
#     list(
#         itertools.chain(
#             *[spectrum.spectral_axis.value[emission_inds],
#               spectrum.spectral_axis.value[absorption_inds]]
#         )) * spectrum.spectral_axis.unit
#     qtable['line_type'] = ['emission'] * len(emission_inds) + \
#                           ['absorption'] * len(absorption_inds)
#     qtable['line_center_index'] = list(
#         itertools.chain(
#             *[emission_inds, absorption_inds]))

#     return qtable


def my_find_lines_derivative(y, flux_threshold=None):
    ## Search for the emission lines:
    ny = np.copy(y)
    nx = np.arange(len(y))
    
    ## Let's try interpolating to be more accurate
    # x = np.arange(len(ny))
    # f = interp1d(x, ny, kind='quadratic')
    # nx = np.linspace(0, len(ny)-1, 100*len(ny))
    # ny = f(nx)
    diffs = np.diff(ny)

    diffs[diffs>0] = 1
    diffs[diffs<0] = -1
    diffs = np.diff(diffs)
    idx = np.where(diffs<0)
    nidx = idx[0]+1
    diffnidx = np.diff(nx[nidx])

    issuewith = np.where(diffnidx<2)[0]
    if len(issuewith)>0:
        for ii in issuewith[::-1]:
            iii = nidx[ii]
            iiii = nidx[ii+1]
            if ny[iiii]>ny[iii]: todelete=ii
            if ny[iiii]<=ny[iii]: todelete=ii+1
            nidx = np.delete(nidx, todelete)

    # from IPython import embed
    # embed()

    idxval = np.where(ny[nidx] > flux_threshold)
    subx = nx[nidx]
    valid_idx = subx[idxval] - 1 ## Not even sure about - 1 but it is to be consitent with the function I am replacing
    return valid_idx

def peak_finder(y, ths=1/20, med=1):
    '''This function finds the bin corresponding to the maxima in the y array'''
    ny = np.copy(y)
    nx = np.arange(len(y))
    
    ## Let's try interpolating to be more accurate
    x = np.arange(len(ny))
    # f = interp1d(x, ny, kind='quadratic')
    # nx = np.linspace(0, len(ny)-1, 100*len(ny))
    # ny = f(nx)
    # nx, ny = xresample(x, ny, 5)
    nx, ny = x, ny

    ## Implement a threshold for the detection
    ny[ny<np.max(ny)*ths] = 0
    # ny[ny<70] = 0

    medy = np.median(ny)
    mask = ny<med*medy#+medy/2
    # nx[nx<medx] = 0

    diffs = np.diff(ny)
    diffs[diffs>0] = 1
    diffs[diffs<0] = -1
    diffs = np.diff(diffs)
    diffs[mask[:-2]] = 0
    idx = np.where(diffs<0)
    nidx = idx[0]+1
    # x[nidx-1] - x[nidx]

    diffnidx = np.diff(nx[nidx])

    issuewith = np.where(diffnidx<2)[0]
    if len(issuewith)>0:
        for ii in issuewith[::-1]:
            iii = nidx[ii]
            iiii = nidx[ii+1]
            if ny[iiii]>ny[iii]: todelete=ii
            if ny[iiii]<=ny[iii]: todelete=ii+1
            nidx = np.delete(nidx, todelete)

    # from IPython import embed
    # embed()

    # plt.figure()
    # plt.plot(np.arange(len(y)), y)
    # plt.plot(nx, ny, ',')
    # for i in range(len(nidx)):
    #     plt.axvline(nx[nidx[i]])
    # plt.show()



    return nx[nidx], ny[nidx]

from astropy.io import fits

import pickle
def get_id_lines_template(extract1d,linelist,continuum_rejection_low,continuum_rejection_high,
                          continuum_rejection_iterations,continuum_rejection_order,threshold_factor,
                          window,id_lines_order,id_lines_rejection_iterations,id_lines_rejection_sigma,
                          id_lines_tol_angs,id_lines_template_fiddle,id_lines_template0,resolution_order,
                          resolution_rejection_iterations, idlinestemplate_datafile):
    import numpy as np
    import astropy.units as u
    from specutils.fitting import find_lines_threshold
    from specutils.fitting import find_lines_derivative
    from specutils.spectra import Spectrum1D
    import matplotlib.pyplot as plt
    from copy import deepcopy
    from astropy.modeling import models,fitting

    global firstvalue, secondvalue
    firstvalue, secondvalue = None, None

    continuum0,spec_contsub,fit_lines=get_fitlines(extract1d,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,threshold_factor,window,continuum_rejection_order)
    idlinestemplate_data = [continuum0, spec_contsub.spectral_axis.value, spec_contsub.flux.value, fit_lines]
    pickle.dump(idlinestemplate_data,open(idlinestemplate_datafile,'wb')) 

    line_centers=[fit_lines.fit[j].mean.value for j in range(0,len(fit_lines.fit))]
    id_lines_pix=[]
    id_lines_wav=[]
    id_lines_species=[]
    id_lines_used=[]
    order=[id_lines_order]
    rejection_iterations=[id_lines_rejection_iterations]
    rejection_sigma=[id_lines_rejection_sigma]
    func=[models.Legendre1D(degree=1)]
    rms=[]
    npoints=[]
    if id_lines_template_fiddle:
        id_lines_used=np.where(id_lines_template0.wav.mask==False)[0].tolist()
        id_lines_pix=[id_lines_template0.fit_lines.fit[q].mean.value for q in id_lines_used]
        id_lines_wav=id_lines_template0.wav[id_lines_used].tolist()
        func=[id_lines_template0.func]
        order=[id_lines_template0.func.degree]
        rms=[id_lines_template0.rms]
        npoints=[id_lines_template0.npoints]

    fig=plt.figure(1)
    ax1,ax2=m2fs.plot_id_lines(extract1d,continuum0,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func[len(func)-1],fig)
#
    print('press \'m\' to ID line nearest cursor \n')
    print('press \'d\' to delete ID for line nearest cursor \n')
    print('press \'o\' to change order of polynomial \n')
    print('press \'r\' to change rejection sigma factor \n')
    print('press \'t\' to change number of rejection iterations \n')
    print('press \'g\' to re-fit wavelength solution \n')
    print('press \'l\' to add lines from linelist according to fit \n')
    print('press \'i\' identify regions of the spectrum to remove \n')
    print('press \'q\' to quit \n')
    print('press \'.\' to print cursor position and position of nearest line \n')
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) # PIC This disables any key that was not set up.
    cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_id_lines(event,[extract1d,continuum0,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints,id_lines_tol_angs,fig]))
    plt.show()
    plt.close()

    wav=np.ma.masked_array(np.full((len(fit_lines.fit)),-999,dtype='float'),mask=np.full((len(fit_lines.fit)),True,dtype=bool))

    for j in range(0,len(id_lines_pix)):
        wav[id_lines_used[j]]=id_lines_wav[j]
        wav.mask[id_lines_used[j]]==False

    resolution,resolution_rms,resolution_npoints=m2fs.get_resolution(deepcopy(fit_lines),deepcopy(wav),resolution_order,resolution_rejection_iterations)

    return m2fs.id_lines(aperture=extract1d.aperture,fit_lines=fit_lines,wav=wav,
                         func=func[len(func)-1],rms=rms[len(rms)-1],npoints=npoints[len(npoints)-1],
                         resolution=resolution,resolution_rms=resolution_rms,
                         resolution_npoints=resolution_npoints)


def on_key_id_lines(event,args_list):
    import numpy as np
    from . import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils.spectra import Spectrum1D
    from astropy.modeling import models,fitting
    from copy import deepcopy

    #print('you pressed ',event.key)

    keyoptions = ['.', 'm', 'd', 'o', 'r', 't', 'z', 'g', 'l', 'q', 'p', 'i']

    global id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints
    extract1d,continuum,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints,id_lines_tol_angs,fig=args_list

    global firstvalue, secondvalue

    if event.key in keyoptions:
        if 'super' in event.key=='.':
            pass
        if event.key=='.':
            x0=event.xdata
            y0=event.ydata
            print('cursor pixel = ',str(x0))
            dist=(line_centers-x0)**2
            best=np.where(dist==np.min(dist))[0][0]
            print('nearest line pixel = ',str(line_centers[best]))
            used=np.array(np.where(id_lines_pix==line_centers[best])[0])
            if len(used)>0:
                print('wavelength from line list =',str(np.array(id_lines_wav)[used][0]))
            else:
                print('line not yet identified')
            print('function wavelength =',str(func[len(func)-1](line_centers[best])))

        if event.key=='i':#delete nearest boundary point
            # print('delete point nearest (',event.xdata,event.ydata,')')
            # val1 = input('Enter the first bin of the window to delete')
            # val2 = input('Enter the second bin of the window to delete')
            # val1 = int(val1)
            # val2 = int(val2)
            if firstvalue is None:
                firstvalue = event.xdata
            else:
                secondvalue = event.xdata
            
            # print('You pressed i:')
            # print('{} {}'.format(firstvalue, secondvalue))

            if firstvalue is not None:
                if secondvalue is not None:
                    val1 = firstvalue
                    val2 = secondvalue
                    if val2<val1:
                        lval=val2
                        hval=val1
                    else:
                        lval=val1
                        hval=val2
                    extract1d.spec1d_mask[int(lval):int(hval)] = True
                    # from IPython import embed
                    # embed()
                    for j in range(len(line_centers)-1, 0, -1):
                        if (line_centers[j]>lval) & (line_centers[j]<hval):
                            line_centers.remove(line_centers[j])
                            fit_lines.fit.remove(fit_lines.fit[j])
                            fit_lines.initial.remove(fit_lines.initial[j])
                            fit_lines.subregion.remove(fit_lines.subregion[j])
                            fit_lines.realvirtual.remove(fit_lines.realvirtual[j])
                    firstvalue = None
                    secondvalue = None

        if event.key=='m':
            x0=event.xdata
            y0=event.ydata
            print('cursor pixel = ',str(x0))
            dist=(line_centers-x0)**2
            best=np.where(dist==np.min(dist))[0][0]
            print('nearest line pixel = ',str(line_centers[best]))
            command=""
            command=input('enter wavelength in Angstroms \n')
            if command=='':
                print('no information entered')
            else:
                id_lines_pix.append(line_centers[best])
                id_lines_wav.append(float(command))
                id_lines_used.append(best)

        if event.key=='d':#delete nearest boundary point
            print('delete point nearest (',event.xdata,event.ydata,')')
            if len(id_lines_pix)>0:
                dist=(id_lines_pix-event.xdata)**2
                best=np.where(dist==np.min(dist))[0][0]
                del id_lines_pix[best]
                del id_lines_wav[best]
                del id_lines_used[best]
            else:
                print('no ID\'d lines to delete!')

        if event.key=='o':
            print('order of polynomial fit is ',order[len(order)-1])
            command=input('enter new order (must be integer): ')
            if command=='':
                print('keeping original value')
            else:
                order.append(int(command))
            func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        if event.key=='r':
            print('rejection sigma is ',rejection_sigma[len(rejection_sigma)-1])
            command=input('enter new rejection sigma (float): ')
            if command=='':
                print('keeping original value')
            else:
                rejection_sigma.append(float(command))
            func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        if event.key=='t':
            print('number of rejection iterations is ',rejection_iterations[len(rejection_iterations)-1])
            command=input('enter new rejection iterations = (integer): ')
            if command=='':
                print('keeping original value')
            else:
                rejection_iterations.append(int(command))
            func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        if event.key=='z':
            for q in range(1,len(id_lines_pix)):
                del(id_lines_pix[len(id_lines_pix)-1])
            for q in range(1,len(id_lines_wav)):
                del(id_lines_wav[len(id_lines_wav)-1])
            for q in range(1,len(id_lines_used)):
                del(id_lines_used[len(id_lines_used)-1])
            for q in range(1,len(order)):
                del(order[len(order)-1])
            for q in range(1,len(rejection_iterations)):
                del(rejection_iterations[len(rejection_iterations)-1])
            for q in range(1,len(rejection_sigma)):
                del(rejection_sigma[len(rejection_sigma)-1])
            for q in range(1,len(func)):
                del(func[len(func)-1])

        if event.key=='g':
            func0,rms0,npoints0,y=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        # if event.key=='p':
        #     # from IPython import embed
        #     # embed()
        #     print('plotting the residuals and the fit')
        #     fig2, axtmp = plt.subplots(1,1)
        #     plt.figure()
        #     plt.plot(id_lines_pix, id_lines_wav)
        #     plt.plot(id_lines_pix, func[0](id_lines_pix))
        #     plt.show()
        
        if event.key=='l':
            func0,rms0,npoints0,y=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
            new_pix,new_wav,new_used=m2fs.line_id_add_lines(deepcopy(linelist),deepcopy(line_centers),deepcopy(id_lines_used),deepcopy(func0),id_lines_tol_angs)

            for i in range(0,len(new_pix)):
                id_lines_pix.append(new_pix[i])
                id_lines_wav.append(new_wav[i])
                id_lines_used.append(new_used[i])
            func0,rms0,npoints0,y=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)
        if event.key=='q':
            plt.close(event.canvas.figure)
            # return
        else:
            pass
    
    ax1,ax2=m2fs.plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func[len(func)-1],fig)
    fig.canvas.draw_idle()
    return

def plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func,fig):
    import numpy as np
    from astropy import units as u

    # fig.clf() ## PIC test - that appears to correct the weird plotting issue.

    xlim=[np.min(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),np.max(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])]
    ax1=fig.add_subplot(211)
    ax1.cla()
    ax1.set_xlim([np.min(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),np.max(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])])
    ax1.plot(extract1d.spec1d_pixel[extract1d.spec1d_mask==False],extract1d.spec1d_flux[extract1d.spec1d_mask==False],color='k',lw=0.5)
    ax1.plot(extract1d.spec1d_pixel[extract1d.spec1d_mask==False],continuum(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),color='b',lw=0.3)
    ax1.set_xlabel('pixel')
    ax1.set_ylabel('counts')
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_xlim(xlim)
    for j in range(0,len(fit_lines.fit)):
        use=np.where((extract1d.spec1d_pixel>=fit_lines.subregion[j].lower.value)&(extract1d.spec1d_pixel<=fit_lines.subregion[j].upper.value))[0]
        sub_spectrum_pixel=extract1d.spec1d_pixel[use]*u.AA
        y_fit=fit_lines.fit[j](sub_spectrum_pixel).value+continuum(sub_spectrum_pixel.value)
        ax1.plot(sub_spectrum_pixel,y_fit,color='r',lw=0.3)
        ax1.axvline(x=fit_lines.fit[j].mean.value,linestyle=':',color='k',lw=0.3)
    for j in range(0,len(id_lines_pix)):
         ax1.axvline(x=id_lines_pix[j],color='g',lw=1,linestyle='-')
#    ax1.text(0,1.01,'n_points='+str(npoints),transform=ax1.transAxes)
#    ax1.text(0,1.06,'order='+str(order),transform=ax1.transAxes)
#    ax1.text(0,1.11,'rms='+str(rms),transform=ax1.transAxes)
#    ax1.text(1,1.01,'aperture='+str(j+1),horizontalalignment='right',transform=ax1.transAxes)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')

    ax2=fig.add_subplot(212)
    ax2.cla()
#    ax2.scatter(id_lines_pix,id_lines_wav,color='k',s=3)
    ax2.scatter(id_lines_pix,id_lines_wav-func(id_lines_pix),color='k',s=3)
    x=extract1d.spec1d_pixel[extract1d.spec1d_mask==False]
#    ax2.plot(x,func(x),color='r')
    ax2.set_xlim(xlim)
    ax2.set_xlabel('pixel')
    ax2.set_ylabel(r'$\lambda$ [Angstroms]')

    return ax1,ax2

def get_meansky(throughputcorr_array,wavcal_array,plugmap):
    '''
    throughputcorr_array:   All the spectra from all the apertures
    wavcal_array        :   Wavelength solution for all of the apertures
    plugmap             :   Map containing the position of the skies
    '''

    import numpy as np
    import scipy
    import astropy.units as u
    from specutils.spectra import Spectrum1D
    from astropy.modeling import models,fitting
    import matplotlib.pyplot as plt
    from astropy.nddata import StdDevUncertainty

    ## Iterate through all the apertures
    ## Take the spectrum, the wavelength and mask
    ## Search for the wav_min and wav_max marking the position
    ## of the first and last unmasked spectrum

    wav_min=[]
    wav_max=[]
    npix=[]
    use=[]
    for i in range(0,len(throughputcorr_array)):
        pix=throughputcorr_array[i].spec1d_pixel
        wav=np.array(wavcal_array[i].wav)
        keep=np.where(throughputcorr_array[i].spec1d_mask==False)[0]
        if ((len(wav)>0)&(len(keep)>0)):
            wav_min.append(np.min(wav[throughputcorr_array[i].spec1d_mask==False]))
            wav_max.append(np.max(wav[throughputcorr_array[i].spec1d_mask==False]))
            npix.append(len(np.where(throughputcorr_array[i].spec1d_mask==False)[0]))
            use.append(1)
        else:
            wav_min.append(1.e+10)
            wav_max.append(-1.e+10)
            npix.append(0)
            use.append(0)
    wav_min=np.array(wav_min)
    wav_max=np.array(wav_max)
    npix=np.array(npix)
    use=np.array(use)


    ## Initialize the meansky object - placing a zero in the fields
    meansky=Spectrum1D(spectral_axis=np.array([0.])*u.AA,flux=np.array([0.])*u.electron,uncertainty=StdDevUncertainty(np.array([np.inf])),mask=np.array([True]))
    ## If there are multiple apertures
    ## If we selected multiple apertures to use
    ## Create a linspace wavelength array of length 10 times the median of the number of pixels to use
    ## select all the skies
    ## For each sky
    ## interpolate the flux, uncertainties and mask - what is the interpolation of the mask gonna give?
    ## Store that in arrays
    if len(throughputcorr_array)>0:
        if len(wav_min[use==1])>0:
            wav0=np.linspace(np.min(wav_min[use==1]),np.max(wav_max[use==1]),int(np.median(npix[use==1])*10))
#            id_nlines=np.array([len(np.where(id_lines_array[q].wav>0.)[0]) for q in range(0,len(id_lines_array))],dtype='int')
            skies=np.where(((plugmap['objtype']=='SKY')|(plugmap['objtype']=='unused'))&(np.array([len(wavcal_array[q].wav) for q in range(0,len(wavcal_array))])>0))[0]
            targets=np.where((plugmap['objtype']=='TARGET')&(np.array([len(wavcal_array[q].wav) for q in range(0,len(wavcal_array))])>0))[0]

            sky_flux_array=[]
            sky_err_array=[]
            sky_mask_array=[]

            for i in range(0,len(skies)):

                # from IPython import embed
                # embed()
#            use=np.where(throughputcorr_array[skies[i]].spec1d_mask==False)[0]
                wav=wavcal_array[skies[i]].wav
                #################################################
                #################################################
                ## PIC:
                ## /!\ CAUTION: Here I hardcode some stuff that really shouldn't
                #################################################
                # bounds = [6565.08, 6568.47]
                # idxinvalid = np.where((wav>bounds[0])&(wav<bounds[1]))
                # throughputcorr_array[skies[i]].spec1d_flux.value[idxinvalid] = 1
                #################################################
                #################################################
                if len(wav)>0:
                    sky_flux_array.append(fftinterp(wav0,wav,throughputcorr_array[skies[i]].spec1d_flux))
                    sky_err_array.append(fftinterp(wav0,wav,throughputcorr_array[skies[i]].spec1d_uncertainty.quantity.value))
                    sky_mask_array.append(fftinterp(wav0,wav,throughputcorr_array[skies[i]].spec1d_mask))
            sky_flux_array=np.array(sky_flux_array)
            sky_err_array=np.array(sky_err_array)
            sky_mask_array=np.array(sky_mask_array)

            ## For each wavelength bin, take data in each aperture in one array (column)
            ## Same thing with mask
            ## For each aperture,
            ## In the column, see if some points are out of bounds and mask these out
            ## Take the median on the valid bins.

            sky0_flux=[]
            sky0_err=[]
            sky0_mask=[]
            for i in range(0,len(wav0)):
                # vec=np.array([sky_flux_array[q][i] for q in range(0,len(sky_flux_array))],dtype='float')
                vec = []
                for q in range(len(sky_flux_array)):
                    vec.append(sky_flux_array[q][i])
                vec=np.array(vec,dtype='float')
                # vec_mask=np.array([sky_mask_array[q][i] for q in range(0,len(sky_flux_array))],dtype='bool')
                vec_mask = []
                for q in range(len(sky_mask_array)):
                    vec_mask.append(sky_mask_array[q][i])
                vec_mask = np.array(vec_mask, dtype='bool')
                for j in range(0,len(vec)):
                    if wav_min[skies[j]]>wav0[i]:
                        vec_mask[j]=True
                    if wav_max[skies[j]]<=wav0[i]:
                        vec_mask[j]=True
                med=np.median(vec[vec_mask==False])
                mad=np.median(np.abs(vec[vec_mask==False]-med))*1.4826*np.sqrt(np.pi/2.)/np.sqrt(float(len(np.where(vec_mask==False)[0])))

                sky0_flux.append(med)
                sky0_err.append(mad)
                if med!=med:
                    sky0_mask.append(True)
                else:
                    sky0_mask.append(False)
            sky0_flux=np.array(sky0_flux)
            sky0_err=np.array(sky0_err)
            sky0_mask=np.array(sky0_mask)
        else:
            wav0=[0.]
            sky0_flux=np.array([0.])
            sky0_err=np.array([0.])
            sky0_mask=np.array([True])

        ## Populate the object
        meansky=Spectrum1D(spectral_axis=wav0*u.AA,flux=sky0_flux*u.electron,uncertainty=StdDevUncertainty(sky0_err),mask=sky0_mask)
    return meansky

from . import normalizationTools as norm_tools


def fit_continuum(wvl, flux, wvl_ref = None, window_size=200, p=95, degree=3, 
                m=0.5):
    '''
    Returns the coefficients of the polynomial fitted on the pth percentiles
    computed on windows of length window_size.
    Input parameters:
    - wvl           :   [1d-array] Wavelength grid.
    - flux          :   [1d-array] Array to normalize.
    - window_size   :   [int] desired length of the window size used 
                        to split flux.
    - p             :   [int] percentile value (0--100) computed on 
                        window_size sections of the flux
    Output parameters:
    - continuum     :   Continuum computed on the input wvl grid.
    - [_]           :   [list] containing :
        -- wvls             Wavelengths associated to percentiles
        -- percentiles      Percentiles
    '''
    if wvl_ref is None:
        wvl_init = np.copy(wvl)
    else:
        wvl_init = wvl_ref
    _flux = np.copy(flux)
    ## THE SECTION IS COMMENTED BECAUSE IT COULD ACTUALLY INTRODUCE ERRORS IN
    ## THE PERCENTILES COMPUTATION
    # ## Initial step: We try to ignore the NaNs by removing them
    # idx = np.where(np.isnan(_flux))
    # _flux = np.delete(_flux, idx)
    # wvl = np.delete(wvl, idx)
    ## First step is to split the input array in windows of a given size
    if window_size>len(_flux):
        raise Exception("Window size must be smaller than len(x)")
    if len(flux)%window_size != 0:
        nb_of_sections = int(np.floor(len(_flux)/window_size))
        new_length = window_size * nb_of_sections
        diff_length = len(_flux) - new_length
        _flux = _flux[diff_length:]
        wvl = wvl[diff_length:]
        warnings.warn('len(flux)/window_size does not result in equal division. Cutting input arrays')
    else :
        nb_of_sections = len(_flux)//window_size
    ## Use this nunber of sections to split the input array
    flux_splited = np.split(_flux, nb_of_sections)
    wvl_splited = np.split(wvl, nb_of_sections)
    ## TODO: Improve selection of wavelength associated to the percentile
    wvls = np.mean(wvl_splited, axis=1) 
    percentiles = np.nanpercentile(flux_splited, p, axis=1)
    ## Determine the associated wavelength
    new_wvls = []
    for i in range(len(flux_splited)):
        _f = flux_splited[i]
        _w = wvl_splited[i]
        _p = percentiles[i]
        if np.isnan(_p):
            _w = np.mean(_w)
        elif not np.isnan(_p):
            _f[np.isnan(_f)] = 0
            idx = np.argsort(_f)
            _f = _f[idx]
            _w = _w[idx]
            _w = _w[_f>=_p][0]
        new_wvls.append(_w)
    wvls = np.array(new_wvls)
    ## Removing values varying of 10% from pth percentile of entire flux
    ## TODO: improve by rejecting points based on percentile environment,
    ## i.e. what the values of percentile are
    # median = np.mean(percentiles)
    # idx = np.where(percentiles>(median+m*median))
    # wvls = np.delete(wvls, idx)
    # percentiles = np.delete(percentiles, idx)
    # idx = np.where(percentiles<(median-m*median))
    # wvls = np.delete(wvls, idx)
    # percentiles = np.delete(percentiles, idx)
    # ## VERY IMPORTANT -- Removing values that are still NaN in percentiles
    # idx = np.where(np.isnan(percentiles))
    # wvls = np.delete(wvls, idx)
    # percentiles = np.delete(percentiles, idx)
    if len(percentiles)==0:
        continuum = np.ones(np.shape(wvl_init))*np.mean(_flux)
        coeffs = np.array([0, 1])
    elif len(percentiles)<degree:
        x = norm_tools.normalize_axis(wvls, wvls)
        coeffs = norm_tools.fit_1d_polynomial(x, percentiles, degree=len(percentiles))
        continuum = norm_tools.polynom(norm_tools.normalize_axis(wvl_init, wvls), coeffs)
    else:
        x = norm_tools.normalize_axis(wvls, wvls)
        coeffs = norm_tools.fit_1d_polynomial(x, percentiles, degree=degree)
        continuum = norm_tools.polynom(norm_tools.normalize_axis(wvl_init, wvls), coeffs)
    ## Fit n degree polynomial on the percentiles
    ## CAUTION: Here we add a step to avoid fitting issues. We normalize the
    ## wvls to be fitted. 
    # x = normalize_axis(wvls, wvls)
    # coeffs = fit_1d_polynomial(x, percentiles, degree=degree)
    # continuum = polynom(normalize_axis(wvl_init, wvls), coeffs)
    return continuum, [wvls,percentiles], coeffs

def normalize(x, y, newxaxis=None):
    _x = np.copy(x); _y = np.copy(y)

    if _x.shape!=_y.shape:
        raise Exception('fdump.normalize -> Input arrays should have the same dimensions.\n'
                        + 'Instead they have {} and {}'.format(_x.shape, _y.shape))

    ## Select the points of the continuum
    med = np.nanmedian(_y)
    rms = np.nanstd(_y)
    idx = ((_y>med-100*rms) & (_y<med+100*rms) & (_y!=0))

    if len(idx[idx==True])<5:
        # print('Not enough points')
        c = np.ones(len(y))
        ps = [np.nan, np.nan]
        return _x, _y, c, ps
    _y = _y[idx]
    _x = _x[idx]
    
    c, ps, coeffs = fit_continuum(
                wvl=_x,
                flux=_y,
                window_size=9,
                p=50,
                degree=1,
                m=50)
    
    # print(ps)
    # plt.figure()
    # plt.plot(_x, _y)
    # plt.show()
    

    # _y / c

    # c, ps, coeffs = fit_continuum(
    #         wvl=_x,
    #         flux=_y/c,
    #         window_size=9,
    #         p=50,
    #         degree=3,
    #         m=50)

    if newxaxis is not None:
        nx = norm_tools.normalize_axis(newxaxis, newxaxis)
    else:
        nx = norm_tools.normalize_axis(x, x)

    c = norm_tools.polynom(nx, coeffs)
    # print(coeffs)
    # plt.figure()
    # plt.plot(c)
    # plt.show()
    # print('coucou')
    # print(c)
    # print(ps[0])

    return _x, _y, c, ps

def new_normalize(inx, iny, degree=4, sigma=3, idx=None):
    """Improve normalization function, based on sigma clipping and 
    Input parameters:
    - inx   :   wavelength solution
    - iny   :   normalized flux
    - sigma :   sigma for sigma clipping
    - degree:   degree of the fitted polynomial
    - idx   :   optional, valid bins to use for the continuum fit"""

    if idx is not None:
        _inx = np.copy(inx)[idx]
        _iny = np.copy(iny)[idx]
    else:
        _inx = np.copy(inx)
        _iny = np.copy(iny)

    # plt.figure()
    # plt.plot(_inx, _iny)
    ## Perform an initial normalization
    _, _, _, ps = normalize(_inx, _iny, newxaxis=_inx)
    # print(_inx, _iny, ps)
    nsps = norm_tools.normalize_axis(ps[0], ps[0])
    coeffs = norm_tools.fit_1d_polynomial(nsps, ps[1], degree=degree)
    nsinx = norm_tools.normalize_axis(_inx, ps[0])
    _c = norm_tools.polynom(nsinx, coeffs)
    ## Compute initial residuals
    res = _iny - _c
    sqrtres = np.sqrt(res**2)
    ## Sigma clip
    while (np.nanmax(sqrtres) > (sigma * np.nanstd(res))):
        idxtomask = np.where(sqrtres==np.nanmax(sqrtres))
        _iny[idxtomask] = np.nan
        _, _, _, ps = normalize(_inx, _iny, newxaxis=inx)
        nsps = norm_tools.normalize_axis(ps[0], ps[0])
        coeffs = norm_tools.fit_1d_polynomial(nsps, ps[1], degree=degree)
        nsinx = norm_tools.normalize_axis(_inx, ps[0])
        _c = norm_tools.polynom(nsinx, coeffs)
        res = _iny - _c# - 1
        sqrtres = np.sqrt(res**2)

    nsinx = norm_tools.normalize_axis(inx, ps[0])
    c = norm_tools.polynom(nsinx, coeffs)
    return c


def normalize_old(wvl, flux, err, p=95, hws=150, degree=3):
    # Define local vars
    _w = wvl
    _f = flux

    med = np.median(flux)
    rms = np.std(flux)
    _f[_f>med+rms] = np.nan
    _f[_f<med-rms] = np.nan

    def return_continuum(degree):
        c, ps = norm_tools.fit_continuum(
            wvl=_w,
            flux=_f,
            window_size=hws,
            p=p,
            degree=degree,
            m=50)
        return c, ps
    d = degree
    success = False
    while not success and d > 0:
        try:
            c, ps = return_continuum(d)
            success = True
        except:
            d = d-1
    if d == 0:
        c = np.nanpercentile(flux, 95) *\
            np.ones(len(_f))
        ps = np.array([[0], [0]])
        # raise Exception('Negative degree')
    wps = ps[0]
    ps = ps[1]
    # Correct with continuum
    norm_flux = flux/c
    norm_err = err/c
    return c

def grab_continuum(wvl, spectrum):
    '''Function returning the continuum of a simple spectrum
    This version should be an improvement over the previous versions'''
    if len(wvl)!=len(spectrum):
        raise Exception('grab_continuum: input array should have the same sizes.')
    c = np.ones(len(spectrum))
    rmsarray = []
    for k in range(500):
        ## define the acceptable devaition from the continuum
        dcont = 0.5
        degree = 1
        if k>8:
            dcont = 0.2
            degree = 2
        if k>16:
            dcont = 0.1
            degree = 2
        if k>24:
            dcont = 0.05
            degree = 2
        if k>32:
            dcont = 0.03
            degree = 2
        ## Copy the input arrays
        w, s = np.array(wvl, dtype=float), np.array(spectrum, dtype=float)
        ## Compute the smoothed spectrum
        test, s = norm_tools.moving_median(s, 15, btd=0)
        ## Mask the pixels strickly equal to 0
        idx = np.where(s==0.)
        s[idx] = np.nan
        ## Mask the pixels that below 0.9 or larger than 1.2 in normalized flux
        if k!=0:
            idx = np.where((s/c)<1-dcont)
            s[idx] = np.nan
            idx = np.where((s/c)>1+dcont)
            s[idx] = np.nan
        ## Compute median and RMS
        med = np.nanmedian(s/c)
        rms = np.nanstd(s/c)
        ## Mask pixels above and below 3*rms
        idx = np.where((s/c)<(med-3*rms))
        s[idx] = np.nan
        idx = np.where((s/c)>(med+3*rms))
        s[idx] = np.nan
        ## Fit the continuum on the smoothed spectrum
        c, ps, coeffs = fit_continuum(wvl=w, flux=s, window_size=9, p=50, degree=degree)
        ## Store the RMS values
        rmsarray.append(rms)
        ## If we have the same RMS 10 times in a row, exit loop
        if k>10:
            if round(rms, 4)==round(np.sum(rmsarray[-10:])/10, 4):
                break ## We are not improving by looping further
        if k>490:
            print('grab_continuum: reached the end of the loop')
    return c, ps

from specutils.spectra import Spectrum1D
def get_throughput_continuum(twilightstack_array,twilightstack_wavcal_array,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order):
    
    '''Function designed to fit the throughput continuum?'''
    
    # import numpy as np
    # import astropy.units as u
    # from specutils.fitting import find_lines_threshold
    # from specutils.fitting import find_lines_derivative
    # from copy import deepcopy
    # import matplotlib.pyplot as plt
    # from astropy.modeling import models
    # from scipy import interpolate
    # import scipy

    continuum_array=[]
    wav_min=[]
    wav_max=[]
    npix=[]
    use=[]
    ## For each aperture
    ## If there are more the 1 valid bins to and the wavelength array is longer than 0
    ##  
    for i in range(0,len(twilightstack_array)):
        wav=twilightstack_wavcal_array[i].wav
        if ((len(np.where(twilightstack_array[i].spec1d_mask==False)[0])>1)&(len(wav)>0)):
            
            # polomask = np.ones(len(twilightstack_array[i].spec1d_mask))
            # polomask[100:200] = 0
            polomask = twilightstack_array[i].spec1d_mask.astype(int)
            spec1d0=Spectrum1D(spectral_axis=wav*u.AA,
                               flux=deepcopy(twilightstack_array[i].spec1d_flux)*u.electron,
                               uncertainty=deepcopy(twilightstack_array[i].spec1d_uncertainty),
                               mask=deepcopy(polomask))
                            #    mask=deepcopy(twilightstack_array[i].spec1d_mask))
            ## PIC: Trying to ignore the apertures for which we have an issue
            # print("Now for aperture {}".format(i))
            # try:
            continuum0,rms0=get_continuum(spec1d0,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order)
            continuum_array.append(continuum0)
            wav_min.append(np.min(wav[twilightstack_array[i].spec1d_mask==False]))
            wav_max.append(np.max(wav[twilightstack_array[i].spec1d_mask==False]))
            npix.append(len(np.where(twilightstack_array[i].spec1d_mask==False)[0]))
            use.append(1)
            # print(i)
            # if i==1:
            #     from IPython import embed
            #     embed()
            # except:
            #     print(i)
            #     from IPython import embed
            #     embed()
            #     continuum0,rms0=get_continuum(spec1d0,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order,debug=True)

            #     print("Issue for aperture {}".format(i))
            #     wav_min.append(-999.)
            #     wav_max.append(-999.)
            #     npix.append(0)
            #     use.append(0)
            #     continuum_array.append(models.Chebyshev1D(degree=1))
        else:
            wav_min.append(-999.)
            wav_max.append(-999.)
            npix.append(0)
            use.append(0)
            continuum_array.append(models.Chebyshev1D(degree=1))
#        plt.plot(wav[twilightstack_array[i].spec1d_mask==False],twilightstack_array[i].spec1d_flux[twilightstack_array[i].spec1d_mask==False],color='k',lw=0.3)
#        plt.plot(wav,continuum0(wav),color='r',lw=0.3)
#        plt.ylim([0,30000])
#        plt.show()
#        plt.close()
    use=np.array(use)
    wav_min=np.array(wav_min)
    wav_max=np.array(wav_max)
    npix=np.array(npix)
    continuum_array=np.array(continuum_array)
    if((len(twilightstack_array)>0)&(len(np.where(use==1)[0])>0)):
        wav0=np.linspace(np.min(wav_min[use==1]),np.max(wav_max[use==1]),int(np.median(npix[use==1])))
        flux0=[]#np.zeros(len(wav0))
        mask0=[]
        for j in range(0,len(wav0)):
            keep=np.where((wav_min<=wav0[j])&(wav_max>=wav0[j])&(use==1))[0]
            if len(keep)>0:
                flux0.append(np.median([continuum_array[q](wav0[j]) for q in keep]))
                mask0.append(False)
            else:
                flux0.append(0.)
                mask0.append(True)
        flux0=np.array(flux0)    
        mask0=np.array(mask0)
        spec1d0=Spectrum1D(spectral_axis=wav0*u.AA,flux=flux0*u.electron,mask=mask0)
        continuum0,rms0=m2fs.get_continuum(spec1d0,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order)
        
#        plt.plot(spec1d0.spectral_axis,spec1d0.flux)
#        plt.plot(spec1d0.spectral_axis,continuum0(spec1d0.spectral_axis.value))
#        plt.show()
#        plt.close()
#        np.pause()
    else:
        continuum0=models.Chebyshev1D(degree=1)
        rms0=-999.

    return continuum_array,continuum0


from astropy.modeling import models,fitting
def get_continuum(spec1d,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order,debug=False):
    # import numpy as np
    # import astropy.units as u
    # from copy import deepcopy
    # import copy
    # import matplotlib.pyplot as plt
    if debug:
        from IPython import embed
        embed()


#    continuum_init=models.Polynomial1D(degree=10)
    ## Initialize the continuum object and fitter
    continuum_init=models.Chebyshev1D(degree=continuum_rejection_order)
    fitter=fitting.LinearLSQFitter()
#    lamb=(np.arange(len(spec1d.data),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
    lamb=spec1d.spectral_axis#(np.arange(len(spec1d.spectral_axis),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
    
    ## PIC: for some reason it happens that values are NaN but were not masked.
    idx = np.where(np.isnan(spec1d.flux))
    spec1d.mask[idx] = True
    
    y=np.ma.masked_array(spec1d.flux,mask=spec1d.mask.astype(bool))

    ## For each iteration in the continuum_rejection_iterations (???)
    ## If we do have enough points 
    for q in range(0,continuum_rejection_iterations):
        ## PIC, we must have at least 4 points to fit a 4 degree polynomial, so the
        ## following safeguard cannot actually work.
        ## I replace:
        # if len(np.where(y.mask==False)[0])>0: 
        ## with:
        if len(np.where(y.mask==False)[0])>continuum_rejection_order:         
            continuum=fitter(continuum_init,lamb.value[y.mask==False],y[y.mask==False])
        else:
            continuum=continuum_init

#        continuum=fitter(continuum_init,lamb.value,y)
        ## Compute the RMS between the continuum and the twilight
        ## Mask outliers
        ## Rerun (this is a sigma clipping pretty much...)
        rms=np.sqrt(np.mean((y.data.value[y.mask==False]-continuum(lamb.value)[y.mask==False])**2))    
        outlier=(np.where(spec1d.data-continuum(lamb.value)>continuum_rejection_high*rms))[0]#reject only high outliers (emission lines)
        y.mask[outlier]=True
        outlier=(np.where(spec1d.data-continuum(lamb.value)<continuum_rejection_low*rms))[0]#reject only high outliers (emission lines)
        y.mask[outlier]=True
#        outlier=(np.where(spec1d.data-continuum(lamb.value)>1.*spec1d.uncertainty.quantity.value))[0]#reject only high outliers (emission lines)
#        y.mask[outlier]=True

        # print(len(y.data), len(continuum(lamb.value)))
        # res = y.data.value-continuum(lamb.value)
        # res2 = spec1d.data-continuum(lamb.value)
        # plt.figure()
        # plt.plot(lamb.value, spec1d.data)
        # plt.plot(lamb.value, y.mask*np.max(res))
        # plt.plot(lamb.value, continuum(lamb.value))
        # plt.show()

        # from IPython import embed
        # embed()
        

    return continuum,rms

def get_skysubtract(meansky,i,throughputcorr_array,wavcal_array, normalized=False):
    '''
    Input parameters:
    - normalized    :   Whether the sky and spectra were normalized. If they were, then the meansky should rather
                        be divided out.'''
    # import numpy as np
    # import scipy
    # import astropy.units as u
    # from specutils.spectra import Spectrum1D
    # from astropy.modeling import models,fitting
    # import matplotlib.pyplot as plt
    # from astropy.nddata import StdDevUncertainty
    # from copy import deepcopy
#        sky=Spectrum1D(spectral_axis=np.arange(len(sky0_flux))*u.AA,flux=sky0_flux*u.electron,uncertainty=StdDevUncertainty(sky0_err),mask=sky0_mask),True))

    spec1d0=Spectrum1D(spectral_axis=throughputcorr_array[i].spec1d_pixel*u.AA,flux=throughputcorr_array[i].spec1d_flux,uncertainty=throughputcorr_array[i].spec1d_uncertainty,mask=throughputcorr_array[i].spec1d_mask)
    sky0=Spectrum1D(spectral_axis=throughputcorr_array[i].spec1d_pixel*u.AA,flux=np.zeros(len(throughputcorr_array[i].spec1d_mask))*u.electron,uncertainty=np.full(len(throughputcorr_array[i].spec1d_mask),np.inf),mask=np.full(len(throughputcorr_array[i].spec1d_mask),True))
    wav0=meansky.spectral_axis.value
    wav=wavcal_array[i].wav
    skysubtract0=deepcopy(spec1d0)
    skysubtract0.mask=np.full(len(skysubtract0.flux),True,dtype=bool)

    if len(wav)>0:
        # from IPython import embed
        # embed()
        sky_flux=fftinterp(wav,wav0,meansky.flux)
        sky_err=fftinterp(wav,wav0,meansky.uncertainty.quantity.value)
        sky_mask=fftinterp(wav,wav0,meansky.mask)
        sky0=Spectrum1D(spectral_axis=throughputcorr_array[i].spec1d_pixel*u.AA,flux=sky_flux,uncertainty=StdDevUncertainty(sky_err),mask=sky_mask)
        
        continuum0,rms0=get_continuum(sky0,-5,1,10,10)
        wtest = continuum0(np.arange(len(sky0.data)))
        test = deepcopy(sky0)
        testsub = test.subtract(sky0*0+100)
        if normalized:
            skysubtract0=spec1d0.divide(sky0)
        else:
            skysubtract0=spec1d0.subtract(sky0)

        # from IPython import embed
        # embed()
        # plt.figure()
        # plt.plot(spec1d0.flux)
        # plt.plot(skysubtract0.flux)
        # plt.plot(sky0.flux)
        # plt.show()

    sky=m2fs.extract1d(throughputcorr_array[i].aperture,spec1d_pixel=throughputcorr_array[i].spec1d_pixel,spec1d_flux=sky0.flux,spec1d_uncertainty=sky0.uncertainty,spec1d_mask=sky0.mask)
    skysubtract=m2fs.extract1d(throughputcorr_array[i].aperture,spec1d_pixel=throughputcorr_array[i].spec1d_pixel,spec1d_flux=skysubtract0.flux,spec1d_uncertainty=skysubtract0.uncertainty,spec1d_mask=skysubtract0.mask)
    return sky,skysubtract

from numba import jit
@jit(nopython=True, cache=True)
def interpolate_numba(x, x1, y1):
    """return array interpolated along time-axis to fill missing values"""
    result = np.zeros_like(x, dtype=np.int16)
    ## 
    stop = False
    i=0
    while x[i]<x1[0]:
        result[i] = y1[i]
        i+=1
    for i in range(len(x)):
        if x[i]>x1[-1]:
            result[i] = y1[-1]
        else:
            for j in range(len(x1)):
                if x[i]<x1[j]:
                    a = (y1[j]-y1[j-1])/(x1[j]-x1[j-1])
                    b = y1[j-1] - a*x1[j-1]
                    result[i] = a*x[i] + b
                    break
    return result
#
@jit(nopython=True, cache=True)
def get_interp_numba(q, x0, y0, pixel0_template, pixelscale_template, xref):
        xscale=(x0-pixel0_template)/pixelscale_template
        # x1=q[1]*pixelscale_template+x0*(1.+np.polynomial.polynomial.polyval(xscale,q[2:]))
        x1=q[1]*pixelscale_template+x0*(1.+mynumbapoly(xscale, q[2:]))
        interp=q[0]*np.interp(xref,x1,y0)
        # interp = y0
        # print(xref[0], xref[-1])
        # print(x1[0], x1[-1])
        # interp=q[0]*interpolate_numba(xref,x1,y0)
        return interp
#
@jit(nopython=True, cache=True)
def my_func_numba(q, x0, y0, pixel0_template, pixelscale_template, xref, fluxref):
    '''Function computing the residuals between the new spectrum and the template'''
    interp=get_interp_numba(q, x0, y0, pixel0_template, pixelscale_template, xref)
    diff2=(fluxref-interp)**2
    return np.sum(diff2)
#
@jit(nopython=True, cache=True)
def loglike_numba(q, x0, y0, pixel0_template, pixelscale_template, xref, fluxref):
    '''Log likelihood for fit'''
    return -0.5*my_func_numba(q, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
#
@jit(nopython=True, cache=True)
def guess_params_numba_descent(q0, q1, x0, y0, pixel0_template, pixelscale_template, xref, fluxref):
    # q0 = np.arange(0., 5., 0.01)
    # q1 = np.arange(-0.1, 0.1, 0.001)
    i=0
    refval = -np.inf
    for _q0 in q0:
        for _q1 in q1:
            val = loglike_numba(np.array([_q0, _q1, 0.]), x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
            if val > refval:
                refval = val
                q0out = _q0
                q1out = _q1
            i+=1
    return q0out, q1out
#
@jit(nopython=True, cache=True)
def guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref):
    # q0 = np.arange(0., 5., 0.01)
    # q1 = np.arange(-0.1, 0.1, 0.001)
    i=0
    refval = -np.inf
    for _q0 in q0:
        for _q1 in q1:
            for _q2 in q2:
                val = loglike_numba(np.array([_q0, _q1, _q2]), x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
                if val > refval:
                    refval = val
                    q0out = _q0
                    q1out = _q1
                    q2out = _q2
                i+=1
    return q0out, q1out, q2out
#
# @jit(nopython=True)
def guess_params_numba(x0, y0, pixel0_template, pixelscale_template, xref, fluxref):
    ## Initial grid
    q0 = np.arange(0., 5., .5)
    q1 = np.arange(-0.1, 0.1, 0.05)
    _q0, _q1 = guess_params_numba_descent(q0, q1, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    ## new grid placing this in the center
    q0 = np.arange(_q0-1.5, _q0+2, 0.025)
    q1 = np.arange(_q1-0.15, _q1+0.2, 0.0025)
    _q0, _q1 = guess_params_numba_descent(q0, q1, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    ## new grid placing this in the center
    q0 = np.arange(_q0-0.05, _q0+0.1, 0.0125)
    q1 = np.arange(_q1-0.005, _q1+0.005, 0.00125)
    _q0, _q1 = guess_params_numba_descent(q0, q1, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    ## new grid placing this in the center
    q0 = np.arange(_q0-0.025, _q0+0.1, 0.0062)
    q1 = np.arange(_q1-0.0025, _q1+0.005, 0.00062)
    _q0, _q1 = guess_params_numba_descent(q0, q1, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    return _q0, _q1

def guess_params_numba_3d(x0, y0, pixel0_template, pixelscale_template, xref, fluxref):
    ## Initial grid
    q0 = np.arange(0., 3., .2)
    q0 = np.array([np.nanmedian(fluxref)/np.nanmedian(y0)])
    q1 = np.arange(-0.2, 0.2, 0.025)
    q2 = np.arange(-0.3, 0.325, 0.005)
    q2 = np.array([0.])
    _q0, _q1, _q2 = guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    print(_q0, _q1, _q2)
    ## new grid placing this in the center
    q0 = np.arange(0.5, 1.5, 0.05)
    q1 = np.arange(_q1-0.35, _q1+0.4, 0.005)
    q2 = np.arange(_q2-0.01, _q2+0.011, 0.001)
    _q0, _q1, _q2 = guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    print(_q0, _q1, _q2)
    # ## new grid placing this in the center
    # q0 = np.arange(_q0-1.5, _q0+2, 0.025)
    # q1 = np.arange(_q1-0.1, _q1+0.11, 0.001)
    # q2 = np.arange(_q2-0.02, _q2+0.04, 0.01)
    # _q0, _q1, _q2 = guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    # print(_q0, _q1, _q2)
    # ## new grid placing this in the center
    # q0 = np.arange(_q0-1.5, _q0+2, 0.025)
    # q1 = np.arange(_q1-0.1, _q1+0.11, 0.001)
    # q2 = np.arange(_q2-0.04, _q2+0.06, 0.01)
    # _q0, _q1, _q2 = guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    # print(_q0, _q1, _q2)
    # ## new grid placing this in the center
    # q0 = np.arange(_q0-0.05, _q0+0.1, 0.0125)
    # q1 = np.arange(_q1-0.005, _q1+0.005, 0.00125)
    # q2 = np.arange(_q2-0.07, _q2+0.08, 0.01)
    # _q0, _q1, _q2 = guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    # ## new grid placing this in the center
    # q0 = np.arange(_q0-0.025, _q0+0.1, 0.0062)
    # q1 = np.arange(_q1-0.0025, _q1+0.005, 0.00062)
    # q2 = np.arange(_q2-0.07, _q2+0.08, 0.01)
    # _q0, _q1, _q2 = guess_params_numba_descent_3d(q0, q1, q2, x0, y0, pixel0_template, pixelscale_template, xref, fluxref)
    return _q0, _q1, _q2

@jit(nopython=True)
def mynumbapoly(x, c):
    nbcoeffs = len(c)
    y = np.zeros(x.shape)
    for i in range(nbcoeffs):
        y+=c[i]*(x**i)
    return y

def get_id_lines_translate(extract1d_template,id_lines_template,extract1d,linelist,continuum_rejection_low,
                           continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order,
                           threshold_factor,window,id_lines_order,id_lines_rejection_iterations,
                           id_lines_rejection_sigma,id_lines_tol_angs,id_lines_tol_pix,resolution_order,
                           resolution_rejection_iterations,add_lines_iterations, idlinestemplate_data):
    import numpy as np
    import scipy
    import astropy.units as u
    from specutils.fitting import find_lines_threshold
    from specutils.fitting import find_lines_derivative
    from specutils.spectra import Spectrum1D
    import matplotlib.pyplot as plt
    from copy import deepcopy
    from astropy.modeling import models,fitting
    from dynesty import DynamicNestedSampler,NestedSampler
    import time
    '''This function is supposed to build the wavelength solution in the other apertures based on the identification performed in the first one.
    Let's have a look at how is all works'''

    ## Initialize lists
    fit_lines=[0] #? 
    wav=[0.] #? 
    func=[0] #?
    rms=[-999.]
    npoints=[0]
    resolution=0.
    resolution_rms=-999.
    resolution_npoints=0

    print(len(np.where(extract1d.spec1d_mask==False)[0]),'asfdasdfasdfasdf')    ##? 
    if len(np.where(extract1d.spec1d_mask==False)[0])>100: ## If we have more than 100 points that are NOT masked (i.e. more than 100 valid bins)
        print('mark 1 {}'.format(time.time()))
        ## Fit each individual lines in the spectrum
        continuum0,spec_contsub,fit_lines=get_fitlines(extract1d,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,threshold_factor,window,continuum_rejection_order)
        ## Find the first and last non-masked bins, compute the coverage in bins
        pixelmin=np.min(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])
        pixelmax=np.max(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])
        # pixelscale=0.5*(pixelmax-pixelmin) ## Unused
        # pixel0=pixelmin+pixelscale ## Unused
        ## Build the use array with indices to use for the calibration
        ## i.e. bins insinde the unmasked region, and with a flux value lower than 1e6
        use=np.where((extract1d.spec1d_pixel>=pixelmin)&(extract1d.spec1d_pixel<=pixelmax)&(extract1d.spec1d_mask==False)&(extract1d.spec1d_flux.value<1.e+6))[0]
#        use=np.where((extract1d.spec1d_pixel>=pixelmin)&(extract1d.spec1d_pixel<=pixelmax)&(extract1d.spec1d_mask==False))[0] ## Clearly the flux thershold was added as some point

        ## Find the lines in the initial template - wait that seems increadibly uneficient, 
        ##                                          have we not done that already before?
        # continuum0_template,spec_contsub_template,fit_lines_template=get_fitlines(extract1d_template,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,threshold_factor,window,continuum_rejection_order)
        _data = pickle.load(open(idlinestemplate_data, 'rb'))
        continuum0_template,spec_contsub_template_xaxis_value, spec_contsub_template_flux_value,fit_lines_template = _data
        ## Find the first and last non-masked bins, compute the coverage in bins 
        pixelmin_template=np.min(extract1d_template.spec1d_pixel[extract1d_template.spec1d_mask==False])
        pixelmax_template=np.max(extract1d_template.spec1d_pixel[extract1d_template.spec1d_mask==False])
        pixelscale_template=0.5*(pixelmax_template-pixelmin_template)
        pixel0_template=pixelmin_template+pixelscale_template
        use_template=np.where((extract1d_template.spec1d_pixel>=pixelmin_template)&(extract1d_template.spec1d_pixel<=pixelmax_template)&(extract1d.spec1d_mask==False))[0]

        print('mark 2 {}'.format(time.time()))
        itime = time.time()

        if len(fit_lines.fit)>0:

#        means=np.array([fit_lines.fit[q].mean.value for q in range(0,len(fit_lines.fit))])
#        lowers=np.array([fit_lines.subregion[q].lower.value for q in range(0,len(fit_lines.fit))])
#        uppers=np.array([fit_lines.subregion[q].upper.value for q in range(0,len(fit_lines.fit))])
#        use2=np.where((means>=pixelmin)&(means<pixelmax)&(means>=lowers-5.)&(means<=uppers+5.))[0]
#        amplitudes=[fit_lines.fit[q].amplitude.value for q in use2]
#        best=amplitudes.index(max(amplitudes))
#
#        means_template=np.array([fit_lines_template.fit[q].mean.value for q in range(0,len(fit_lines_template.fit))])
#        lowers_template=np.array([fit_lines_template.subregion[q].lower.value for q in range(0,len(fit_lines_template.fit))])
#        uppers_template=np.array([fit_lines_template.subregion[q].upper.value for q in range(0,len(fit_lines_template.fit))])
#        use2_template=np.where((means_template>=pixelmin_template)&(means_template<pixelmax_template)&(means_template>=lowers_template-5.)&(means_template<=uppers_template+5.))[0]
#        amplitudes_template=[fit_lines_template.fit[q].amplitude.value for q in use2_template]
#        best_template=amplitudes_template.index(max(amplitudes_template))

#        q1=(means[use2[best]]-means_template[use2_template[best_template]])/pixelscale_template
#        q0=amplitudes[best]/amplitudes_template[best_template]
#        print(means[use2[best]],means_template[use2_template[best_template]])
#        print(amplitudes[best],amplitudes_template[best_template])
#        print(q0,q1)

#        print(np.median(spec_contsub0.flux.value[use]))
#        spec_contsub=spec_contsub0/(np.median(spec_contsub0.flux.value[use]))
#        spec_contsub_template=spec_contsub_template0/(np.median(spec_contsub_template0.flux.value[use_template]))
            '''first run dynesty to fit for zero order offset between current lamp spectrum and template (gradient-descent gets stuck in local minima)'''
            ndim=3
            def get_interp(q):#spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template):
                '''Function returning the interpolated spectrum obtained after adjusting the wavelength
                with the polynomial parameters stored in q.
                TODO: improve with a cubic interpolator?
                Input parameters:
                - q     : q[0] scales the intensity of the spectrum
                          q[1] stretches the spectrum (adjusts the scale)
                          q[2:] contains polynomial coefficients to adjust the spectrum
                '''
                #    spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template=args
                # x0=np.array(spec_contsub_template.spectral_axis.value[use_template],dtype='float')
                # y0=spec_contsub_template.flux.value[use_template]
                x0=np.array(spec_contsub_template_xaxis_value[use_template],dtype='float')
                y0=spec_contsub_template_flux_value[use_template]
                
                xscale=(x0-pixel0_template)/pixelscale_template
                x1=q[1]*pixelscale_template+x0*(1.+np.polynomial.polynomial.polyval(xscale,q[2:]))
                # x1=q[1]*pixelscale_template+x0*(1.+np.polynomial.chebyshev.chebval(xscale,q[2:]))
                interp=q[0]*fftinterp(spec_contsub.spectral_axis.value[use],x1,y0)
                return interp

            def my_func(q):#spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template):
                '''Function computing the residuals between the new spectrum and the template'''
                interp=get_interp(q)#,spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template)
                # diff2=((spec_contsub.flux.value[use]-interp)/spec_contsub.uncertainty.quantity.value[use])**2
                diff2=(spec_contsub.flux.value[use]-interp)**2
                return np.sum(diff2)

            def loglike(q):
                '''Log likelihood for fit'''
                return -0.5*my_func(q)#,spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template)

            def ptform(u):
                '''Priors for the fit.'''
                prior=[]
                prior.append([0.,5.]) ## Flux scale Why is it only positive?
                # prior.append([-2.0,2.0]) ## Radial velocity ! Now 2 seems waaaay too much. 
                prior.append([-0.1,0.1]) ## Radial velocity ! Now 2 seems waaaay too much. 
                prior.append([-0.0,0.0])
                # prior.append([-0.0,0.0])
                # prior.append([-0.0,0.0])
                # prior.append([-0.0,0.0])
                # prior.append([-0.0,0.0])
                # prior.append([-0.0,0.0])
                prior=np.array(prior)
                x=np.array(u)
                for i in range(0,len(x)):
                    x[i]=prior[i][0]+(prior[i][1]-prior[i][0])*u[i]
                return x

            # etime = time.time()
            # print('mark 3 {:0.2f}'.format(etime-itime))
            # itime = time.time()

            ## PIC The DynamicNestedSampler is apparently used to obtain some sort of a 
            ## first guess on the first two parameters, i.e. on the INTENSITY of the spectrum 
            ## and the overal stretch.
            ## Why don't we get a value of the main RV as well? How can that work at all if we
            ## do not account for the RV shift?
            # dsampler=DynamicNestedSampler(loglike,ptform,ndim,bound='multi')
            #        sampler=NestedSampler(loglike,ptform,ndim,bound='multi')
            # dsampler.run_nested(maxcall=12000, print_progress=False)
            #        sampler.run_nested(dlogz=0.05)

            ## Where is the dsampler maximal? 
            # best=np.where(dsampler.results.logl==np.max(dsampler.results.logl))[0][0]

            ## PIC: I am very surprised to see that this Dynamic sampler does not even find the 
            ## best solution. By doing a simple grid search I actually get values that are closer
            ## to the maximum likelihood. Perhaps is it poorly parametrized here. Because it's slow,
            ## I replace it with a simple mini-grid search to help the optimization process later on.
            ## Plus with this grid, I am sure of what I am feeding scipy optimize.
            ## This solution is about 15 times faster as the previous implementation.
            #
            # x0, y0, pixel0_template, pixelscale_template, xref, fluxref
            _x0 = np.array(spec_contsub_template_xaxis_value[use_template],dtype=float)
            _y0 = np.array(spec_contsub_template_flux_value[use_template], dtype=float)
            _xref = np.array(spec_contsub.spectral_axis.value[use], dtype=float)
            _fluxref = np.array(spec_contsub.flux.value[use], dtype=float)
            q1, q2, q3 = guess_params_numba_3d(_x0, _y0, pixel0_template, pixelscale_template, _xref, _fluxref)
            # get_interp_numba(params, _x0, _y0, pixel0_template, pixelscale_template, _xref)
            # pos = np.where(vals==np.max(vals))[0][0]
            # _params = combis[pos]
            _params = np.array([q1, q2])
            # q0 = np.arange(0., 5., 0.1)
            # q1 = np.arange(-0.1, 0.1, 0.01)
            # # q2 = np.array([0])
            # vals = []; combis = []
            # for _q0 in q0:
            #     for _q1 in q1:
            #         val = loglike(np.array([_q0, _q1, 0.]))
            #         vals.append(val)
            #         combis.append([_q0, _q1])
            # pos = np.where(vals==np.max(vals))[0][0]
            # _params = combis[pos]
#        best=np.where(sampler.results.logl==np.max(sampler.results.logl))[0][0]

            '''now run gradient descent to find higher-order corrections to get best match'''    
            # params=np.append(dsampler.results.samples[best],np.array([0.]))#,0.,0.,0.,0.]))
            # params = np.append(_params, np.array([0.1]))
            params = np.append(_params, np.array([q3]))
            # ### ---------------
            # # PIC: Now we debug: let's look at the spectra and how the previous method estimates the
            # # stretch and flux.
            # myparams = np.copy(params)
            # interp=get_interp(params)#,spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template)
            # plt.figure()
            # plt.plot(spec_contsub_template_flux_value[use_template])
            # plt.plot(spec_contsub.flux.value[use])
            # plt.plot(interp, '--')
            # plt.show()
            # from IPython import embed
            # embed()
            # ### ---------------

            etime = time.time()
            print('mark 3.5 {:0.2f}'.format(etime-itime))
            itime = time.time()

            shiftstretch=scipy.optimize.minimize(my_func,params,method='Powell')
            # enablePrint()
            # interp=get_interp(shiftstretch.x) ## Unused
            print('log likelihood/1e9 = ',loglike(shiftstretch.x)/1.e9)

            # # ### ---------------
            # # PIC: Now we debug: let's look at the spectra and how the previous method estimates the
            # # stretch and flux.
            # interp=get_interp(shiftstretch.x)#,spec_contsub,spec_contsub_template,use,use_template,pixel0_template,pixelscale_template)
            # plt.figure()
            # plt.plot(spec_contsub_template_flux_value[use_template])
            # plt.plot(spec_contsub.flux.value[use])
            # plt.plot(interp, '--')
            # plt.show()
            # from IPython import embed
            # embed()
            # # ### ---------------

            # found=0 ## Unused
            id_lines_pix=[]
            id_lines_wav=[]
            # id_lines_species=[] ## Unused
            id_lines_used=[]
            order=[id_lines_template.func.degree]
            rejection_iterations=[id_lines_rejection_iterations]
            rejection_sigma=[id_lines_rejection_sigma]

            x0=np.array([id_lines_template.fit_lines.fit[q].mean.value for q in range(0,len(id_lines_template.fit_lines.fit))])

            xscale=(x0-pixel0_template)/pixelscale_template
            x1=float(shiftstretch.x[1])*pixelscale_template+x0*(1.+np.polynomial.polynomial.polyval(xscale,shiftstretch.x[2:]))
#        x1=float(shiftstretch.x[1])*pixelscale_template+x0*(1.+np.polynomial.chebyshev.chebval(xscale,shiftstretch.x[2:]))
            for i in range(0,len(x0)):
                if id_lines_template.wav.mask[i]==False:
                    dist=np.sqrt((x1[i]-np.array([fit_lines.fit[q].mean.value for q in range(0,len(fit_lines.fit))]))**2)
                    nan=np.where(dist!=dist)[0]
                    if len(nan)>0:
                        dist[nan]=np.inf#artificially replace any nans with huge values
                    best=np.where(dist==np.min(dist))[0][0]
#                    print(best,dist[best],id_lines_tol_pix)
                    if dist[best]<id_lines_tol_pix:
                        id_lines_pix.append(fit_lines.fit[best].mean.value)
                        id_lines_wav.append(id_lines_template.wav[i])
                        id_lines_used.append(best)
            
            # print('mark 4 {}'.format(time.time()))

            if len(id_lines_pix)>0:
                func=[models.Legendre1D(degree=1)]
                rms=[]
                npoints=[]
                func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
                for j in range(0,add_lines_iterations):
                    new_pix,new_wav,new_used=m2fs.line_id_add_lines(linelist,[fit_lines.fit[q].mean.value for q in range(0,len(fit_lines.fit))],id_lines_used,func0,id_lines_tol_angs)
                    for i in range(0,len(new_pix)):
                        id_lines_pix.append(new_pix[i])
                        id_lines_wav.append(new_wav[i])
                        id_lines_used.append(new_used[i])
                    func0,rms0,npoints0,y=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
                func.append(func0)
                rms.append(rms0)
                npoints.append(npoints0)

                wav=np.ma.masked_array(np.full((len(fit_lines.fit)),-999,dtype='float'),mask=np.full((len(fit_lines.fit)),True,dtype=bool))
                for j in range(0,len(id_lines_pix)):
                    wav[id_lines_used[j]]=id_lines_wav[j]
                    wav.mask[id_lines_used[j]]==False

                resolution,resolution_rms,resolution_npoints=m2fs.get_resolution(deepcopy(fit_lines),deepcopy(wav),resolution_order,resolution_rejection_iterations)
                print('mark 5 {}'.format(time.time()))
        # if extract1d.aperture == 44:
        #     from IPython import embed
        #     embed()
        #     plt.figure()
        #     plt.plot(func[1](np.arange(2000)))
        #     plt.show()
    return m2fs.id_lines(aperture=extract1d.aperture,fit_lines=fit_lines,wav=wav,func=func[len(func)-1],rms=rms[len(rms)-1],npoints=npoints[len(npoints)-1],resolution=resolution,resolution_rms=resolution_rms,resolution_npoints=resolution_npoints)
            
def get_fitlines(extract1d,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,threshold_factor,window,continuum_rejection_order):
    import numpy as np
    from specutils.spectra import Spectrum1D
    import astropy.units as u
    from specutils.fitting import find_lines_threshold
    from specutils.fitting import find_lines_derivative
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import time
    it = time.time(); print('submark 1 {}'.format(it))
    ## Initialize the object
    spec1d0=Spectrum1D(spectral_axis=deepcopy(extract1d.spec1d_pixel)*u.AA,flux=deepcopy(extract1d.spec1d_flux),uncertainty=deepcopy(extract1d.spec1d_uncertainty),mask=deepcopy(extract1d.spec1d_mask))
#    spec1d0.mask[spec1d0.flux!=spec1d0.flux]=True
#    spec1d0.flux[spec1d0.flux!=spec1d0.flux]=0.
    ## Get the continuum
    et = time.time(); print('submark 2 {}'.format(et - it))
    continuum0,rms0=get_continuum(spec1d0,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order)
    et = time.time(); print('submark 3 {}'.format(et - it))
    pixel0=extract1d.spec1d_pixel*u.AA
    spec_contsub=spec1d0-continuum0(pixel0.value)
    spec_contsub.uncertainty.quantity.value[:]=rms0
    spec_contsub.mask=spec1d0.mask
    spec_contsub.flux[spec_contsub.flux!=spec_contsub.flux]=0.
#    spec_contsub.flux[spec_contsub.flux.value<-100]=0.*u.electron
    id_lines_initial0=find_lines_derivative(spec_contsub,flux_threshold=threshold_factor*rms0)#find peaks in continuum-subtracted "spectrum"
    et = time.time(); print('submark 4 {}'.format(et - it))
    fit_lines=get_aperture_profile(id_lines_initial0,spec1d0,continuum0,window)
    et = time.time(); print('submark 5 {}'.format(et - it))
    return continuum0,spec_contsub,fit_lines

def get_aperture_profile(apertures_initial,spec1d,continuum,window):

    subregion=[]
    g_fit=[]
    realvirtual=[]#1 for real aperture, 0 for virtual aperture (ie a placeholder aperture for bad/unplugged fiber etc.)
    initial=[]#1 for apertures identified automatically, 0 for those identified by hand

    for j in range(0,len(apertures_initial)):
        x_center=apertures_initial['line_center'][j].value
        subregion0,g_fit0=fit_aperture(spec1d-continuum(spec1d.spectral_axis.value),window,x_center)
        subregion.append(subregion0)
        g_fit.append(g_fit0)
        realvirtual.append(True)
        initial.append(True)
    return m2fs.aperture_profile(g_fit,subregion,realvirtual,initial)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from specutils.manipulation import extract_region
from specutils import SpectralRegion
from specutils.manipulation import extract_region
import astropy.units as u
from specutils.spectra import Spectrum1D
from specutils.fitting import estimate_line_parameters
from astropy.modeling import models,fitting
from specutils.fitting import fit_lines

def fit_aperture(spec,window,x_center):
    '''Function fitting a gaussian to spec searching for 
    the shape in the window centered on x_center.'''
    val1=x_center-window/2.#window for fitting aperture
    val2=x_center+window/2.#window for fitting aperture
    subregion0=SpectralRegion(val1*u.AA,val2*u.AA)#define extraction region from window
    # sub_spectrum0=extract_region(spec,subregion0)#extract from window

    ## PIC: the extract_region function appears to be extremely slow...
    ## I do not understand the point of this function, it appears to very
    ## simply cut the spectrum. Let's try a simpler approach
    myxaxis = spec.spectral_axis.value
    idx = np.where((myxaxis>subregion0.bounds[0].value) & (myxaxis<subregion0.bounds[1].value))[0]
    sub_spectrum0 = Spectrum1D(flux=spec.flux[idx], spectral_axis=spec.spectral_axis[idx], 
                               uncertainty=spec.uncertainty[idx], mask=spec.mask[idx])
    
    ## PIC: Apparently we must have an object without the mask for that to work. 
    ## Probably because False is generally use to mark invalid bins and not the other way around.
    sub_spectrum=Spectrum1D(flux=sub_spectrum0.flux,spectral_axis=sub_spectrum0.spectral_axis,uncertainty=sub_spectrum0.uncertainty)
#    print(sub_spectrum)
#    print(sub_spectrum)
    rough=estimate_line_parameters(sub_spectrum,models.Gaussian1D())#get rough estimate of gaussian parameters for aperture
#    np.random.seed(0)
#    x=np.linspace(0.,10.,2056)
#    y=3*np.exp(-0.5*(x-6.3)**2/0.8**2)
#    y+=np.random.normal(0.,0.2,x.shape)
#    spectrum=Spectrum1D(flux=y*u.Jy,spectral_axis=x*u.um)
#    g_init=models.Gaussian1D(amplitude=3.*u.Jy,mean=6.1*u.um,stddev=1.*u.um)
#    g_fit=fit_lines(spectrum,g_init)
#    y_fit=g_fit(x*u.um)
#    print(spectrum)

    g_init=models.Gaussian1D(amplitude=rough.amplitude.value*u.electron,mean=rough.mean.value*u.AA,stddev=rough.stddev.value*u.AA)#now do a fit using rough estimate as first guess
    try:
        g_fit0=fit_lines(sub_spectrum,g_init)#now do a fit using rough estimate as first guess
    except:
        g_fit0=g_init ## PIC: for now, if the fit is impossible I don't let the program crash.
        # from IPython import embed
        # embed()
    # g_fit2=myfit_lines(sub_spectrum,g_init)#now do a fit using rough estimate as first guess
    
    ## PIC: The following was just to make a plot and test a different fitting function
    #
    # ## Now what I do is to check whether the two fitting processes gave the same results
    # ampdiff = g_fit2.amplitude.value - g_fit0.amplitude.value
    # meandiff = g_fit2.mean.value - g_fit0.mean.value
    # stddiff = g_fit2.stddev.value - g_fit0.stddev.value
    # plot=False
    # if abs(ampdiff) > 0.1*g_fit2.amplitude.value:
    #     print("NOT THE SAME AMPLITUDE") 
    #     print(g_fit0.amplitude.value, g_fit2.amplitude.value) 
    #     plot=True
    # if abs(meandiff) > 0.1*g_fit2.mean.value:
    #     print("NOT THE SAME MEAN") 
    #     print(g_fit0.mean.value, g_fit2.mean.value) 
    #     plot=True
    # if abs(stddiff) > 0.1*g_fit2.stddev.value:
    #     print("NOT THE SAME STDDEV") 
    #     print(g_fit0.stddev.value, g_fit2.stddev.value) 
    #     plot=True
    #
    # if plot:
    #     plt.figure()
    #     plt.plot(sub_spectrum.spectral_axis.value, sub_spectrum.flux.value, '.', color='black')
    #     plt.plot(sub_spectrum.spectral_axis.value, g_fit2(sub_spectrum.spectral_axis), color='green', label='curve fit')
    #     plt.plot(sub_spectrum.spectral_axis.value, g_fit0(sub_spectrum.spectral_axis), color='red', label='specutils fitline')
    #     plt.legend()
    #     plt.show()
    
    return subregion0,g_fit0

def gaussian(x, amplitude, mean, std):
    return amplitude * np.exp(-(x - mean)**2 / (2 * std**2))

from scipy.optimize import curve_fit
def myfit_lines(sub_spectrum,g_init):
    '''Alternative function to fit a Gaussian profile to data.
    Original added here because I though the fitting was slow,
    It turns out that specutils performs great for the fitting, but
    the exatract_region function is super slow.'''
    #
    ## initial guess
    p = [g_init.amplitude.value, g_init.mean.value, g_init.stddev.value]
    xaxis = sub_spectrum.spectral_axis.value
    yaxis = sub_spectrum.flux.value
    sigma = sub_spectrum.uncertainty.array
    # from IPython import embed
    # embed()
    popt, pcov = curve_fit(gaussian, xaxis, yaxis, p0=p, sigma=sigma)
    amplitude_fit, mean_fit, std_fit = popt
    #
    g_curvefit=models.Gaussian1D(amplitude=amplitude_fit*u.electron,mean=mean_fit*u.AA,stddev=std_fit*u.AA)
    return g_curvefit

def get_columnspec(data,trace_step,n_lines,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order,threshold_factor,window):
    import numpy as np
    import astropy.units as u
    from specutils.fitting import find_lines_threshold
    from specutils.fitting import find_lines_derivative

    n_cols=np.shape(data)[1]
    trace_n=int(n_cols/trace_step)
#    print(n_cols,trace_n)
    trace_cols=np.linspace(0,n_cols,trace_n,dtype='int')

#    apertures_initial=[]
#    col=[]
#    spec1d=[]
#    continuum=[]
#    rms=[]
#    pixel=[]
#    apertures_profile=[]
    columnspec_array=[]
    for i in range(0,len(trace_cols)-1):
        # print('working on '+str(i+1)+' of '+str(len(trace_cols))+' trace columns')
        col0=np.arange(n_lines)+trace_cols[i]
        spec1d0=m2fs.column_stack(data,col0)
#        np.pause()
        continuum0,rms0=get_continuum(spec1d0,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order, debug=False)
        pixel0=(np.arange(len(spec1d0.data),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
        spec_contsub=spec1d0-continuum0(pixel0.value)
        spec_contsub.uncertainty.quantity.value[:]=rms0
#        spec_contsub.mask[:]=False
        # try:
        apertures_initial0=find_lines_derivative(spec_contsub,flux_threshold=threshold_factor*rms0)#find peaks in continuum-subtracted "spectrum"
        # except:
        #     from IPython import embed
        #     embed()
        # xvals = my_find_lines_derivative(spec_contsub.data,flux_threshold=threshold_factor*rms0)
        # from astropy.table.table import QTable
        # qtable = QTable()
        # qtable['line_center'] = np.array(xvals, dtype=float)*u.AA
        # qtable['line_type'] = ['emission' for i in range(len(xvals))]
        # qtable['line_center_index'] = xvals
        # apertures_initial0 = qtable
        # from IPython import embed
        # embed()
#        apertures_profile0=get_aperture_profile(spec1d0,window)

        apertures_profile0=get_aperture_profile(apertures_initial0,spec1d0,continuum0,window)
        columnspec0=m2fs.columnspec(columns=col0,spec=spec1d0.data,mask=spec1d0.mask,err=spec1d0.uncertainty,pixel=pixel0,continuum=continuum0,rms=rms0,apertures_initial=apertures_initial0,apertures_profile=apertures_profile0)#have to break up spec1d into individual data/mask/uncertainty fields in order to allow pickling of columnspec_array
#        columnspec0=columnspec(columns=col0,spec=spec1d0,mask=spec1d0.mask,err=spec1d0.uncertainty,pixel=pixel0,continuum=continuum0,rms=rms0,apertures_initial=apertures_initial0,apertures_profile=apertures_profile0)#have to break up spec1d into individual data/mask/uncertainty fields in order to allow pickling of columnspec_array
#        columnspec0.apertures_profile=apertures_profile0
        print('found '+str(len(apertures_initial0))+' apertures in column '+str(col0))
#        col.append(col0)
#        spec1d.append(spec1d0)
#        continuum.append(continuum0)
#        rms.append(rms0)
#        pixel.append(pixel0)
#        apertures_initial.append(apertures_initial0[apertures_initial0['line_type']=='emission'])#keep only emission lines
        columnspec_array.append(columnspec0)
        print('Working for column {} / {}'.format(i, len(trace_cols)))
#    return columnspec(columns=col,spec1d=spec1d,pixel=pixel,continuum=continuum,rms=rms,apertures_initial=apertures_initial,apertures_profile)
    return columnspec_array

def get_aperture_profile_fast(apertures_initial,spec1d,continuum,window):
    import numpy as np
    import astropy.units as u
    from specutils import SpectralRegion
    from astropy.modeling import models,fitting

    subregion=[]
    g_fit=[]
    realvirtual=[]#1 for real aperture, 0 for virtual aperture (ie a placeholder aperture for bad/unplugged fiber etc.)
    initial=[]#1 for apertures identified automatically, 0 for those identified by hand

    # from IPython import embed
    # embed()

    # a = spec1d-continuum(spec1d.spectral_axis.value)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(a.data)
    # for j in range(len(apertures_initial['line_center'])):
    #     deriv = apertures_initial['line_center'][j].value
    #     plt.axvline(deriv)
    # plt.show()

    for j in range(0,len(apertures_initial)):
        x_center=apertures_initial['line_center'][j].value
        spectrum = spec1d-continuum(spec1d.spectral_axis.value)
        # subregion0,g_fit0=fit_aperture(spec1d-continuum(spec1d.spectral_axis.value),window,x_center)
        val1=x_center-window/2.#window for fitting aperture
        val2=x_center+window/2.#window for fitting aperture
        subregion0=SpectralRegion(val1*u.AA,val2*u.AA)#define extraction region from window
        g_fit0=models.Gaussian1D(amplitude=spectrum.data[int(x_center)]*u.electron,mean=x_center*u.AA,stddev=0.1*x_center*u.AA)#now do a fit using rough estimate as first guess        
        subregion.append(subregion0)
        g_fit.append(g_fit0)
        realvirtual.append(True)
        initial.append(True)
    return m2fs.aperture_profile(g_fit,subregion,realvirtual,initial)


def get_hdul(data,skysubtract_array,sky_array,wavcal_array,plugmap,m2fsrun,field_name,thar,temperature,aperture_array,linelist):
    import numpy as np

    from astropy.io import fits
    from astropy import time, coordinates as coord, units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u
    from copy import deepcopy

    data_array=[]
    var_array=[]
    mask_array=[]
    sky=[]
    wav_array=[]
    mjd_array=[]
    hjd_array=[]
    vheliocorr_array=[]
    snratio_array=[]
    thar_npoints_array=[]
    thar_rms_array=[]
    thar_resolution_array=[]
    thar_wav_min_array=[]
    thar_wav_max_array=[]
    temperature_array=[]
    row_array=[]

    temperature0=[]
    for i in range(0,len(temperature)):
        temperature0.append(float(temperature[i]))

    for i in range(0,len(plugmap)):
        temperature_array.append(temperature0)
        thar_npoints=[]
        thar_rms=[]
        thar_resolution=[]
        thar_wav_min=[]
        thar_wav_max=[]
        for j in range(0,len(thar)):
            this=np.where([thar[j][q].aperture for q in range(0,len(thar[j]))]==plugmap['aperture'][i])[0]
            if len(this)>0:
                thar_npoints.append(thar[j][this[0]].npoints)
                thar_rms.append(thar[j][this[0]].rms)
                thar_resolution0=float(-999.)
                thar_wav_min0=float(-999.)
                thar_wav_max0=float(-999.)
                if len(thar[j][this[0]].wav)>1:
                    keep=np.where(thar[j][this[0]].wav.mask==False)[0]
                    if len(keep)>1:
                        dwav=np.max(thar[j][this[0]].wav[keep])-np.min(thar[j][this[0]].wav[keep])
                        dpix=np.max([thar[j][this[0]].fit_lines.fit[q].mean.value for q in keep])-np.min([thar[j][this[0]].fit_lines.fit[q].mean.value for q in keep])
                        central_wavelength=np.min(linelist.wavelength)+(np.max(linelist.wavelength)-np.min(linelist.wavelength))/2.
#                        x=[thar[j][this[0]].fit_lines.fit[q].mean.value for q in keep]
                        thar_resolution0=thar[j][this[0]].resolution(fftinterp(x=central_wavelength,xp=[thar[j][this[0]].wav[q] for q in keep],fp=[thar[j][this[0]].fit_lines.fit[q].mean.value for q in keep]))*dwav/dpix
                        thar_wav_min0=np.min(np.ma.compressed(thar[j][this[0]].wav))
                        thar_wav_max0=np.max(np.ma.compressed(thar[j][this[0]].wav))

                thar_resolution.append(thar_resolution0)
                thar_wav_min.append(thar_wav_min0)
                thar_wav_max.append(thar_wav_max0)

        thar_npoints_array.append(thar_npoints)
        thar_rms_array.append(thar_rms)
        thar_resolution_array.append(thar_resolution)
        thar_wav_min_array.append(thar_wav_min)
        thar_wav_max_array.append(thar_wav_max)

        if plugmap['objtype'][i]=='TARGET':
            # coords=coord.SkyCoord(plugmap['ra'][i],plugmap['dec'][i],unit=(u.deg,u.deg),frame='icrs')
            # lco=coord.EarthLocation.from_geodetic(lon=-70.6919444*u.degree,lat=-29.0158333*u.degree,height=2380.*u.meter)
            # times=time.Time(wavcal_array[0].mjd,format='mjd',location=lco)
            # light_travel_time_helio=times.light_travel_time(coords,'heliocentric')
            mjd_array.append(wavcal_array[0].mjd)
            # hjd_array.append((Time(wavcal_array[0].mjd,format='mjd').jd+light_travel_time_helio).value)
            hjd_array.append(wavcal_array[0].mjd+2400000.)
            # vheliocorr_array.append((coords.radial_velocity_correction('heliocentric',obstime=times).to(u.km/u.s)).value)
            vheliocorr_array.append(0.)
        else:
            mjd_array.append(wavcal_array[0].mjd)
            hjd_array.append(wavcal_array[0].mjd+2400000.)
            vheliocorr_array.append(0.)
        this=np.where([skysubtract_array[q].aperture for q in range(0,len(skysubtract_array))]==plugmap['aperture'][i])[0][0]
        snratio_array.append(np.median(skysubtract_array[this].spec1d_flux.value[skysubtract_array[this].spec1d_mask==False]/skysubtract_array[this].spec1d_uncertainty.quantity.value[skysubtract_array[this].spec1d_mask==False]))
        this=np.where([aperture_array[q].trace_aperture for q in range(0,len(aperture_array))]==plugmap['aperture'][i])[0][0]
        row=aperture_array[this].trace_func(float(len(data.data[0])/2.))
        row_array.append(row)
    snratio_array=np.array(snratio_array)
    m2fsrun_array=np.full(len(snratio_array),m2fsrun,dtype='a100')
    field_name_array=np.full(len(snratio_array),field_name,dtype='a100')
    thar_npoints_array=np.array(thar_npoints_array,dtype=np.object_)
    thar_rms_array=np.array(thar_rms_array,dtype=np.object_)
    thar_resolution_array=np.array(thar_resolution_array,dtype=np.object_)
    thar_wav_min_array=np.array(thar_wav_min_array,dtype=np.object_)
    thar_wav_max_array=np.array(thar_wav_max_array,dtype=np.object_)
    temperature_array=np.array(temperature_array,dtype=np.object_)
    row_array=np.array(row_array)

    cols=fits.ColDefs([
                    #    fits.Column(name='EXPID',format='A100',array=plugmap['expid']),
                       fits.Column(name='OBJTYPE',format='A6',array=plugmap['objtype']),
                    #    fits.Column(name='RA',format='D',array=plugmap['ra']),
                    #    fits.Column(name='DEC',format='D',array=plugmap['dec']),
                       fits.Column(name='APERTURE',format='I',array=plugmap['aperture']),
                    #    fits.Column(name='RMAG',format='D',array=plugmap['rmag']),
                    #    fits.Column(name='RAPMAG',format='D',array=plugmap['rapmag']),
                    #    fits.Column(name='ICODE',format='D',array=plugmap['icode']),
                    #    fits.Column(name='RCODE',format='D',array=plugmap['rcode']),
                    #    fits.Column(name='BCODE',format='A6',array=plugmap['bcode']),
                    #    fits.Column(name='MAG',format='5D',array=plugmap['mag']),
                    #    fits.Column(name='XFOCAL',format='D',array=plugmap['xfocal']),
                    #    fits.Column(name='YFOCAL',format='D',array=plugmap['yfocal']),
                    #    fits.Column(name='FRAMES',format='B',array=plugmap['frames']),
                    #    fits.Column(name='CHANNEL',format='A100',array=plugmap['channel']),
                    #    fits.Column(name='RESOLUTION',format='A100',array=plugmap['resolution']),
                    #    fits.Column(name='FILTER',format='A100',array=plugmap['filter']),
                    #    fits.Column(name='CHANNEL_CASSETTE_FIBER',format='A100',array=plugmap['channel_cassette_fiber']),
                       fits.Column(name='MJD',format='D',array=mjd_array),
                       fits.Column(name='HJD',format='D',array=hjd_array),
                       fits.Column(name='vheliocorr',format='d',array=vheliocorr_array),
                       fits.Column(name='SNRATIO',format='d',array=snratio_array),
                       fits.Column(name='run_id',format='A100',array=m2fsrun_array),
                       fits.Column(name='field_name',format='A100',array=field_name_array),
                       fits.Column(name='wav_npoints',format='PI()',array=thar_npoints_array),
                       fits.Column(name='wav_rms',format='PD()',array=thar_rms_array),
                       fits.Column(name='wav_resolution',format='PD()',array=thar_resolution_array),
                       fits.Column(name='wav_min',format='PD()',array=thar_wav_min_array),
                       fits.Column(name='wav_max',format='PD()',array=thar_wav_max_array),
                       fits.Column(name='row',format='D',array=row_array),
                       fits.Column(name='temperature',format='PD()',array=temperature_array)])
    table_hdu=fits.FITS_rec.from_columns(cols)

    if len(skysubtract_array)>0:
        for i in range(0,len(plugmap)):
            print(i)
            this=np.where(np.array([skysubtract_array[q].aperture for q in range(0,len(skysubtract_array))])==plugmap['aperture'][i])[0]
            if len(this)>0:
                mask0=deepcopy(skysubtract_array[this[0]].spec1d_mask)
                if len(np.where(skysubtract_array[this[0]].spec1d_mask==False)[0])>0:#mask pixels outside valid extrapolation region of wavelength solution
                    best=np.where(thar_npoints_array[i]==np.max(thar_npoints_array[i]))[0]
                    spec_range=thar_wav_max_array[i][best]-thar_wav_min_array[i][best]
                    best2=np.where(spec_range==np.min(spec_range))[0][0]
                    if thar_npoints_array[i][best[best2]]==0:
                        print(thar_npoints_array[i],this,wavcal_array[this[0]].wav)
                        # from IPython import embed
                        # embed()
                        # np.pause()
                        ## If we reach this point, that means we had a problem with the calibration. 
                        ## It could be an 'unused' aperture? It is not.
                        ## Just mask everything
                        new_mask = np.arange(len(mask0))
#                    if thar_npoints_array[i][best[best2]]>0:###why are there unmasked cases where thar_npoints=0 and/or wavcal.wav=[]
#                        if len(wavcal_array[this[0]].wav)>0:
                    else:
                        extra=(thar_wav_max_array[i][best[best2]]-thar_wav_min_array[i][best[best2]])/float(thar_npoints_array[i][best[best2]])
                        lambdamin=thar_wav_min_array[i][best[best2]]-extra
                        lambdamax=thar_wav_max_array[i][best[best2]]+extra
    #                            print(wavcal_array[this[0]].wav)
                        try:
                            new_mask=np.where((wavcal_array[this[0]].wav<lambdamin)|(wavcal_array[this[0]].wav>lambdamax))[0]
                        except:
                            new_mask = np.arange(len(mask0))
                    mask0[new_mask]=True
                data_array.append(skysubtract_array[this[0]].spec1d_flux.value)
                var_array.append(skysubtract_array[this[0]].spec1d_uncertainty.quantity.value**2)
                mask_array.append(mask0)
                sky.append(sky_array[this[0]].spec1d_flux)
                if len(wavcal_array[this[0]].wav)>0:
                    wav_array.append(wavcal_array[this[0]].wav)
                else:
                    wav_array.append(np.full(len(skysubtract_array[0].spec1d_pixel),0.,dtype=float))
            else:
                data_array.append(np.full(len(skysubtract_array[0].spec1d_pixel),0.,dtype=float))
                var_array.append(np.full(len(skysubtract_array[0].spec1d_pixel),1.e+30,dtype=float))
                mask_array.append(np.full(len(skysubtract_array[0].spec1d_pixel),True,dtype=bool))
                sky.append(np.full(len(skysubtract_array[0].spec1d_pixel),0.,dtype=float))
                wav_array.append(np.full(len(skysubtract_array[0].spec1d_pixel),0.,dtype=float))

        var_array=np.array(var_array)
        mask_array=np.array(mask_array)
        data_array=np.array(data_array)
        sky=np.array(sky)
        wav_array=np.array(wav_array)

        ivar=1./var_array
        mask_or=np.zeros((len(mask_array),len(mask_array[0])),dtype=int)
        for xxx in range(0,len(mask_or)):
            for yyy in range(0,len(mask_or[xxx])):
                if mask_array[xxx][yyy]:
                    mask_or[xxx][yyy]=1
        mask_and=mask_or
                        
        if len(plugmap)!=len(skysubtract_array):
            print('error - plugmap has different length than data array!!!')
            print(len(plugmap),' vs',len(skysubtract_array))
            np.pause()
    else:
        wav_array=np.full(0,0.)
        data_array=np.full(0,0.)
        ivar=np.full(0,0.)
        mask_and=np.full(0,1)
        mask_or=np.full(0,1)
        sky=np.full(0,0.)

    primary_hdu=fits.PrimaryHDU(wav_array,header=data.header)
    new_hdul=fits.HDUList([primary_hdu])
    new_hdul.append(fits.ImageHDU(data_array,name='sky_subtracted'))
    new_hdul.append(fits.ImageHDU(ivar,name='inverse_variance'))
    new_hdul.append(fits.ImageHDU(mask_and,name='mask_and'))
    new_hdul.append(fits.ImageHDU(mask_or,name='mask_or'))
    new_hdul.append(fits.BinTableHDU(table_hdu,name='bin_table'))
    new_hdul.append(fits.ImageHDU(sky,name='mean_sky'))
    return new_hdul

def get_aperture_fast(j,columnspec_array,apertures_profile_middle,middle_column,trace_order,trace_rejection_sigma,trace_rejection_iterations,image_boundary,trace_shift_max,trace_nlost_max,profile_rejection_iterations,profile_nsample,profile_order,window):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models,fitting
    from astropy.modeling import models
    from specutils.spectra import Spectrum1D

    trace_x,trace_y,trace_z=m2fs.get_trace_xyz(j,columnspec_array,apertures_profile_middle,middle_column,trace_shift_max,trace_nlost_max)
    trace_func_init=models.Polynomial1D(degree=trace_order)
    fitter=fitting.LinearLSQFitter()
    y=np.ma.masked_array(trace_y,mask=False)
    z=np.ma.masked_array(trace_z,mask=False)
    sig_y=trace_y#fudge for now, not used for anything

    trace_pix_min=[]
    trace_pix_max=[]
    for i in range(0,len(trace_x)):
        trace_pix_min.append(fftinterp(x=trace_y[i],xp=[image_boundary.lower[q][1] for q in range(0,len(image_boundary.lower))],fp=[image_boundary.lower[q][0] for q in range(0,len(image_boundary.lower))]))
        trace_pix_max.append(fftinterp(x=trace_y[i],xp=[image_boundary.upper[q][1] for q in range(0,len(image_boundary.upper))],fp=[image_boundary.upper[q][0] for q in range(0,len(image_boundary.upper))]))
    trace_pix_min=np.array(trace_pix_min)
    trace_pix_max=np.array(trace_pix_max)

    y.mask[trace_x<trace_pix_min]=True
    y.mask[trace_x>trace_pix_max]=True
    z.mask[trace_x<trace_pix_min]=True
    z.mask[trace_x>trace_pix_max]=True

    trace_func=trace_func_init
    trace_rms=np.sqrt(np.mean((y-trace_func(trace_x))**2))

    for q in range(0,trace_rejection_iterations):
        use=np.where(y.mask==False)[0]
        if len(use)>=3:
            trace_func=fitter(trace_func_init,trace_x[use],trace_y[use])
            trace_rms=np.sqrt(np.mean((y[use]-trace_func(trace_x)[use])**2))
            outlier=np.where(np.abs(trace_y-trace_func(trace_x))>trace_rejection_sigma*trace_rms)[0]
            y.mask[outlier]=True
    use=np.where(y.mask==False)[0]
    trace_npoints=len(use)

    order=[trace_order]
    rejection_sigma=[trace_rejection_sigma]
#    pix_min=[trace_pix_min]
#    pix_max=[trace_pix_max]
    func=[trace_func]
    rms=[trace_rms]
    npoints=[trace_npoints]
    if len(trace_x[y.mask==False])>0:
        pix_min=[np.min(trace_x[y.mask==False])]
        pix_max=[np.max(trace_x[y.mask==False])]
    else:
        pix_min=[float(-999.)]
        pix_max=[float(-999.)]


    print(j,trace_npoints,pix_min,pix_max)

#    fig=plt.figure(1)
#    ax1,ax2=plot_trace(j,trace_x,trace_y,trace_z,y,trace_func,np.median(trace_pix_min),np.median(trace_pix_max),trace_rms,trace_npoints,trace_order,fig)

#    print('press \'d\' to delete point nearest cursor')
#    print('press \'c\' to un-delete deleted point nearest cursor')
#    print('press \'l\' to re-set lower limit of fit region')
#    print('press \'u\' to re-set lower limit of fit region')
#    print('press \'i\' to iterate outlier rejection')
#    print('press \'r\' to change rejection sigma threshold')
#    print('press \'o\' to change order of fit')
#    print('press \'z\' to return to original and erase edits')
#    cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_trace(event,[j,trace_x,trace_y,trace_z,y,sig_y,pix_min,pix_max,func,rms,npoints,order,rejection_sigma,trace_rejection_iterations,trace_func_init,fig]))
#    plt.show()
#    plt.close()

    if len(trace_x[y.mask==False])>0:
        pix_min.append(np.min(trace_x[y.mask==False]))
        pix_max.append(np.max(trace_x[y.mask==False]))
    else:
        pix_min.append(float(-999.))
        pix_max.append(float(-999.))
    profile_init=models.Polynomial1D(degree=profile_order)
    profile_fitter=fitting.LinearLSQFitter()

    x=np.array([np.median(columnspec_array[k].columns) for k in range(0,len(columnspec_array))])
    if apertures_profile_middle.realvirtual[j]:#if this is a real aperture, fit line profile as function of pixel
        y1=[]
        y2=[]
        for i in range(0,len(x)):#re-fit Gaussian profile at different x values along spectral axis
            center=func[len(func)-1](x[i])
            if((center>0.)&(center<np.max(columnspec_array[i].pixel.value))):#make sure the trace function y(x) makes sense at this x
                spec1d=Spectrum1D(spectral_axis=columnspec_array[i].pixel,flux=columnspec_array[i].spec*u.electron,uncertainty=columnspec_array[i].err,mask=columnspec_array[i].mask)
                ## PIC: Instead of refitting the entire thing all the time, I assume that the width is fixed 
                ## (estimated from the middle profile.)
                ## There "all" we need to update is the maximum position and value
                # pos, amp = peak_finder((spec1d-columnspec_array[i].continuum(columnspec_array[i].pixel.value)).data)
                # newmeanidx = np.where((pos-center)**2==np.min((pos-center)**2))[0][0]
                # newmean = pos[newmeanidx]
                # newamp = amp[newmeanidx]
                # g_fit0 = models.Gaussian1D(amplitude=newamp*u.electron, mean=newmean*u.AA,
                #                            stddev=apertures_profile_middle.fit[j].stddev.value*u.AA)
                ## But now that I optimized fit_aperture, we can try with the old method:
                subregion0,g_fit0=fit_aperture(spec1d-columnspec_array[i].continuum(columnspec_array[i].pixel.value),window,center)
                y1.append(g_fit0.stddev.value)
                y2.append(g_fit0.amplitude.value)
            else:#otherwise give a place-holder value and mask it below
                y1.append(float(-999.))
                y2.append(float(-999.))
        y1=np.array(y1)
        y2=np.array(y2)
        sigma_y=np.ma.masked_array(y1,mask=False)
        sigma_y.mask[y1<-998.]=True
        sigma_y.mask[x<pix_min[len(pix_min)-1]]=True
        sigma_y.mask[x>pix_max[len(pix_max)-1]]=True
        amplitude_y=np.ma.masked_array(y2,mask=False)
        amplitude_y.mask[y2<-998.]=True
        amplitude_y.mask[x<pix_min[len(pix_min)-1]]=True
        amplitude_y.mask[x>pix_max[len(pix_max)-1]]=True

        profile_sigma=profile_init
        profile_sigma_rms=np.sqrt(np.mean((y[use]-profile_sigma(x[use]))**2))
        profile_amplitude=profile_init
        profile_amplitude_rms=np.sqrt(np.mean((y[use]-profile_amplitude(x[use]))**2))
        for q in range(0,profile_rejection_iterations):
            use=np.where((sigma_y.mask==False)&(amplitude_y.mask==False))[0]
            if len(use)>3.:
                profile_sigma=profile_fitter(profile_init,x[use],sigma_y[use])
                profile_sigma_rms=np.sqrt(np.mean((sigma_y[use]-profile_sigma(x[use]))**2))
                profile_amplitude=profile_fitter(profile_init,x[use],amplitude_y[use])
                profile_amplitude_rms=np.sqrt(np.mean((amplitude_y[use]-profile_amplitude(x[use]))**2))
                outlier1=(np.where(np.abs(sigma_y-profile_sigma(x))>3.*profile_sigma_rms))[0]#reject outliers
                outlier2=(np.where(np.abs(amplitude_y-profile_amplitude(x))>3.*profile_amplitude_rms))[0]#reject outliers
                sigma_y.mask[outlier1]=True
                amplitude_y.mask[outlier2]=True
        use=np.where((sigma_y.mask==False)&(amplitude_y.mask==False))[0]
        profile_npoints=len(use)

#        print(profile_sigma_rms/np.median(sigma_y[sigma_y.mask==False]),profile_amplitude_rms/np.median(amplitude_y[amplitude_y.mask==False]))
#        plt.scatter(x,amplitude_y,s=3,color='k')
#        plt.scatter(x[amplitude_y.mask==True],amplitude_y[amplitude_y.mask==True],s=13,marker='x',color='y')
#        plt.plot(x,profile_amplitude(x),color='r')
#        plt.axvline(x=pix_min[len(pix_min)-1],lw=0.3,linestyle=':',color='k')
#        plt.axvline(x=pix_max[len(pix_max)-1],lw=0.3,linestyle=':',color='k')
##        plt.ylim([1,3])
#        plt.xlabel('x [pixel]')
#        plt.ylabel('LSF sigma [pixel]')
#        plt.show()
#        plt.close()
    else:
        profile_sigma=-999
        profile_sigma_rms=-999
        profile_amplitude=-999
        profile_amplitude_rms=-999
        profile_npoints=-999
    aperture0=m2fs.aperture(trace_aperture=j+1,trace_func=func[len(func)-1],trace_rms=rms[len(rms)-1],trace_npoints=npoints[len(npoints)-1],trace_pixel_min=pix_min[len(pix_min)-1],trace_pixel_max=pix_max[len(pix_max)-1],profile_sigma=profile_sigma,profile_sigma_rms=profile_sigma_rms,profile_amplitude=profile_amplitude,profile_amplitude_rms=profile_amplitude_rms,profile_npoints=profile_npoints)
    return aperture0

## Add some function to read and format fibermaps
def check_headers(nbfibers, ccd, all_list, filedict):

    ## Check that the headers are the same for all files.
    statuses = []
    headers = []
    objtype = []
    for file_id in all_list:
        filename = filedict[(ccd, file_id)]
        plugmapdic = gen_plugmap(filename)
        _headers = []
        _statuses = []
        _objtype = []
        for cassette in np.arange(1,9):
            for fibernb in np.arange(1,17):
                _headers.append('FIBER{}{}'.format(cassette, fibernb))
                _statuses.append(plugmapdic[cassette][fibernb]['identifier'])
                _objtype.append(plugmapdic[cassette][fibernb]['objtype'])

        if len(_statuses)!=nbfibers:
            raise Exception("gen_plugmap: Cannot ID 128 fibers?")
        statuses.append(_statuses)
        headers.append(_headers)
        objtype.append(_objtype)
    ## Check that we have the same thing for all files:
    for i in range(nbfibers):
        _stat = statuses[0][i] ## First file status
        _head = headers[0][i]
        _obj = objtype[0][i]
        for j in range(1, len(all_list)):
            if statuses[j][i]!=_stat:
                raise Exception('gen_plugmap: inconsistency in status headers')
            if headers[j][i]!=_head:
                raise Exception('gen_plugmap: inconsistency in status headers')
            if objtype[j][i]!=_obj:
                raise Exception('gen_plugmap: inconsistency in status headers')

# def read_header(nbfibers, ccd, all_list, filedict):
#     '''Function to read a fibermap from header
#     Input parameters:
#     - nbfibers  :   number of fibers expected'''
#     ## We then take the first file headers: 
#     objtype = objtype[0]
#     identifiers = statuses[0]
#     _statuses, _headers, _objtype = gen_plugmap(filedict[(ccd, all_list[0])])

def gen_plugmap(filename):
    '''Function to read the fits header a build a plug map.
    The function will check all the fits headers for inconsistencies.
    For now we only check the ThAr, LED, and Science frames.
    '''     
    ids = []
    headers = []
    objtype = []

    hdu = fits.open(filename)
    for header in hdu[0].header:
        if 'FIBER' in header:
            stat = hdu[0].header[header]
            ids.append(stat)
            headers.append(header)
            if 'unplug' in stat.lower():
                objtype.append('unused')
            elif 'sky' in stat.lower():
                objtype.append('SKY')
            else:
                objtype.append('TARGET')
    hdu.close()

    plugmapdic = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}} ## Initialize cassettes dics
    for i in range(len(headers)):
        fiberstring = headers[i].replace('FIBER', '')
        cassette = int(float(fiberstring[0])) ## This is the cassette number
        fibernb = int(float(fiberstring[1:])) ## This is the fiber number
        plugmapdic[cassette][fibernb] = {'objtype': objtype[i], 'identifier': ids[i]}

    return plugmapdic #ids, headers, objtype

def gen_plugmap_from_file(filename, ccd):
    '''Function to read the fits header a build a plug map.
    The function will check all the fits headers for inconsistencies.
    For now we only check the ThAr, LED, and Science frames.
    '''     
    ids = []
    headers = []
    objtype = []

    #### Identify the file type.
    ## 
    rawfile = False
    f = open(filename, 'r')
    for line in f.readlines():
        if '[assignments]' in line:
            rawfile = True
    f.close()

    lines = [] ## Store the lines
    #
    if rawfile:
        record = False
        f = open(filename, 'r')
        for line in f.readlines():
            if '[assignments]' in line: ## Data block
                # print('recording')
                record = True
                continue
            elif '[guides]' in line:
                # print('breaking')
                record=False
                break
            if record:
                if 'fiber' in line: 
                    pass
                else:
                    lines.append(line)
        f.close()   
    else:
        f = open(filename, 'r')
        for line in f.readlines():
            lines.append(line)
        f.close()

    for line in lines:
        fiber = line.split()[0].lower()
        stat = line.split()[1]
        if ccd not in fiber: continue
        headers.append(fiber.replace(ccd, 'FIBER').replace('-', '')) ## This is just formating  
        ids.append(stat)
        if 'unplug' in stat.lower():
            objtype.append('unused')
        elif 'sky' in stat.lower():
            objtype.append('SKY')
        else:
            objtype.append('TARGET')

    plugmapdic = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}} ## Initialize cassettes dics
    for i in range(len(headers)):
        fiberstring = headers[i].replace('FIBER', '')
        cassette = int(float(fiberstring[0])) ## This is the cassette number
        fibernb = int(float(fiberstring[1:])) ## This is the fiber number
        plugmapdic[cassette][fibernb] = {'objtype': objtype[i], 'identifier': ids[i]}

    return plugmapdic #ids, headers, objtype

def order_halphali(plugmapdic, cassettes_order):
    '''Function to order the Halpha-Li filter.
    The order has two consecutive orders.
    In this mode, I assume that only impair '''

    objtypes = []
    apertures = []
    identifiers = []
    fibers = []

    aperture=1 ## We're gonna count the apertures
    for cassette in cassettes_order:
        for fibernb in range(16, 0, -1): ## from top to botton, for both ccd is the same
            # if 'unused' in plugmapdic[cassette][fibernb]['objtype']: continue
            ## In this mode I assume that only fibers with impair numbers are being used.
            if fibernb%2==0: continue
            # First order
            apertures.append(aperture)
            objtypes.append(plugmapdic[cassette][fibernb]['objtype'])
            identifiers.append(plugmapdic[cassette][fibernb]['identifier'])
            fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
            aperture+=1
            # Second order
            apertures.append(aperture)
            objtypes.append(plugmapdic[cassette][fibernb]['objtype'])
            identifiers.append(plugmapdic[cassette][fibernb]['identifier'])
            fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
            aperture+=1

    ## We may be having less than 128 apertures, and yet, our program currently needs 128
    ## For now I "complete" the missing apertures to reach 128. In the future we could
    ## Try to do things in a more clever way?
    ## I am still confused by this 128 number - I can't quite understand if that's an absolute
    ## maximum, since I see many more with the blue filter. And with Novembder Halpha-Li data
    ## We have only 120 plugged fibers, but still 128 traces...

    ## THIS WILL NOT WORK BECAUSE I WOULD END UP ADDING IMPROPER LABELS TO THE FIBERS !
    # while len(fibers)<128: 
    #     apertures.append(aperture)
    #     objtypes.append('UNUSED')
    #     identifiers.append('')
    #     fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
    #     aperture+=1
    
    return apertures, identifiers, objtypes, fibers

def order_dupreeblue(plugmapdic, cassettes_order):
    '''Function to order the dupreeblue filter.
    The order has 4 NON-consecutive orders (in practive more, things are messed up,
    but we can used one side of the image to try to count 128.'''

    objtypes = []
    apertures = []
    identifiers = []
    fibers = []

    aperture=1 ## We're gonna count the apertures
    for cassette in cassettes_order:
        for fibernb in range(16, 0, -2): ## from top to botton, for both ccd is the same
            if 'unused' in plugmapdic[cassette][fibernb]['objtype']: continue
            for order in range(4,0,-1):
                # first fiber
                apertures.append(aperture)
                objtypes.append(plugmapdic[cassette][fibernb]['objtype'])
                identifiers.append(plugmapdic[cassette][fibernb]['identifier'])
                fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
                aperture+=1
                # second fiber
                apertures.append(aperture)
                objtypes.append(plugmapdic[cassette][fibernb-1]['objtype'])
                identifiers.append(plugmapdic[cassette][fibernb-1]['identifier'])
                fibers.append('FIBER{}{:02d}'.format(cassette, fibernb-1))
                aperture+=1

    ## We may be having less than 128 apertures, and yet, our program currently needs 128
    ## For now I "complete" the missing apertures to reach 128. In the future we could
    ## Try to do things in a more clever way?
    ## I am still confused by this 128 number - I can't quite understand if that's an absolute
    ## maximum, since I see many more with the blue filter. And with Novembder Halpha-Li data
    ## We have only 120 plugged fibers, but still 128 traces...

    # from IPython import embed
    # embed()
    ## This also simply cannot work properly.
    # while len(fibers)<128: 
    #     apertures.append(aperture)
    #     objtypes.append('UNUSED')
    #     identifiers.append('')
    #     fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
    #     aperture+=1

    return apertures, identifiers, objtypes, fibers


def order_hkhires(plugmapdic, cassettes_order):
    '''Function to order the dupreeblue filter.
    The order has 4 NON-consecutive orders (in practive more, things are messed up,
    but we can used one side of the image to try to count 144.
    NOTE THAT I ACTUALLY CONSIDER 3 ORDERS HERE'''

    objtypes = []
    apertures = []
    identifiers = []
    fibers = []

    ## First pass: what are the valid fibers percassettes?
    validfibers = {}
    for cassette in cassettes_order:
        _valids = []
        for fibernb in range(16, 0, -1): ## from top to botton, for both ccd is the same
            if 'unused' not in plugmapdic[cassette][fibernb]['objtype']: 
                _valids.append(fibernb)
        validfibers[cassette] = _valids.copy()

    aperture=1 ## We're gonna count the apertures
    for cassette in cassettes_order:
        fiber_numbers = validfibers[cassette]
        for iter in range(0, len(fiber_numbers), 2): ## from top to botton, for both ccd is the same
            fibernb = fiber_numbers[iter]
            if 'unused' in plugmapdic[cassette][fibernb]['objtype']: 
                print("FIBER{}{:02d}".format(cassette, fibernb))
                continue
            if 'unused' in plugmapdic[cassette][fibernb]['objtype']: 
                print("FIBER{}{:02d}".format(cassette, fiber_numbers[iter+1]))
                continue
            for order in range(3,0,-1):
                # first fiber
                apertures.append(aperture)
                objtypes.append(plugmapdic[cassette][fibernb]['objtype'])
                identifiers.append(plugmapdic[cassette][fibernb]['identifier'])
                fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
                aperture+=1
                # second fiber
                apertures.append(aperture)
                objtypes.append(plugmapdic[cassette][fiber_numbers[iter+1]]['objtype'])
                identifiers.append(plugmapdic[cassette][fiber_numbers[iter+1]]['identifier'])
                fibers.append('FIBER{}{:02d}'.format(cassette, fiber_numbers[iter+1]))
                aperture+=1

    ## We may be having less than 128 apertures, and yet, our program currently needs 128
    ## For now I "complete" the missing apertures to reach 128. In the future we could
    ## Try to do things in a more clever way?
    ## I am still confused by this 128 number - I can't quite understand if that's an absolute
    ## maximum, since I see many more with the blue filter. And with Novembder Halpha-Li data
    ## We have only 120 plugged fibers, but still 128 traces...

    # from IPython import embed
    # embed()
    ## This also simply cannot work properly.
    # while len(fibers)<128: 
    #     apertures.append(aperture)
    #     objtypes.append('UNUSED')
    #     identifiers.append('')
    #     fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
    #     aperture+=1

    return apertures, identifiers, objtypes, fibers

def order_nadhiro60(plugmapdic, cassettes_order):
    '''Function to order the dupreeblue filter.
    The order has consecutive orders (1 per star only)'''

    objtypes = []
    apertures = []
    identifiers = []
    fibers = []

    ## First pass: what are the valid fibers percassettes?
    validfibers = {}
    for cassette in cassettes_order:
        _valids = []
        for fibernb in range(16, 0, -1): ## from top to botton, for both ccd is the same
            if 'unused' not in plugmapdic[cassette][fibernb]['objtype']: 
                _valids.append(fibernb)
        validfibers[cassette] = _valids.copy()

    aperture=1 ## We're gonna count the apertures
    for cassette in cassettes_order:
        fiber_numbers = validfibers[cassette]
        for iter in range(0, len(fiber_numbers), 1): ## from top to botton, for both ccd is the same
            fibernb = fiber_numbers[iter]
            if 'unused' in plugmapdic[cassette][fibernb]['objtype']: 
                print("FIBER{}{:02d}".format(cassette, fibernb))
                continue
            if 'unused' in plugmapdic[cassette][fibernb]['objtype']: 
                print("FIBER{}{:02d}".format(cassette, fiber_numbers[iter+1]))
                continue
            for order in [0]: ## There is only one order per fiber
                # first fiber ## And only one fiber per star here
                apertures.append(aperture)
                objtypes.append(plugmapdic[cassette][fibernb]['objtype'])
                identifiers.append(plugmapdic[cassette][fibernb]['identifier'])
                fibers.append('FIBER{}{:02d}'.format(cassette, fibernb))
                aperture+=1

    return apertures, identifiers, objtypes, fibers

def order_fibers(plugmapdic, ccd, filter):
    '''Function to order the fiber based on filter and CCD'''
    
    if 'r' in ccd: ## Red ccd
        cassettes_order = np.arange(8, 0, -1) ## From top to bottom of image
    elif 'b' in ccd: ## Red ccd
        cassettes_order = np.arange(1, 9, 1) ## From top to bottom of image
    if 'halphali' in filter:
        apertures, identifiers, objtypes, fibers = order_halphali(plugmapdic, cassettes_order)
    elif 'dupreeblue' in filter:
        apertures, identifiers, objtypes, fibers = order_dupreeblue(plugmapdic, cassettes_order)
    elif 'hkhires' in filter:
        apertures, identifiers, objtypes, fibers = order_hkhires(plugmapdic, cassettes_order)
    elif 'nadhiro60' in filter: ## I assume that the layout is the same for na filter --> EXCEPT IT ISN'T
        apertures, identifiers, objtypes, fibers = order_nadhiro60(plugmapdic, cassettes_order)
    else:
        raise Exception('function_dump.order_fibers: unknown filter {}, try options: {}, {}'.format(filter, 'halphali', 'dupreeblue', 'hkhires'))
    return apertures, identifiers, objtypes, fibers


#### Two functions here to perform a fourier interpolation
from scipy.signal import resample
def xresample(x, y, factor=4):
    ny = resample(y, factor*len(y)) ## Oversampled array
    ny = ny[:-factor+1] ## Remove ending
    #
    xx = np.linspace(0, len(x)-1, factor*len(x)-factor+1) ## Build x array
    nx = np.interp(xx, np.arange(len(x)), x) ## interpolate wavelength solution
    #
    return nx, ny

def fftinterp(x, xp, fp, factor=4):
    '''Function to perform Fourrier interpolation of the
    data. The fuction relied on scipy.signal ressample.
    It assumed an array evenly spaced in bins, and estimate
    the wavelength solution associated by performing a linear
    interpolation of the wavelength solution.
    The syntax was adapted to make sure it can replace the np.interp
    function.'''
    ## Trying to retain the astropy quantity type.
    mytype = type(fp)
    if mytype==astropy.units.quantity.Quantity:
        myunit = fp.unit
    else:
        myunit=1
    #
    _x, _y = xresample(xp, fp, factor=factor)
    res = np.interp(x, _x, _y)
    return res*myunit ## Should allow converting to quantity

def get_extract1d(j,data,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width,offset=0):
    import numpy as np
    import scipy
    from astropy.nddata import CCDData
    from specutils.spectra import Spectrum1D
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty

    above0,below0=m2fs.get_above_below(j,data,aperture_array,apertures_profile_middle,aperture_peak)

    spec=[]
    spec_error=[]
    spec_mask=[]
    # from IPython import embed
    # embed()
    ymids=aperture_array[j].trace_func(pix)+offset
    profile_sigmas=aperture_array[j].profile_sigma(pix)
    extract1d_aperture_widths = np.ones(len(profile_sigmas))*extract1d_aperture_width
    halfwidths = np.ones(len(profile_sigmas))*(above0-below0)/2./2.
    wings=np.amin([extract1d_aperture_widths,3.*profile_sigmas,halfwidths], axis=0)
    _data = np.array(data.data, dtype=float)
    _mask = np.array(data.mask, dtype=float)
    _uncertainty = np.array(data.uncertainty.array, dtype=float)
    spec, spec_error, spec_mask = extract_aperture(pix, _data, _mask, _uncertainty, ymids, profile_sigmas, wings)

#     for x in pix:
#         ymid=aperture_array[j].trace_func(x)+offset
#         profile_sigma=aperture_array[j].profile_sigma(x)
#         wing=np.min([extract1d_aperture_width,3.*profile_sigma,(above0-below0)/2./2.])
# #        wing=3.
#         y1=int(ymid-wing)
#         y2=int(ymid+wing)
#         sss=data[y1:y2+1,x]
#         if ((wing>0.)&(len(np.where(sss.mask==False)[0])>=1)):
# #            sum1=CCDData([0.],unit=data.unit,mask=[False])
#             sum1=0.
#             sum2=0.
#             for k in range(0,len(sss.data)):
#                 if sss.mask[k]==False:
#                     int1=0.5*(scipy.special.erf(ymid/profile_sigma/np.sqrt(2.))-scipy.special.erf((ymid-(y1-0.5+k))/profile_sigma/np.sqrt(2.)))
#                     int2=0.5*(scipy.special.erf(ymid/profile_sigma/np.sqrt(2.))-scipy.special.erf((ymid-(y1-0.5+k+1))/profile_sigma/np.sqrt(2.)))
#                     weight=int2-int1
# #                    sum1=sum1.add((sss[k].multiply(weight)).divide(sss.uncertainty.quantity.value[k]**2))
#                     sum1+=sss.data[k]*weight/sss.uncertainty.quantity.value[k]**2
#                     sum2+=weight**2/sss.uncertainty.quantity.value[k]**2
# #            val=sum1.divide(sum2)
#             val=sum1/sum2
#             err=1./np.sqrt(sum2)
#             spec.append(val)
#             spec_error.append(err)
#             if ((val==val)&(err>0.)&(err==err)):
#                 spec_mask.append(False)
#             else:
#                 spec_mask.append(True)
# #            spec.append(sss.data[0])
# #            spec_error.append(sss.uncertainty.quantity.value[0])
# #            spec_mask.append(False)            
#         else:
#             spec.append(0.)
#             spec_mask.append(True)
#             spec_error.append(999.)
    # spec=np.array(spec)
    # spec_mask=np.array(spec_mask)
    # spec_error=np.array(spec_error)

    spec=np.array(spec, dtype=float)
    spec_mask=np.array(spec_mask, dtype=bool) ## Must be bool otherwise we'll have problems later
    spec_error=np.array(spec_error, dtype=float)

    spec_mask[np.where(((pix<aperture_array[j].trace_pixel_min)|(pix>aperture_array[j].trace_pixel_max)))[0]]=True
    return m2fs.extract1d(aperture=aperture_array[j].trace_aperture,spec1d_pixel=pix,spec1d_flux=spec*data.unit,spec1d_uncertainty=StdDevUncertainty(spec_error),spec1d_mask=spec_mask)


from numba import jit
@jit(nopython=True)
def numbaerf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

import scipy
@jit(nopython=True, cache=True)
def extract_aperture(pix, _data, _mask, _uncertainty, ymids, profile_sigmas, wings):
    '''Jit function for fast extraction of an aperture'''    
    myspec = np.ones(len(pix))
    myerr = np.ones(len(pix))    
    mymask = np.ones(len(pix))*True
    _data_debug = np.zeros(_data.shape)
    _data_debug2 = np.zeros(_data.shape)
    for x in pix:
#        wing=3.
        ymid = ymids[x]
        profile_sigma = profile_sigmas[x]
        wing = wings[x]
        ymid = ymids[x]
        y1=int(ymid-wing)
        y2=int(ymid+wing)
        data=_data[y1:y2+1,x]
        mask=_mask[y1:y2+1,x]
        uncertainty = _uncertainty[y1:y2+1,x]
        if ((wing>0.)&(len(np.where(mask==0)[0])>=1)):
#            sum1=CCDData([0.],unit=data.unit,mask=[False])
            sum1=0.
            sum2=0.
            for k in range(0,len(data)):
                if (mask[k]==False) & (~np.isnan(uncertainty[k])):
                    int1=0.5*(numbaerf(ymid/profile_sigma/np.sqrt(2.))-numbaerf((ymid-(y1-0.5+k))/profile_sigma/np.sqrt(2.)))
                    int2=0.5*(numbaerf(ymid/profile_sigma/np.sqrt(2.))-numbaerf((ymid-(y1-0.5+k+1))/profile_sigma/np.sqrt(2.)))
                    #
                    weight=int2-int1
#                    sum1=sum1.add((sss[k].multiply(weight)).divide(uncertainty.quantity.value[k]**2))
                    if uncertainty[k]==0.:
                        print('extract_aperture: 0 value in uncertainty')
                    if np.isnan(uncertainty[k]):
                        print('extract_aperture: NaN value in uncertainty')
                    # sum1+=data[k]*weight/uncertainty[k]**2
                    # sum2+=weight**2/uncertainty[k]**2
                    sum1+=data[k]*weight
                    sum2+=weight**2
                    _data_debug[y1+k, x] = data[k]*weight/uncertainty[k]**2
                    _data_debug2[y1+k, x] = data[k]*weight/uncertainty[k]**2
#            val=sum1.divide(sum2)
            if sum2==0.:
                # print('extract_aperture: 0 value in sum2')
                val=0.
                err=999.
            else:
                val=sum1/sum2
                err=1./np.sqrt(sum2)
            myspec[x] = val
            myerr[x] = err
            if ((val==val)&(err>0.)&(err==err)):
                mymask[x] = False
            else:
                mymask[x] = True
#            spec.append(sss.data[0])
#            spec_error.append(sss.uncertainty.quantity.value[0])
#            spec_mask.append(False)            
        else:
            myspec[x] = 0.
            mymask[x] = True
            myerr[x] = 999.
    
    # from IPython import embed
    # embed()
    # plt.figure()
    # plt.imshow(_data_debug, vmin=0, 
    #            vmax=1)
    # plt.show()

    # plt.figure()
    # plt.plot(np.mean(_data_debug, 0))
    # plt.show()

    # plt.figure()
    # plt.plot(myspec)
    # plt.show()
    return myspec, myerr, mymask        

# @jit(nopython=True)
def extract_aperture2(pix, _data, _mask, _uncertainty, ymids, profile_sigmas, wings):
    '''PIC: I'm just trying to not use the formula that I don't understand'''
    myspec = np.ones(len(pix))
    myspec2 = np.ones(len(pix))
    myweight2 = np.ones(len(pix))
    myerr = np.ones(len(pix))    
    mymask = np.ones(len(pix))*True
    for x in pix:
#        wing=3.
        ymid = ymids[x]
        profile_sigma = profile_sigmas[x]
        wing = wings[x]
        ymid = ymids[x]
        y1=int(ymid-wing)
        y2=int(ymid+wing)
        data=_data[y1:y2+1,x]
        mask=_mask[y1:y2+1,x]
        uncertainty = _uncertainty[y1:y2+1,x]
        if x==1000:
            plt.figure()
            plt.plot(_data[:,x])
            plt.plot(np.arange(y1, y2+1), _data[y1:y2+1,x])
            plt.show()
        if ((wing>0.)&(len(np.where(mask==0)[0])>=1)):
#            sum1=CCDData([0.],unit=data.unit,mask=[False])
            sum1=0.
            sum2=0.
            mysum=0.
            mysubweights = np.zeros(len(data))
            for k in range(0,len(data)):
                # if mask[k]==False:
                if True is True:
                    int1=0.5*(numbaerf(ymid/profile_sigma/np.sqrt(2.))-numbaerf((ymid-(y1-0.5+k))/profile_sigma/np.sqrt(2.)))
                    int2=0.5*(numbaerf(ymid/profile_sigma/np.sqrt(2.))-numbaerf((ymid-(y1-0.5+k+1))/profile_sigma/np.sqrt(2.)))
                    weight=int2-int1
#                    sum1=sum1.add((sss[k].multiply(weight)).divide(uncertainty.quantity.value[k]**2))
                    sum1+=data[k]*weight/uncertainty[k]**2
                    sum2+=weight**2/(uncertainty[k]/2)**2
                    mysum+=data[k]*weight
                    mysubweights[k]+=weight
#            val=sum1.divide(sum2)
            val=sum1/sum2
            err=1./np.sqrt(sum2)
            myspec[x] = val
            myerr[x] = err
            myspec2[x] = mysum
            myweight2[x] = weight
            if ((val==val)&(err>0.)&(err==err)):
                mymask[x] = False
            else:
                mymask[x] = True
#            spec.append(sss.data[0])
#            spec_error.append(sss.uncertainty.quantity.value[0])
#            spec_mask.append(False)            
        else:
            myspec[x] = 0.
            mymask[x] = True
            myerr[x] = 999.
        # if x==1000:
        #     print(ymid)
        #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        #     ax1.plot(data)
        #     ax1.plot(mysubweights)
        #     ax2.plot(data - mysubweights)
        #     plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(myspec)
    fac = 1
    ax1.plot(myspec2*fac)
    ax2.plot(myspec - myspec2*fac)
    plt.show()
    return myspec, myerr, mymask        

def get_scatteredlightfunc(data,apmask,scatteredlightcorr_order,scatteredlightcorr_rejection_iterations,scatteredlightcorr_rejection_sigma):
    '''The original function had very little chance to give something accurate if we did not use all of the
    apertures at once. This is because it tries to mask the apertures and then dertermine the scattered light
    on what is left. This will not work. What we can try is having a second mask to remove ALL the apertures
    independently from what we estimated.
    Then the function fits a 2D polynomial on the surface at once. That appears to be little different from
    what IRAF does...'''
    
    
    import numpy as np
    import astropy.units as u
    from astropy.modeling import models
    from specutils.spectra import Spectrum1D
    from astropy.modeling import models,fitting
    import matplotlib.pyplot as plt

    # from IPython import embed
    # embed()

    # mymask = np.copy(apmask)
    # mymask = ~mymask
    # plt.figure()
    # plt.imshow(data*(~mymask), vmax=100)
    # plt.show()


    # plt.figure()
    # plt.plot(data.data[1000])
    # plt.plot(scattered_model.data[1000])
    # plt.show()

    func_init=models.Polynomial2D(degree=scatteredlightcorr_order)
    fitter=fitting.LinearLSQFitter()
    y,x=np.mgrid[:len(data.data),:len(data.data[0])]

    for q in range(0,scatteredlightcorr_rejection_iterations):
        use=np.where(apmask==False)[0]
        func=fitter(func_init,x[apmask==False],y[apmask==False],data.data[apmask==False])
        rms=np.sqrt(np.mean((data.data[apmask==False]-func(x[apmask==False],y[apmask==False]))**2))
        outlier=np.where(np.abs(data.data-func(x,y))>scatteredlightcorr_rejection_sigma*rms)
#        outlier=np.where(data.data-func(x,y)>scatteredlightcorr_rejection_sigma*rms)
        apmask[outlier]=True
    return m2fs.scatteredlightfunc(func=func,rms=rms)

def get_apmask2(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary):
    import numpy as np
    from astropy.nddata import CCDData
    from specutils.spectra import Spectrum1D
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty

    apmask0=np.array(data.mask,dtype=bool)
    pix=np.arange(len(apmask0[0]))

    apmask=m2fs.mask_boundary(apmask0,image_boundary)

    for j in range(0,len(aperture_array)):
        # if apertures_profile_middle.realvirtual[j]:
            above0,below0=m2fs.get_above_below(j,data,aperture_array,apertures_profile_middle,aperture_peak)
            for x in pix:
#                if ((x>=aperture_array[j].trace_pixel_min)&(x<=aperture_array[j].trace_pixel_max)):
                ymid=aperture_array[j].trace_func(x)
                if apertures_profile_middle.realvirtual[j]:
                    profile_sigma=aperture_array[j].profile_sigma(x)
                else:
                    profile_sigma = 2 ## Default value
                wing=np.min([3.*profile_sigma,(above0-below0)/2./2.])
                if wing>0.:
                    y1=int(ymid-wing)
                    y2=int(ymid+wing)
                    apmask[y1:y2,x]=True
                    if ((x<aperture_array[j].trace_pixel_min)|(x>aperture_array[j].trace_pixel_max)):
                        apmask[y1:y2,x]=True
                    if ((ymid<0.)|(ymid>len(data.data))):
                        apmask[y1:y2,x]=True
    return apmask

# @jit(nopython=True)
def trace_apmask_aperture_fast(apmask, apmask2, pix, trace_vals, realaperture, profile_sigmas, 
                               above0, below0, aperture_array_trace_pixel_min, aperture_array_trace_pixel_max, data):
    for ix, x in enumerate(pix):
        ymid = trace_vals[ix]
        if realaperture:
            profile_sigma=profile_sigmas[ix]
        else:
            profile_sigma = 2. # 2 ## Default value
        # if (3.*profile_sigma > (above0-below0)/3.):
        #     print("choosing wings size:")
        #     print(3.*profile_sigma,(above0-below0)/3.)
        # wing=np.min(np.array([3.*profile_sigma,(above0-below0)/3.]))
        if (3.*profile_sigma)>((above0-below0)/3.):
            wing = (above0-below0)/3.
        else: wing = 3.*profile_sigma

        if wing>0.:
            y1=int(ymid-wing)
            y2=int(ymid+wing)
            if realaperture:
                apmask[y1:y2,x]=True
                if ((x<aperture_array_trace_pixel_min)|(x>aperture_array_trace_pixel_max)):
                    apmask[y1:y2,x]=True
                if ((ymid<0.)|(ymid>len(data))):
                    apmask[y1:y2,x]=True
            ## Whether it is real or not, we populate the apmask2
            apmask2[y1:y2,x]=True
            if ((x<aperture_array_trace_pixel_min)|(x>aperture_array_trace_pixel_max)):
                apmask2[y1:y2,x]=True
            if ((ymid<0.)|(ymid>len(data))):
                apmask2[y1:y2,x]=True
    return apmask, apmask2
def get_apmask(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary):
    import numpy as np
    from astropy.nddata import CCDData
    from specutils.spectra import Spectrum1D
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty

    apmask0=np.array(data.mask,dtype=bool)
    pix=np.arange(len(apmask0[0]))

    apmask=m2fs.mask_boundary(apmask0,image_boundary)
    apmask2=m2fs.mask_boundary(apmask0,image_boundary)

    for j in range(0,len(aperture_array)):
        # if apertures_profile_middle.realvirtual[j]:
        above0,below0=m2fs.get_above_below(j,data,aperture_array,apertures_profile_middle,aperture_peak)
        ## Let's go faster with numba
        realaperture = np.array(apertures_profile_middle.realvirtual[j], dtype=float)
        trace_vals = np.array(aperture_array[j].trace_func(pix), dtype=float)
        if realaperture:
            profile_sigmas = np.array(aperture_array[j].profile_sigma(pix), dtype=float)
        else:
            profile_sigmas = np.ones(len(pix))*2. # 2 ## Default value
        mynewmask = np.array(np.copy(apmask), dtype=float)
        mynewmask2 = np.array(np.copy(apmask2), dtype=float)
        aperture_array_trace_pixel_min = aperture_array[j].trace_pixel_min
        aperture_array_trace_pixel_max = aperture_array[j].trace_pixel_max
        mydata = np.array(np.copy(data.data), dtype=float)
        mynewmask, mynewmask2 = trace_apmask_aperture_fast(mynewmask, mynewmask2, pix, trace_vals, realaperture, profile_sigmas, 
                                   above0, below0,
                                   aperture_array_trace_pixel_min, aperture_array_trace_pixel_max, mydata)
        # from IPython import embed
        # embed()
        # exit()

        # for x in pix:
        #     ymid=aperture_array[j].trace_func(x)
        #     if apertures_profile_middle.realvirtual[j]:
        #         profile_sigma=aperture_array[j].profile_sigma(x)
        #     else:
        #         profile_sigma = 2. # 2 ## Default value
        #     # if (3.*profile_sigma > (above0-below0)/3.):
        #     #     print("choosing wings size:")
        #     #     print(3.*profile_sigma,(above0-below0)/3.)
        #     wing=np.min([3.*profile_sigma,(above0-below0)/3.])
        #     if wing>0.:
        #         y1=int(ymid-wing)
        #         y2=int(ymid+wing)
        #         if apertures_profile_middle.realvirtual[j]:
        #             apmask[y1:y2,x]=True
        #             if ((x<aperture_array[j].trace_pixel_min)|(x>aperture_array[j].trace_pixel_max)):
        #                 apmask[y1:y2,x]=True
        #             if ((ymid<0.)|(ymid>len(data.data))):
        #                 apmask[y1:y2,x]=True
        #         ## Whether it is real or not, we populate the apmask2
        #         apmask2[y1:y2,x]=True
        #         if ((x<aperture_array[j].trace_pixel_min)|(x>aperture_array[j].trace_pixel_max)):
        #             apmask2[y1:y2,x]=True
        #         if ((ymid<0.)|(ymid>len(data.data))):
        #             apmask2[y1:y2,x]=True
    return apmask, apmask2


def get_apflat(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary,apmask,field_name):
    import numpy as np
    import dill as pickle
    import scipy
    from astropy.nddata import CCDData
    from specutils.spectra import Spectrum1D
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty
    import matplotlib.pyplot as plt

    pix=np.arange(len(data.data[0]))
    arr2d=np.zeros((len(data.data),len(data.data[0])))
    apmask0=~apmask#np.array(data.mask,dtype=bool)

    apflat_mask=mask_boundary(apmask0,image_boundary)

    for j in range(0,len(aperture_array)):
        print(j,len(aperture_array))
        if apertures_profile_middle.realvirtual[j]:
            
            above0,below0=get_above_below(j,data,aperture_array,apertures_profile_middle,aperture_peak)

            for x in pix:
                ymid=aperture_array[j].trace_func(x)
#                print(j,x,ymid)
                profile_sigma=aperture_array[j].profile_sigma(x)
                profile_amplitude=aperture_array[j].profile_amplitude(x)
                wing=np.min([3.*profile_sigma,(above0-below0)/2./2.])
                if wing>0.:
                    y1=int(ymid-wing)
                    y2=int(ymid+wing)
                    if ((x<aperture_array[j].trace_pixel_min)|(x>aperture_array[j].trace_pixel_max)):
                        apflat_mask[y1:y2,x]=True
                    if ((ymid<0.)|(ymid>len(data.data))):
                        apflat_mask[y1:y2,x]=True
                    if (y1>=0)&(y2<=len(data.data)):
                        sss=data[y1:y2+1,x]
#                a.mask[y1:y2,x]=True
                        for k in range(0,len(sss.data)):
                            int1=0.5*(scipy.special.erf(ymid/profile_sigma/np.sqrt(2.))-scipy.special.erf((ymid-(y1-0.5+k))/profile_sigma/np.sqrt(2.)))
                            int2=0.5*(scipy.special.erf(ymid/profile_sigma/np.sqrt(2.))-scipy.special.erf((ymid-(y1-0.5+k+1))/profile_sigma/np.sqrt(2.)))
                            weight=int2-int1
#                            print(k,y1,y2,ymid,len(sss.data),x)
                            arr2d[y1+k,x]=weight*profile_amplitude*np.sqrt(2.*np.pi*profile_sigma**2)
#                        print(weight,profile_amplitude,profile_sigma,data.data[y1+k,x],arr2d[y1+k,x],data.data[y1+k,x]/arr2d[y1+k,x])
    residual=CCDData((data.data-arr2d)/data.uncertainty.quantity.value,uncertainty=StdDevUncertainty(data.uncertainty.quantity.value/data.uncertainty.quantity.value),unit=u.electron/u.electron,mask=apflat_mask)
    rms=np.sqrt(np.mean((data.data[apmask==True]-arr2d[apmask==True])**2))
    new_masked=np.where(np.abs(residual.data)>13.)#this flags the 'ghosts' default was 3. I'm trying 13.
    if len(new_masked[0])>0:
        if 'twilight' not in field_name:#when twilight is its own flat-field spectrum, this doesn't work
            apflat_mask[new_masked]=True

#    expected_2darray=CCDData(arr2d,unit=data.unit,mask=data.mask,uncertainty=StdDevUncertainty(np.full((len(data.data),len(data.data[0])),rms,dtype='float')))
#    pickle.dump(expected_2darray,open('/nfs/nas-0-9/mgwalker.proj/m2fs/crap.pickle','wb'))
#    np.pause()
#    crap=data.uncertainty.quantity.value/arr2d
#    shite=~apmask
    apflat=CCDData(data.data/arr2d,unit=u.electron/u.electron,mask=apflat_mask,uncertainty=StdDevUncertainty(data.uncertainty.quantity.value/arr2d))

#    apflat=data.multiply(expected_2darray)    
 #   print(expected_2darray.uncertainty)
 #   print(' ')
#    apflatcorr=data.divide(apflat)
    return apflat,residual

def weightedmeanspec(stack00):
    import numpy as np

    spec=[]
    spec_err=[]
    spec_mask=[]

    for i in range(0,len(stack00[0].data)):
        keep=np.where([stack00[q].mask[i]==False for q in range(0,len(stack00))])[0]
        if len(keep)>0:
            sum1=np.sum(np.array([stack00[q].data[i] for q in keep])/np.array([stack00[q].uncertainty.quantity.value[i] for q in keep])**2)
            sum2=np.sum(1./np.array([stack00[q].uncertainty.quantity.value[i] for q in keep])**2)
            
            spec.append(sum1/sum2)
            spec_err.append(np.sqrt(1./sum2))
            spec_mask.append(False)
#            sum1=np.sum(np.array([stack00[q].data[i] for q in keep]))
#            sum2=np.sum(np.array([stack00[q].uncertainty.quantity.value[i] for q in keep])**2)
#            spec.append(sum1)
#            spec_err.append(np.sqrt(sum2))
#            spec_mask.append(False)
        else:
            spec.append(0.)
            spec_err.append(-999.)
            spec_mask.append(True)
    spec=np.array(spec)
    spec_err=np.array(spec_err)
    spec_mask=np.array(spec_mask)

    return spec,spec_err,spec_mask
    
def spec_median(stack00):
    import numpy as np

    data = []
    mask = []
    err = []
    for i in range(len(stack00)):
        data.append(stack00[i].data)
        mask.append(stack00[i].mask)
        err.append(np.sqrt(1./1./np.array(stack00[i].uncertainty.quantity.value)**2)) ## This is false
    
    data = np.array(data)
    mask = np.array(mask)
    err = np.array(err)
    
    data[mask] = np.nan
    plt.plot(data[0][~mask[0]])

    data = np.array(data)

    medspec = np.nanmedian(data, axis=0)
    spec_mask = np.isnan(medspec)
    spec_err = np.nanmean(err, axis=0)
    spec = medspec

    return spec,spec_err,spec_mask

def get_stack(stack0,j):
    from astropy.nddata import CCDData
    from specutils.spectra import Spectrum1D
    from astropy import units as u
    from astropy.nddata import StdDevUncertainty
    import numpy as np

    stack00=[]
    for q in range(0,len(stack0)):
#        print(len(stack0),len(stack0[0]),len(stack0[q]),q,j)
        stack00.append(CCDData(stack0[q][j].spec1d_flux,uncertainty=stack0[q][j].spec1d_uncertainty,mask=stack0[q][j].spec1d_mask))
#        print(np.where(stack0[q][j].spec1d_mask==False)[0])
    spec,spec_err,spec_mask=spec_median(stack00)
    return m2fs.extract1d(aperture=stack0[q][j].aperture,spec1d_pixel=stack0[q][j].spec1d_pixel*u.AA,spec1d_flux=spec*u.electron,spec1d_uncertainty=StdDevUncertainty(spec_err),spec1d_mask=spec_mask)

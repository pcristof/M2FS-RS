import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
#from astropy.nddata import NDData
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
import mycode

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
#m2fsrun='nov18'
#m2fsrun='jul15'
m2fsrun='may19'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2019/'

utdate=[]
file1=[]
file2=[]
#with open(directory+m2fsrun+'_fibermap_raw') as f:
#    data=f.readlines()[0:]
#for line in data:
#    p=line.split()
#    if p[0]!='none':
#        utdate.append(str(p[0]))
#        file1.append(int(p[1]))
#        file2.append(int(p[2]))
with open(directory+m2fsrun+'_twilight_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))

utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
for i in range(0,len(utdate)):
    for j in range(file1[i],file2[i]+1):
        for ccd in (['b','r']):
            out=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched.fits'
            out2=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched_iraf.fits'
            out3=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched_iraf_var.fits'
            for chip in (['c1','c2','c3','c4']):
#                print(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits')
                master_bias=astropy.nddata.CCDData.read(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits')
                master_dark=astropy.nddata.CCDData.read(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_dark.fits')
                filename=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+chip+'.fits'
                data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
                print(filename,data.header['object'],data.header['binning'])
                data_with_deviation=ccdproc.create_deviation(data,gain=data.meta['egain']*u.electron/u.adu,readnoise=data.meta['enoise']*u.electron)
                gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=5)
                oscan_subtracted=ccdproc.subtract_overscan(gain_corrected,overscan=gain_corrected[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
                trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
                trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})
                debiased0=ccdproc.subtract_bias(trimmed2,master_bias)
                dedark0=ccdproc.subtract_dark(debiased0,master_dark,exposure_time='exptime',exposure_unit=u.second,scale=True,add_keyword={'dark_corr':'Done'})
                dedark0.write(out,overwrite=True)

                if chip=='c1':
                    c1_reduce=dedark0
                if chip=='c2':
                    c2_reduce=dedark0
                if chip=='c3':
                    c3_reduce=dedark0
                if chip=='c4':
                    c4_reduce=dedark0
            left_data=np.concatenate((c1_reduce,np.flipud(c4_reduce)),axis=0)#left half of stitched image
            left_uncertainty=np.concatenate((c1_reduce.uncertainty._array,np.flipud(c4_reduce.uncertainty._array)),axis=0)
            left_mask=np.concatenate((c1_reduce.mask,np.flipud(c4_reduce.mask)),axis=0)
            right_data=np.concatenate((np.fliplr(c2_reduce),np.fliplr(np.flipud(c3_reduce))),axis=0)#right half of stitched image
            right_uncertainty=np.concatenate((np.fliplr(c2_reduce.uncertainty._array),np.fliplr(np.flipud(c3_reduce.uncertainty._array))),axis=0)
            right_mask=np.concatenate((np.fliplr(c2_reduce.mask),np.fliplr(np.flipud(c3_reduce.mask))),axis=0)

            stitched_data=np.concatenate((left_data,right_data),axis=1)
            stitched_uncertainty=np.concatenate((left_uncertainty,right_uncertainty),axis=1)
            stitched_mask=np.concatenate((left_mask,right_mask),axis=1)

            stitched=astropy.nddata.CCDData(stitched_data,unit=u.electron)
            stitched.uncertainty=stitched_uncertainty
            stitched.mask=stitched_mask
            stitched.header=c1_reduce.header
            stitched.write(out,overwrite=True)
            hdu=fits.PrimaryHDU(stitched.data,stitched.header)
            hdul=fits.HDUList([hdu])
            hdul.writeto(out2,overwrite=True)
#            hdul=fits.PrimaryHDU(stitched.data,stitched.header)
            hdul.writeto(out2,overwrite=True)
#            hdul=fits.PrimaryHDU(stitched.uncertainty.array**2,stitched.header)
            hdu=fits.PrimaryHDU(stitched.uncertainty.array**2,stitched.header)
            hdul=fits.HDUList([hdu])
            hdul.writeto(out3,overwrite=True)

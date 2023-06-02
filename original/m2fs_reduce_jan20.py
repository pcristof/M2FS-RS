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
import m2fs_process as m2fs
from astropy.nddata import StdDevUncertainty
from copy import deepcopy
matplotlib.use('TkAgg')

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='jan20' 
datadir=m2fs.get_datadir(m2fsrun)

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
field_name=[]
scifile=[]

fibermap_utdate=[]
fibermap_file=[]

with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(p[3])
        tharfile.append(p[4])
        field_name.append(p[5])
        scifile.append(p[6])

with open(directory+m2fsrun+'_fibermap_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        fibermap_utdate.append(str(p[0]))
        fibermap_file.append(str(p[1]))

utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
field_name=np.array(field_name)
scifile=np.array(scifile)

fibermap_utdate=np.array(fibermap_utdate)
fibermap_file=np.array(fibermap_file)

flatfile0=[]
tharfile0=[]
scifile0=[]
allfile0=[]
mapfile0=[]
for i in range(0,len(tharfile)):
    flatfile0.append(flatfile[i].split('-'))
    tharfile0.append(tharfile[i].split('-'))
    scifile0.append(scifile[i].split('-'))
    allfile0.append(flatfile[i].split('-')+tharfile[i].split('-')+scifile[i].split('-'))
flatfile0=np.array(flatfile0,dtype='object')
tharfile0=np.array(tharfile0,dtype='object')
scifile0=np.array(scifile0,dtype='object')
allfile0=np.array(allfile0,dtype='object')
for i in range(0,len(fibermap_file)):
    mapfile0.append(fibermap_file[i].split('-'))

for i in range(0,len(utdate)):
    for j in allfile0[i]:
        for ccd in (['b','r']):
            out=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched.fits'
            for chip in (['c1','c2','c3','c4']):

                master_bias=astropy.nddata.CCDData.read(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits')
                obs_readnoise=np.float(master_bias.header['obs_rdnoise'])
                master_dark=astropy.nddata.CCDData.read(directory+ccd+'_'+chip+'_master_dark.fits')
                filename=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+chip+'.fits'

                data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
                gain=np.float(data.header['egain'])
                print(filename,data.header['object'],data.header['binning'])

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
                exptime_ratio=np.float(data.header['exptime'])/np.float(master_dark.meta['exptime'])

                for k in range(0,len(gain_corrected2.data)):
                    for q in range(0,len(gain_corrected2.data[k])):
                        gain_corrected2.uncertainty.quantity.value[k][q]=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        poop1=(np.max(np.array([gain_corrected2.data[k][q]+master_dark.data[k][q]*gain*exptime_ratio+2.+obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2,0.6*(obs_readnoise**2+(master_dark.uncertainty.quantity.value[k][q]*gain*exptime_ratio)**2+(master_bias.uncertainty.quantity.value[k][q]*gain)**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        poop2=(np.max(np.array([gain_corrected2.data[k][q]+2.+obs_readnoise**2,0.6*(obs_readnoise**2)])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)
#                        print(poop1/poop2)
#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=10)

#                bad=np.where(gain_corrected.data<0.)
#                bad=np.where(gain_corrected._uncertainty.quantity.value!=gain_corrected._uncertainty.quantity.value)#bad variances due to negative counts after overscan/bias/dark correction
#                gain_corrected.uncertainty.quantity.value[bad]=obs_readnoise

                if chip=='c1':
                    c1_reduce=gain_corrected2
                if chip=='c2':
                    c2_reduce=gain_corrected2
                if chip=='c3':
                    c3_reduce=gain_corrected2
                if chip=='c4':
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
            stitched.write(out,overwrite=True)



for i in range(0,len(fibermap_utdate)):
    for j in mapfile0[i]:
        for ccd in (['b','r']):
            out=datadir+fibermap_utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched.fits'
            for chip in (['c1','c2','c3','c4']):

                master_bias=astropy.nddata.CCDData.read(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits')
                obs_readnoise=np.float(master_bias.header['obs_rdnoise'])
                master_dark=astropy.nddata.CCDData.read(directory+ccd+'_'+chip+'_master_dark.fits')
                filename=datadir+fibermap_utdate[i]+'/'+ccd+str(j).zfill(4)+chip+'.fits'

                data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
                print(filename,data.header['object'],data.header['binning'])

                oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
                trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
                trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})

                debiased0=ccdproc.subtract_bias(trimmed2,master_bias)
                dedark0=ccdproc.subtract_dark(debiased0,master_dark,exposure_time='exptime',exposure_unit=u.second,scale=True,add_keyword={'dark_corr':'Done'})

                data_with_deviation=ccdproc.create_deviation(dedark0,gain=data.meta['egain']*u.electron/u.adu,readnoise=obs_readnoise*u.electron)
                gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
                gain_corrected2=deepcopy(gain_corrected)

                for k in range(0,len(gain_corrected2.data)):
                    for q in range(0,len(gain_corrected2.data[k])):
                        gain_corrected2.uncertainty.quantity.value[k][q]=(np.max(np.array([gain_corrected2.data[k][q]+2.+obs_readnoise**2,0.6*obs_readnoise**2])))**0.5##rescale variances using empirically-determined fudges that hold when readnoise ~ 2.5 electrons (via S. Koposov, private comm. May 2020)

#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=10)

#                bad=np.where(gain_corrected.data<0.)
#                bad=np.where(gain_corrected._uncertainty.quantity.value!=gain_corrected._uncertainty.quantity.value)#bad variances due to negative counts after overscan/bias/dark correction
#                gain_corrected.uncertainty.quantity.value[bad]=obs_readnoise

                if chip=='c1':
                    c1_reduce=gain_corrected2
                if chip=='c2':
                    c2_reduce=gain_corrected2
                if chip=='c3':
                    c3_reduce=gain_corrected2
                if chip=='c4':
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
            stitched.write(out,overwrite=True)


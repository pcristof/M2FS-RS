import numpy as np
import astropy
from astropy import units
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
import mycode
import m2fs_process as m2fs
matplotlib.use('TkAgg')

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
#m2fsrun='[enter_run_here]'

with open(directory+'all_dark_raw') as f:
    data=f.readlines()[0:]
m2fsrun=[]
utdate=[]
file1=[]
file2=[]
for line in data:
    p=line.split()
    m2fsrun.append(str(p[0]))
    utdate.append(str(p[1]))
    file1.append(int(p[2]))
    file2.append(int(p[3]))
m2fsrun=np.array(m2fsrun)
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)

for ccd in (['b','r']):
    for chip in (['c1','c2','c3','c4']):
        debiased=[]
        exptime=[]
        master_debiased=[]
        master_exptime=[]

        for i in range(0,len(m2fsrun)):
            print(directory+m2fsrun[i]+'_'+ccd+'_'+chip+'_master_bias.fits')
            datadir=m2fs.get_datadir(m2fsrun[i])
            master_bias=astropy.nddata.CCDData.read(directory+m2fsrun[i]+'_'+ccd+'_'+chip+'_master_bias.fits')
            for j in range(file1[i],file2[i]+1):
                filename=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+chip+'.fits'
                
                data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
                print(filename,data.header['object'],data.header['binning'])

                oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
                trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
                trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})
                exptime.append(data.header['exptime'])
                master_exptime.append(data.header['exptime'])

                debiased0=ccdproc.subtract_bias(trimmed2,master_bias)
                master_debiased.append(debiased0)

        master_exptime=np.array(master_exptime)
        if np.std(master_exptime)>0.:
            print('WARNING: subexposures for darks have different exposure times!!!!')
            print(master_exptime)
            np.pause()

        c=Combiner(master_debiased)
        c.clip_extrema(nlow=1,nhigh=1)
        old_n_masked=0
        new_n_masked=c.data_arr.mask.sum()
        while (new_n_masked > old_n_masked):
            c.sigma_clipping(low_thresh=3,high_thresh=3,func=np.ma.median)
            old_n_masked=new_n_masked
            new_n_masked=c.data_arr.mask.sum()

        ccdall=c.average_combine()
        ccdall.header['exptime']=np.mean(master_exptime)
        ccdall.write(directory+ccd+'_'+chip+'_master_dark.fits',overwrite=True)

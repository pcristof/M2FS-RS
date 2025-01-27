'''This is an example script meant to reduce a specific night from A to Z for a using the functions
and objects defined in the directory.
Should be re-written in the fututre, still just for dev.
'''

def main():
    # Simple script logic
    print("Running pre_reduce.py")
    import locale
    import os
    import argparse ## To read optional arguments

    print("Checking locale")
    try:
        # Set the locale to a UTF-8 compatible setting
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        print('Setting the locale to en_US.UTF-8')
        # Fallback to a known safe locale
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    print("Locale ok")

    ## Relative imports
    from m2fs_rs import function_dump as fdump
    from m2fs_rs.class_dump import ReduceM2FS

    
    parser = argparse.ArgumentParser()
    parser.add_argument("utdate", type=str)
    parser.add_argument("ccd", type=str)

    args = parser.parse_args()
    utdate = args.utdate.lower().strip()
    ccd = args.ccd.lower().strip()

    if (ccd!='r') & (ccd!='b'):
        print('Error, usage: pre_reduce.py utdate CCD')
        print('CCD must be r or b')
        exit()
    
    currentpath = os.getcwd()+'/'

    print('------------------')
    print('Running for:')
    print('- night/target  :   {}'.format(utdate))
    print('- ccd  :   {}'.format(ccd))
    print('- PWD  :   {}'.format(currentpath))


    from configparser import ConfigParser, ExtendedInterpolation
    config_file = 'config.ini'
    if not os.path.isfile(currentpath+config_file):
        raise Exception('config.ini does not exist')
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_file)

    CORRDARK = config.getboolean('PREREDUCE', 'corrdark')
    CORRBIAS = config.getboolean('PREREDUCE', 'corrbias')
    MASTERDARK = config.getboolean('PREREDUCE', 'masterdark')
    MASTERBIAS = config.getboolean('PREREDUCE', 'masterbias')
    AUTOTRIM   = config.getboolean('PREREDUCE', 'autotrim')
    STITCH     = config.getboolean('PREREDUCE', 'stitch')

    inpath      = config['PATHS']['inpath']
    tmppath     = config['PATHS']['tmppath']
    biaspath    = config['PATHS']['biaspath']
    darkpath    = config['PATHS']['darkpath']
    frameslist  = config['PATHS']['frameslist']
    
    plugmapfile = config['ADVANCED']['plugmapfile']

    inpath = "/Users/pcristofari/Data/m2fs/data-m2fs/dupree-data/nov-2019/{}/".format(utdate) ## raw data
    tmppath = "products/{}-tmpwork/".format(utdate)
    
    if darkpath.lower().strip()=='none':
        darkpath = inpath 
    if biaspath.lower().strip()=='none':
        biaspath = inpath 
    if plugmapfile.lower().strip()=='none':
        plugmapfile = None

    ## Check tmppath exists
    if not os.path.isdir(inpath): raise Exception('Input directory not found: {}'.format(inpath))
    if not os.path.isdir(tmppath): os.mkdir(tmppath)

    ## We might need something to create the bias_raw files
    ## We need to identify which frames are bias, sience etc. I don't really see the point in having multiple
    ## files for all this.
    ## We'll figure out how to set that later in a database that is easy to read. We could also have an initial
    ## interactive script to set it up.
    ## Ok, so the thing is, the master bias and master dark in this program are constructed taking the frames of
    ## all nights specified in the input... I feel like this really should be an option. We must be able to say:
    ## here are the dark frames we want to used to create the master bias and dark. It is sort of what is done,
    ## But it was not very clear, and is too tied to the rest of the program.
    ## So, let's say that we previously do something to establish the dark frames that we want to use:
    ## The following should obviously prepared in some other way

    ## Ok, that's worse somehow. Let's do dictionaries containing the file names corresponding to ccd + color + number

    RM = ReduceM2FS()

    RM.set_utdate(utdate)
    RM.inpath = inpath
    RM.biaspath = biaspath
    RM.darkpath = darkpath
    RM.tmppath = tmppath
    RM.outpath = tmppath

    # for ccd in ['r', 'b']
    RM.set_ccd(ccd)
    RM.gen_filelists(frameslist)
    RM.check_data()

    # # # # print('ok'); exit()
    # # # # from IPython import embed
    # # # # embed()
    RM.corrdark = CORRDARK ## Do not correct for dark
    RM.corrbias = CORRBIAS ## Do not correct for dark

    ## Here I am using archive data for the dark and bias. They are already overscan corrected.
    RM.zero_corr()
    RM.dark_corr()
    import shutil
    from astropy.io import fits
    import astropy
    from ccdproc import Combiner
    from astropy import units as u
    # ## Copy the bias files
    # for tile in range(1, 5):
    #     dst = RM.masterbiasframes[(ccd, tile)] ## This is the destination file
    #     src = biasfile.format(ccd, tile)
    #     ## Here I imitate what the function do on raw file so that things are saved with the proper fits format.
    #     master_bias=astropy.nddata.CCDData.read(src,unit=u.adu)
    #     gain=np.float(master_bias.header['egain'])
    #     array1d=master_bias.data.flatten() ## flatten the data
    #     keep=np.where(np.abs(array1d)<100.)[0] ## remove crazy outliers
    #     obs_readnoise = np.std(array1d[keep]*gain) ## one value std of all values
    #     c=Combiner([master_bias])
    #     ccdall=c.average_combine()
    #     ccdall[0].header['obs_rdnoise']=str(np.median(obs_readnoise))
    #     ccdall[0].header['egain']=str(gain)
    #     ccdall.write(dst, overwrite=True)
    #     print('master_bias shape')
    #     print(master_bias.shape)
    # ## Copy the dark files
    # for tile in range(1, 5):
    #     dst = RM.masterdarkframes[(ccd, tile)] ## This is the destination file
    #     src = darkfile.format(ccd, tile)
    #     ##
    #     master_dark=astropy.nddata.CCDData.read(src,unit=u.adu)
    #     master_exptime = master_dark.header['exptime']
    #     # shutil.copyfile(src, dst)
    #     c=Combiner([master_dark])
    #     ccdall=c.average_combine()
    #     ccdall.header['exptime']=np.mean(master_exptime)
    #     ccdall.write(dst, overwrite=True)
    #     print('master_dark shape')
    #     print(master_dark.shape)

    if STITCH:
        RM.stitch_frames()

    RM.gen_plugmap(plugmapfile)

    if AUTOTRIM:
        RM.auto_trim()
    else:
        RM.trim() ## We run that for the red ccd
    RM.initialize() ## We run that for the red ccd

    print('pre_reduce.py: SUCCESS')
    exit()

if __name__ == "__main__":
    main()

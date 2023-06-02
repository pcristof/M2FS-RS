The M2FS data reduction package
Build on a the code of Matthew Walker.
Revamped by P. I. Cristofari in March/April 2023.

Overal structure:
- m2fs_process.py contains functions written by Matthew Walker
- function_dump.py contains additional functions or modified verisons of the m2fs_process.py functions
- class_dump.py contains a single class to wrap up the reduction with several methods greatly inspired from Matthew Walkers' initial reduction scripts.

A simple script can be edited to perform the full reduction using the object defined in class_dump.py. An example of such script is reduce.py.

NB: Some of the functions are interactive and will require user inputs. A program could be built upon this script, and a user config file could control the reduction options.

NB: Right now several quantities are initiliazed in the object. These should rather be placed in a configuration file. The config file file should then populate the object attributes.

Example, notebook, manual to come.

KNOWN ISSUES:
- Throughput correction and twightlight stacking not yet finished.

----- QUICK START GUIDE -----

Let us review how to perform a reduction with this pipeline.
First, we assume that you have a directory containing fits files that look like rxxxxc1.fits, or bxxxxc4.fits, etc. This is the raw data you'll get from M2FS.

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
[1] THE FRAMES.LIST FILE
The program needs to know which frame number corresponds to what. The code reads and simple text file in order to read this information. Let us name this file frames.list (the name does not matter). Here is an example of how it should be formatted:
________________
# utdate  Science-list arc-list led-list bias-list,range dark-list,range 
ut20191116 36-37-39-40    33     34-42     149,298           401,405
ut20180218 1251-1252     1248    1249     1130,1229       1126-1128,1131
________________

- Lines starting with # are not read (comments).
- Lines will be split, and the seperator is a space.
- Lines should have exactly 6 columns (the comments in the example explain what those are)
- When you want to select multiple frames, seperate them with -; example for ut20191116, we select the 36, 37, 38, 39 and 40 science frames !
- When the increment is 1, you can also you define a range with a comma; example for ut20191116, we select the 401, 402, 403, 404 and 405 dark frames !
- You can use both convention in a row; example for ut20180218 we select the 1126, 1128, 1129, 1130 and 1131 frames, but we have skipped the 1127.
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
[2] STICHING THE FRAMES
Once we have this file, we can write a script to perform the reducion. I wrote two small scripts, still being developped that make the reduction ready to use.

Here is an example of script to perform frame-stiching:
________________________
'''This is an example script meant to reduce a specific night from A to Z for a using the functions
and objects defined in the directory.
Should be re-written in the fututre, still just for dev.
'''
import os
import function_dump as fdump
import numpy as np
from class_dump import ReduceM2FS
import argparse ## To read optional arguments
#
parser = argparse.ArgumentParser()
parser.add_argument("utdate", type=str)
#
args = parser.parse_args()
utdate = args.utdate.lower().strip()
## Let's set up some paths. This should later be set in a config file.
inpath = "/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-data/{}/".format(utdate) ## raw data
tmppath = "/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-data/{}-tmpwork/".format(utdate)
## Check tmppath exists
if not os.path.isdir(inpath): raise Exception('Input directory not found: {}'.format(inpath))
if not os.path.isdir(tmppath): os.mkdir(tmppath)
#
RM = ReduceM2FS() ## Initialize the object
RM.set_utdate(utdate) ## Set set the directory name
RM.inpath = inpath ## This is ugly but I leave it for now
RM.tmppath = tmppath
RM.outpath = 'dummy'
RM.set_ccd('r') ## Choose a CCD (r or b)
RM.gen_filelists('frames.list')
RM.check_data() ## Check that the data exists and that the frames are not corrupted
RM.zero_corr()
RM.dark_corr()
RM.stitch_frames()
________________________
- This file takes an input argument, the directory name, and search for it in the inpath directory.
- The stiched frames will be saved in the tmppath directory

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
[3] PERFORM REDUCTION

Once you have stiched the frames, you will want to do everything else.
Here is an example to do that. PLEASE NOTE that the implemented functions are not all ready yet.
_________________________ 
'''This is an example script meant to reduce a specific night from A to Z for a using the functions
and objects defined in the directory.
Should be re-written in the fututre, still just for dev.
'''
import os
import function_dump as fdump
import numpy as np
from class_dump import ReduceM2FS
import argparse ## To read optional arguments
#
parser = argparse.ArgumentParser()
parser.add_argument("utdate", type=str)
parser.add_argument("ccd", type=str)
#
args = parser.parse_args()
utdate = args.utdate.lower().strip()
ccd = args.ccd.lower().strip()
#
inpath = "/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-data/{}/".format(utdate) ## raw data
tmppath = "/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-data/{}-tmpwork/".format(utdate)
outpath = "/Users/pcristofari/CfA/mgwalker_M2FS_soft/m2fs-data/{}-tmpwork2-{}-order1/".format(utdate, ccd)
## Check tmppath exists
if not os.path.isdir(inpath): raise Exception('Input directory not found: {}'.format(inpath))
if not os.path.isdir(tmppath): os.mkdir(tmppath)
if not os.path.isdir(outpath): os.mkdir(outpath)
## Prepare the reduction
RM = ReduceM2FS()
RM.set_utdate(utdate)
RM.inpath = inpath
RM.tmppath = tmppath
RM.outpath = outpath
RM.set_ccd(ccd)
RM.gen_filelists('frames.list')
RM.check_data()
## --------------------
## Generate the plugmap from fits headers
pmap = RM.gen_plugmap() # Reads filtername from headers. Throws error if unknown filter
## --------------------
RM.overwrite = False # If true, outputs of each steps will be overwritten
RM.auto_trim() ## Takes pretty much the entire image
# RM.trim() ## You can choose to interactively trim instead
RM.initialize() ## Get rough position of the center of lines in image columns
RM.find() ## Interactively find the apertures
RM.trace_all() ## Trace all the apertures
RM.apmask()
RM.apflat()
RM.apflatcorr()
RM.scatteredlightcorr()
## We are now ready for extraction
RM.extract_1d('led') ## Extract for the led frames
RM.extract_1d('arc') ## Extract for the calibration frames
RM.extract_1d('sci') ## Extract for the sciences frames
## ---------------------
RM.linelist_file = 'thar-linelist-2.txt' ## Uggly, will be somewhere else
RM.id_lines_template() ## Interactively identify lines in arc spectrum
RM.id_lines_translate() ## Automatically finds line for the other apetures.
RM.wavcal() ## Perform wavelength calib
# RM.cr_reject() ## Not used for now, deactivated by a flag in __init__
# RM.stack_twilight() ## Not working yet, ignored, but almost ready
# RM.throughputcorr() ## Not working yet, ignored, but almost ready
RM.throughputcorr_bool = False ## Uggly, will be set somewhere else
## ---------------------
RM.flat_field() 
RM.skysubtract()
RM.stack_frames()
RM.writefits()
_________________________ 
- This script takes as arguments the directory name and the ccd (r or b)
- The script will need a wavelength calibration line list, here thar-linelist-2.txt.

IMPORTANT NOTE: A great number of parameters can be tuned for this reduction. They are currently hardcoded in the class_dump.py file, in the constructor __init__. In the future they will stored in a seperate configuration file.

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


----- OLD README BY M. WALKER ------


m2fs_zero_jan20.py
m2fs_dark.py
m2fs_reduce_jan20.py
m2fs_apertures_jan20.py

Run the above in that sequence.  All of them refer to m2fs_process.py



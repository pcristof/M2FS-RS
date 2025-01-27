'''Seperate module for interactive plotting for wavelength calibration'''


import matplotlib.pyplot as plt
from astropy.io import fits
from . import function_dump as fdump
from . import m2fs_process as m2fs

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

    continuum0,spec_contsub,fit_lines=fdump.get_fitlines(extract1d,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,threshold_factor,window,continuum_rejection_order)
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
    mask = np.zeros(len(id_lines_pix), dtype=bool)
    ax1,ax2=plot_id_lines(extract1d,continuum0,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func[len(func)-1],mask,fig)
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
    print('\'f\' delete the most discrepent point\n')
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) # PIC This disables any key that was not set up.
    id_lines_tol_angs_arr = [id_lines_tol_angs]
    cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_id_lines(event,[extract1d,continuum0,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints,id_lines_tol_angs_arr,mask,fig]))
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
    import sys
    import termios
    import tty

    def flush_input():
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)

    #print('you pressed ',event.key)

    keyoptions = ['.', 'm', 'd', 'o', 'r', 't', 'y', 'z', 'g', 'l', 'q', 'p', 'i', 'f']

    global id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints
    extract1d,continuum,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma,func,rms,npoints,id_lines_tol_angs_arr,mask,fig=args_list

    global firstvalue, secondvalue

    id_lines_tol_angs = id_lines_tol_angs_arr[0]

    if event.key not in keyoptions: ## Force fitting
        event.key = 'g'

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
            flush_input()
            command=input('enter wavelength in Angstroms \n')
            command.split('\n')[-1]
            print('This is the command {}'.format(command))
            if command=='':
                print('no information entered')
            elif not command.replace('.', '').isnumeric():
                print('Please enter a float')
                command=''
            else:
                id_lines_pix.append(line_centers[best])
                id_lines_wav.append(np.float(command))
                id_lines_used.append(best)
            event.key='g'

        if event.key=='d':#delete nearest boundary point
            print('delete point nearest (',event.xdata,event.ydata,')')
            if len(id_lines_pix)>0:
                dist=(id_lines_pix-event.xdata)**2
                best=np.where(dist==np.min(dist))[0][0]
                del id_lines_pix[best]
                del id_lines_wav[best]
                del id_lines_used[best]
                event.key = 'g'
            else:
                print('no ID\'d lines to delete!')
        
        if event.key=='f':#delete the most discrepent point
            # print('delete point nearest (',event.xdata,event.ydata,')')
            if len(id_lines_pix)>0:
                ## delete everything that was masked
                func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
                mask = ~maskarr.mask
                idx = np.where(mask==False)[0]
                if len(idx)>0:
                    for ii in idx[::-1]:
                        del id_lines_pix[ii]
                        del id_lines_wav[ii]
                        del id_lines_used[ii]

                if len(id_lines_pix)>0:
                    squarediff = (id_lines_wav - func0(id_lines_pix))**2
                    ii = np.where(squarediff==np.max(squarediff))[0][0]
                    del id_lines_pix[ii]
                    del id_lines_wav[ii]
                    del id_lines_used[ii]

                event.key = 'g'
            else:
                print('no ID\'d lines to delete!')

        if event.key=='o':
            print('order of polynomial fit is ',order[len(order)-1])
            flush_input()
            command=input('enter new order (must be integer): ')
            if command=='':
                print('keeping original value')
            else:
                order.append(np.long(command))
            func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
            mask = ~maskarr.mask
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        if event.key=='r':
            print('rejection sigma is ',rejection_sigma[len(rejection_sigma)-1])
            command=''
            flush_input()
            command=input('enter new rejection sigma (float): ')
            if command=='':
                print('keeping original value')
            else:
                rejection_sigma.append(np.float(command))
            func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
            mask = ~maskarr.mask

            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        if event.key=='t':
            print('number of rejection iterations is ',rejection_iterations[len(rejection_iterations)-1])
            flush_input()
            command=input('enter new rejection iterations = (integer): ')
            if command=='':
                print('keeping original value')
            else:
                rejection_iterations.append(np.long(command))
            func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(id_lines_pix,id_lines_wav,id_lines_used,order,rejection_iterations,rejection_sigma)
            mask = ~maskarr.mask
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        if event.key=='y':
            flush_input()
            command=input('enter new tolerance level [current={}]: '.format(id_lines_tol_angs_arr[0]))
            if command=='':
                print('keeping original value')
            else:
                newvalue = float(command)
                id_lines_tol_angs_arr[0] = newvalue

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
            func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
            mask = ~maskarr.mask
            new_pix,new_wav,new_used=m2fs.line_id_add_lines(deepcopy(linelist),deepcopy(line_centers),deepcopy(id_lines_used),deepcopy(func0),id_lines_tol_angs)

            for i in range(0,len(new_pix)):
                id_lines_pix.append(new_pix[i])
                id_lines_wav.append(new_wav[i])
                id_lines_used.append(new_used[i])
            func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
            mask = ~maskarr.mask
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)
        if event.key=='q':
            plt.close(event.canvas.figure)
            # return

        ## Delete anypoint that would be outside of the xlims
        xlim=[np.min(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),np.max(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])]

        for ii in range(len(id_lines_pix)-1, -1, -1):
            if (id_lines_pix[ii]<xlim[0]) | (id_lines_pix[ii]>xlim[1]):
                del id_lines_pix[ii]
                del id_lines_wav[ii]
                del id_lines_used[ii]

        event.key='g' ## Force plotting always to get mask
        if event.key=='g':

            func0,rms0,npoints0,maskarr=m2fs.id_lines_fit(deepcopy(id_lines_pix),deepcopy(id_lines_wav),deepcopy(id_lines_used),order,rejection_iterations,rejection_sigma)
            mask = ~maskarr.mask
            func.append(func0)
            rms.append(rms0)
            npoints.append(npoints0)

        else:
            pass
    
    ax1,ax2=plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func[len(func)-1],mask,fig)
    fig.canvas.draw_idle()
    return

def plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,id_lines_used,func,mask,fig):
    import numpy as np
    from astropy import units as u

    fig.clf() ## PIC test - that appears to correct the weird plotting issue.

    xlim=[np.min(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),np.max(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])]
    ax1=fig.add_subplot(311)
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
         _color='g'
         if len(mask)>j:
            if mask[j]==False:
                _color='r'             
         ax1.axvline(x=id_lines_pix[j],color=_color,lw=1,linestyle='-')
#    ax1.text(0,1.01,'n_points='+str(npoints),transform=ax1.transAxes)
#    ax1.text(0,1.06,'order='+str(order),transform=ax1.transAxes)
#    ax1.text(0,1.11,'rms='+str(rms),transform=ax1.transAxes)
#    ax1.text(1,1.01,'aperture='+str(j+1),horizontalalignment='right',transform=ax1.transAxes)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')

    ax2=fig.add_subplot(312)
    ax2.cla()
#    ax2.scatter(id_lines_pix,id_lines_wav,color='k',s=3)
    # ax2.scatter(id_lines_pix,id_lines_wav-func(id_lines_pix),color='k',s=3)
    # ax2.scatter(np.array(id_lines_pix)[~mask],np.array(id_lines_wav-func(id_lines_pix))[~mask],color='k',s=10, marker='d')
    if len(mask)==len(np.array(id_lines_pix)):
        ax2.scatter(np.array(id_lines_pix)[mask],np.array(id_lines_wav-func(id_lines_pix))[mask],color='k',s=10)

    x=extract1d.spec1d_pixel[extract1d.spec1d_mask==False]
#    ax2.plot(x,func(x),color='r')
    ax2.set_xlim(xlim)
    ax2.set_xlabel('pixel')
    ax2.set_ylabel(r'$\lambda$ [Angstroms]')


    ax3=fig.add_subplot(313)
    ax3.cla()
#    ax3.scatter(id_lines_pix,id_lines_wav,color='k',s=3)
#   ## Plot ALL points positions
    ## The current fit of the points
    # for j in range(0,len(fit_lines.fit)):
    #     ax3.plot(fit_lines.fit[j].mean.value, func(fit_lines.fit[j].mean.value), 'x',color='k',lw=0.3) 
    if len(id_lines_pix)>0:
        ax3.plot(np.linspace(np.min(id_lines_pix), np.max(id_lines_pix), 1000), func(np.linspace(np.min(id_lines_pix), np.max(id_lines_pix), 1000)), color='red')
        ## Plot the mask used:
        # ax3.scatter(id_lines_pix,id_lines_wav,color='k',s=3)
        ax3.scatter(np.array(id_lines_pix),np.array(id_lines_wav),color='red',s=50)
        ax3.scatter(np.array(id_lines_pix)[mask],np.array(id_lines_wav)[mask],color='green',s=50)
        x=extract1d.spec1d_pixel[extract1d.spec1d_mask==False]
#    ax3.plot(x,func(x),color='r')
    ax3.set_xlim(xlim)
    ax3.set_xlabel('pixel')
    ax3.set_ylabel(r'$\lambda$ [Angstroms]')


    return ax1,ax2

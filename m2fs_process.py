def plot_trace(j,trace_x,trace_y,trace_z,y,trace_func,pix_min,pix_max,rms,npoints,order,fig):
    ax1=fig.add_subplot(211)
    ax1.cla()
    ax1.scatter(trace_x,trace_y,s=3,color='k')
    ax1.scatter(trace_x[y.mask==True],trace_y[y.mask==True],s=13,marker='x',color='y')
    ax1.axvline(x=pix_min,color='k',lw=0.3,linestyle=':')
    ax1.axvline(x=pix_max,color='k',lw=0.3,linestyle=':')
    ax1.set_xlabel('trace x [pixel]')
    ax1.set_ylabel('trace y [pixel]')
    ax1.plot(trace_x,trace_func(trace_x),color='r')
    ax1.text(0,1.01,'n_points='+str(npoints),transform=ax1.transAxes)
    ax1.text(0,1.06,'order='+str(order),transform=ax1.transAxes)
    ax1.text(0,1.11,'rms='+str(rms),transform=ax1.transAxes)
    ax1.text(1,1.01,'aperture='+str(j+1),horizontalalignment='right',transform=ax1.transAxes)

    ax2=fig.add_subplot(212)
    ax2.cla()
    ax2.scatter(trace_x,trace_z,s=3,color='k')
    ax2.scatter(trace_x[y.mask==True],trace_z[y.mask==True],s=13,marker='x',color='y')
    ax2.axvline(x=pix_min,color='k',lw=0.3,linestyle=':')
    ax2.axvline(x=pix_max,color='k',lw=0.3,linestyle=':')
    ax2.set_xlabel('trace x [pixel]')
    ax2.set_ylabel('counts')
    return ax1,ax2

def plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,fig):
    import numpy as np
    from astropy import units as u

    ax1=fig.add_subplot(111)
    ax1.cla()
    ax1.set_xlim([np.min(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),np.max(extract1d.spec1d_pixel[extract1d.spec1d_mask==False])])
    ax1.plot(extract1d.spec1d_pixel[extract1d.spec1d_mask==False],extract1d.spec1d_flux[extract1d.spec1d_mask==False],color='k',lw=0.5)
    ax1.plot(extract1d.spec1d_pixel[extract1d.spec1d_mask==False],continuum(extract1d.spec1d_pixel[extract1d.spec1d_mask==False]),color='b',lw=0.3)
    ax1.set_xlabel('pixel')
    ax1.set_ylabel('counts')
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

    return ax1

def on_key_boundary(event,args_list):
    import numpy as np
    import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils import Spectrum1D

    global lower_x,lower_y,upper_x,upper_y
    data,lower_x,lower_y,upper_x,upper_y,fig=args_list

    if event.key=='l':#add lower boundary point
        print('add boundary point at (',event.xdata,event.ydata,')')
        lower_x.append(event.xdata)
        lower_y.append(event.ydata)

    if event.key=='u':#add upper boundary point
        print('add boundary point at (',event.xdata,event.ydata,')')
        upper_x.append(event.xdata)
        upper_y.append(event.ydata)

    if event.key=='d':#delete nearest boundary point
        print('delete point nearest (',event.xdata,event.ydata,')')
        x0=event.xdata
        y0=event.ydata
        if len(lower_x)>0:
            dist_lower=(lower_x-x0)**2+(lower_y-y0)**2
            min1=np.min(dist_lower)
            best1=np.where(dist_lower==min1)[0][0]
        else:
            min1=1.e+30
        if len(upper_x)>0:
            dist_upper=(upper_x-x0)**2+(upper_y-y0)**2
            min2=np.min(dist_upper)
            best2=np.where(dist_upper==min2)[0][0]
        else:
            min2=1.e+30
        if min1<=min2:
            if len(lower_x)>0:
                del lower_x[best1]
                del lower_y[best1]
        else:
            if len(upper_x)>0:
                del upper_x[best2]
                del upper_y[best2]

    ax1=fig.add_subplot(111)
    ax1.cla()
    vmin=np.quantile(data.data.flatten(),0.05)
    vmax=np.quantile(data.data.flatten(),0.95)
    ax1.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)

    def plotboundary(x0,y0):
        x=x0
        y=y0
        x=np.array(x)
        y=np.array(y)
        order=np.argsort(y)
        x=x[order]
        y=y[order]
        x=np.append(x[0],x)
        y=np.append(0,y)
        x=np.append(x,x[len(x)-1])
        y=np.append(y,len(data.data))
        ax1.scatter(x,y,color='y',s=10,alpha=0.2)
        ax1.plot(x,y,color='y',alpha=0.2)
    
    if len(lower_x)>0:
        plotboundary(lower_x,lower_y)
    if len(upper_x)>0:
        plotboundary(upper_x,upper_y)

    fig.canvas.draw_idle()
    return

def on_key_find(event,args_list):
    import numpy as np
    import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils import Spectrum1D

    print('you pressed ',event.key)

    global columnspec_array,subregion,fit,realvirtual,initial,window
    columnspec_array,column,subregion,fit,realvirtual,initial,window,fig=args_list

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
        aperture_centers=[fit[q].mean for q in range(0,len(fit))]
        x0=event.xdata
        y0=event.ydata
        dist=(aperture_centers-event.xdata)**2
        best=np.where(dist==np.min(dist))[0][0]
        print('deleting aperture '+str(best+1)+', centered at '+str(aperture_centers[best]))
        del subregion[best]
        del fit[best]
        del realvirtual[best]
        del initial[best]
        subregion,fit,realvirtual,initial=m2fs.aperture_order(subregion,fit,realvirtual,initial)

    if event.key=='n':
        new_center=np.float(event.xdata)
        x_center=new_center
        spec1d=Spectrum1D(spectral_axis=columnspec_array[column].pixel,flux=columnspec_array[column].spec*u.electron,uncertainty=columnspec_array[column].err,mask=columnspec_array[column].mask)
        subregion0,fit0=fit_aperture(spec1d-columnspec_array[column].continuum(columnspec_array[column].pixel.value),window,x_center)
        subregion.append(subregion0)
        fit.append(fit0)
        realvirtual.append(True)
        initial.append(False)
        subregion,fit,realvirtual,initial=aperture_order(subregion,fit,realvirtual,initial)

    if event.key=='a':
        new_center=np.float(event.xdata)
        x_center=new_center
        val1=x_center-window/2.
        val2=x_center+window/2.
        subregion.append(SpectralRegion(val1*u.AA,val2*u.AA))#define extraction region from window
        aaa=np.float(np.max(columnspec_array[column].spec-columnspec_array[column].continuum(columnspec_array[column].pixel.value)))
        halfwindow=window/2.
        fit.append(models.Gaussian1D(amplitude=aaa*u.electron,mean=x_center*u.AA,stddev=halfwindow*u.AA))
        realvirtual.append(False)
        initial.append(False)
        subregion,fit,realvirtual,initial=aperture_order(subregion,fit,realvirtual,initial)

    x=np.arange(0,len(columnspec_array[column].spec))
    ax1=fig.add_subplot(111)
    ax1.cla()
    ax1.plot(columnspec_array[column].pixel,columnspec_array[column].spec,lw=0.3,color='k')
    ax1.plot(columnspec_array[column].pixel,columnspec_array[column].continuum(x),color='green',lw=1)
    ax1.set_xlim([0,np.max(x)])
    for j in range(0,len(fit)):
        use=np.where((columnspec_array[column].pixel.value>=subregion[j].lower.value)&(columnspec_array[column].pixel.value<=subregion[j].upper.value))[0]
        sub_spectrum_pixel=columnspec_array[column].pixel[use]
        y_fit=fit[j](sub_spectrum_pixel).value+columnspec_array[column].continuum(sub_spectrum_pixel.value)

        ax1.axvline(x=fit[j].mean.value,color='b',lw=0.3,alpha=0.7)
        ax1.plot(sub_spectrum_pixel.value,y_fit,color='r',lw=0.2,alpha=0.7)
        ax1.text(fit[j].mean.value,0,str(j+1),fontsize=8)
        if not realvirtual[j]:
            ax1.axvline(x=fit[j].mean.value,color='k',lw=0.5,alpha=1,linestyle='--')
    fig.canvas.draw_idle()
    return

def on_key_trace(event,args_list):
    import numpy as np
    import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils import Spectrum1D
    from astropy.modeling import models,fitting
    print('you pressed ',event.key)

    global trace_x,trace_y,trace_z,y,sig_y,pix_min,pix_max,func,rms,npoints,order,rejection_sigma,trace_rejection_iterations,trace_func_init
    j,trace_x,trace_y,trace_z,y,sig_y,pix_min,pix_max,func,rms,npoints,order,rejection_sigma,trace_rejection_iterations,trace_func_init,fig=args_list

    if event.key=='d':
        x0=event.xdata
        y0=event.ydata
        candidates=np.where(y.mask==False)[0]
        dist=((trace_x[candidates]-event.xdata)**2+(trace_y[candidates]-event.ydata)**2)
        best=np.where(dist==np.min(dist))[0][0]
        print('deleting point centered at ('+str(trace_x[candidates[best]])+','+str(trace_y[candidates[best]])+')')
        y.mask[candidates[best]]=True

    if event.key=='c':
        x0=event.xdata
        y0=event.ydata
        candidates=np.where(y.mask==True)[0]
        dist=((trace_x[candidates]-event.xdata)**2+(trace_y[candidates]-event.ydata)**2)
        best=np.where(dist==np.min(dist))[0][0]
        print('re-instating point '+str(best+1)+', centered at ('+str(trace_x[best])+','+str(trace_y[best])+')')
        y.mask[candidates[best]]=False

    if event.key=='l':
        pix_min.append(event.xdata)
    if event.key=='u':
        pix_max.append(event.xdata)
    if event.key=='o':
        print('order of polynomial fit is ',order[len(order)-1])
        command=input('enter new order (must be integer): ')
        order.append(np.long(command))
    if event.key=='r':
        print('rejection sigma is ',rejection_sigma[len(rejection_sigma)-1])
        command=input('enter new rejection sigma (float): ')
        rejection_sigma.append(np.float(command))
    if event.key=='z':
        for q in range(1,len(pix_min)):
            del(pix_min[len(pix_min)-1])
        for q in range(1,len(pix_max)):
            del(pix_max[len(pix_max)-1])
        for q in range(1,len(order)):
            del(order[len(order)-1])
        for q in range(1,len(rejection_sigma)):
            del(rejection_sigma[len(rejection_sigma)-1])
        for q in range(0,len(y)):
            y.mask[q]=False

    y.mask[trace_x<pix_min[len(pix_min)-1]]=True
    y.mask[trace_x>pix_max[len(pix_max)-1]]=True
    fitter=fitting.LinearLSQFitter()
    for q in range(0,trace_rejection_iterations):
        use=np.where(y.mask==False)[0]
        trace_func=fitter(trace_func_init,trace_x[use],trace_y[use])
        trace_rms=np.sqrt(np.mean((y[use]-trace_func(trace_x)[use])**2))
        outlier=np.where(np.abs(trace_y-trace_func(trace_x))>rejection_sigma[len(rejection_sigma)-1]*trace_rms)[0]
        y.mask[outlier]=True

    use=np.where(y.mask==False)[0]
    trace_npoints=len(use)

    func.append(trace_func)
    rms.append(trace_rms)
    npoints.append(trace_npoints)
    print('fitting range: ',pix_min[len(pix_min)-1],pix_max[len(pix_max)-1])
    print('order of polynomial fit is ',order[len(order)-1])
    print('number of points in fit=',trace_npoints)
    print('rms='+str(trace_rms))
    print('')
    ax1,ax2=plot_trace(j,trace_x,trace_y,trace_z,y,trace_func,pix_min[len(pix_min)-1],pix_max[len(pix_max)-1],trace_rms,trace_npoints,order[len(order)-1],fig)
    fig.canvas.draw_idle()
    return

def on_key_id_lines(event,args_list):
    import numpy as np
    import m2fs_process as m2fs
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models
    from specutils import Spectrum1D
    from astropy.modeling import models,fitting
    print('you pressed ',event.key)

    global id_lines_pix,id_lines_wav,order,rejection_iterations,rejection_sigma
    extract1d,continuum,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,order,rejection_iterations,rejection_sigma,fig=args_list

    if event.key=='m':
        x0=event.xdata
        y0=event.ydata
        dist=(line_centers-event.xdata)**2
        best=np.where(dist==np.min(dist))[0][0]
        print('pixel = ',str(line_centers[best].value))
        command=input('enter wavelength in Angstroms \n')
        if command=='':
            print('no information entered')
        else:
            id_lines_pix.append(line_centers[best].value)
            id_lines_wav.append(np.float(command))
    if event.key=='d':#delete nearest boundary point
        print('delete point nearest (',event.xdata,event.ydata,')')
        if len(id_lines_pix)>0:
            dist=(id_lines_pix-event.xdata)**2
            best=np.where(dist==np.min(dist))[0][0]
            del id_lines_pix[best]
            del id_lines_wav[best]
        else:
            print('no ID\'d lines to delete!')
    if event.key=='o':
        print('order of polynomial fit is ',order[len(order)-1])
        command=input('enter new order (must be integer): ')
        if command=='':
            print('keeping original value')
        else:
            order.append(np.long(command))
    if event.key=='r':
        print('rejection sigma is ',rejection_sigma[len(rejection_sigma)-1])
        command=input('enter new rejection sigma (float): ')
        if command=='':
            print('keeping original value')
        else:
            rejection_sigma.append(np.float(command))
    if event.key=='t':
        print('number of rejection iterations is ',rejection_iterations[len(rejection_iterations)-1])
        command=input('enter new rejection iterations = (integer): ')
        if command=='':
            print('keeping original value')
        else:
            rejection_iterations.append(np.long(command))
    if event.key=='z':
        for q in range(1,len(id_lines_pix)):
            del(id_lines_pix[len(id_lines_pix)-1])
        for q in range(1,len(id_lines_wav)):
            del(id_lines_wav[len(id_lines_wav)-1])
        for q in range(1,len(order)):
            del(order[len(order)-1])
        for q in range(1,len(rejection_iterations)):
            del(rejection_iterations[len(rejection_iterations)-1])
        for q in range(1,len(rejection_sigma)):
            del(rejection_sigma[len(rejection_sigma)-1])
    if event.key=='l':
        print('inserting from line list')
        x=np.array(id_lines_pix)
        y=np.ma(id_lines_wav)
        y.mask[:]=False
        func_init=models.Polynomial1D(degree=order[len(order)-1])
        fitter=fitting.LinearLSQFitter()
        for q in range(0,rejection_iterations):
            use=np.where(y.mask==False)[0]
            func0=fitter(func_init,x[use],y[use])
            rms0=np.sqrt(np.mean((y[use]-func(x)[use])**2))
            outlier=np.where(np.abs(y-func0(x))>rejection_sigma[len(rejection_sigma)-1]*rms0)[0]
            y.mask[outlier]=True
            
        use=np.where(y.mask==False)[0]
        npoints=len(use)
        
        func.append(func0)
        rms.append(rms0)
        npoints.append(npoints0)
        print('fitting range: ',pix_min[len(pix_min)-1],pix_max[len(pix_max)-1])
        print('order of polynomial fit is ',order[len(order)-1])
        print('number of points in fit=',trace_npoints)
        print('rms='+str(trace_rms))
        print('')

    ax1=plot_id_lines(extract1d,continuum,fit_lines,line_centers,id_lines_pix,id_lines_wav,fig)
    fig.canvas.draw_idle()
    return

def on_click(event):#(event.xdata,event.ydata) give the user coordinates, event.x and event.y are internal coordinates
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single',event.button,
           event.x,event.y,event.xdata,event.ydata))
    click.append((event.xdata,event.ydata))

def get_continuum(spec1d,continuum_rejection_iterations):
    import numpy as np
    import astropy.units as u
    from astropy.modeling import models,fitting
    from copy import deepcopy
    import copy

    continuum_init=models.Polynomial1D(degree=10)
    fitter=fitting.LinearLSQFitter()
    lamb=(np.arange(len(spec1d.data),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
    for q in range(0,continuum_rejection_iterations):
#        y=np.ma.masked_array(deepcopy(spec1d.data),mask=deepcopy(spec1d.mask))
        y=np.ma.masked_array(spec1d.data,mask=spec1d.mask)
        continuum=fitter(continuum_init,lamb.value,y)
        outlier=(np.where(spec1d.data-continuum(lamb.value)>1.*spec1d.uncertainty.array))[0]#reject only high outliers (emission lines)
        print('rejected '+str(len(outlier))+' from continuum fit in iteration '+str(q+1))
        y.mask[outlier]=True
    rms=np.sqrt(np.mean((y.data[y.mask==False]-continuum(lamb.value)[y.mask==False])**2))
    return continuum,rms

def column_stack(data,col):
    import numpy as np
    import mycode
    from ccdproc import Combiner
    import astropy.units as u
    from specutils import Spectrum1D

    column=data.data[:,col]
    column_uncertainty=data.uncertainty[:,col]
    column_mask=data.mask[:,col]

    stack=[]
    sig_stack=[]
    for j in range(0,len(column[0])):
        stack.append(data[:,col[j]])
        sig_stack.append(data.uncertainty._array[:,col[j]])
    sig_stack=np.array(sig_stack)
    
    c=Combiner(stack)
    c.weights=1./sig_stack**2
    comb=c.average_combine(uncertainty_func=mycode.stdmean)
    lamb=(np.arange(len(comb.data),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
    spec1d=Spectrum1D(spectral_axis=lamb,flux=comb.data*u.electron,uncertainty=comb.uncertainty,mask=comb.mask)
    return spec1d

def plot_apertures(pixel,spec,continuum,subregion,fit,realvirtual):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from specutils.manipulation import extract_region
    from astropy import units as u

    gs=plt.GridSpec(10,10) # define multi-panel plot
    gs.update(wspace=0,hspace=0) # specify inter-panel spacing
#    fig=plt.figure(figsize=(6,6)) # define plot size
    ax1=fig.add_subplot(111)

    x=np.arange(0,len(spec))
    ax1.plot(pixel,spec,lw=0.3,color='k')
    ax1.plot(pixel,continuum(x),color='green',lw=1)
    ax1.set_xlim([0,np.max(x)])

    for j in range(0,len(fit)):
        use=np.where((pixel.value>=subregion[j].lower.value)&(pixel.value<=subregion[j].upper.value))[0]
        sub_spectrum_pixel=pixel[use]
        y_fit=fit[j](sub_spectrum_pixel).value+continuum(sub_spectrum_pixel.value)

        ax1.axvline(x=fit[j].mean.value,color='b',lw=0.3,alpha=0.7)
        ax1.plot(sub_spectrum_pixel.value,y_fit,color='r',lw=0.2,alpha=0.7)
        ax1.text(fit[j].mean.value,0,str(j+1),fontsize=8)
        if not realvirtual[j]:
            ax1.axvline(x=fit[j].mean.value,color='k',lw=0.5,alpha=1,linestyle='--')
    return ax1

def fit_aperture(spec,window,x_center):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from specutils.manipulation import extract_region
    from specutils import SpectralRegion
    from specutils.manipulation import extract_region
    import astropy.units as u
    from specutils.fitting import estimate_line_parameters
    from astropy.modeling import models,fitting
    from specutils.fitting import fit_lines

    val1=x_center-window/2.#window for fitting aperture
    val2=x_center+window/2.#window for fitting aperture
    subregion0=SpectralRegion(val1*u.AA,val2*u.AA)#define extraction region from window
    sub_spectrum=extract_region(spec,subregion0)#extract from window
    rough=estimate_line_parameters(sub_spectrum,models.Gaussian1D())#get rough estimate of gaussian parameters for aperture
    g_init=models.Gaussian1D(amplitude=rough.amplitude.value*u.electron,mean=rough.mean.value*u.AA,stddev=rough.stddev.value*u.AA)#now do a fit using rough estimate as first guess
    g_fit0=fit_lines(sub_spectrum,g_init)#now do a fit using rough estimate as first guess
    return subregion0,g_fit0

def aperture_order(subregion,g_fit,realvirtual,initial):#sort apertures in order of increasing column center
    import numpy as np

    means=[g_fit[i].mean.value for i in range(0,len(g_fit))]
    order0=np.argsort(means)
    g_fit2=[g_fit[order0[i]] for i in range(0,len(g_fit))]
    g_fit=g_fit2
    subregion2=[subregion[order0[i]] for i in range(0,len(g_fit))]
    subregion=subregion2
    realvirtual2=[realvirtual[order0[i]] for i in range(0,len(g_fit))]
    realvirtual=realvirtual2
    initial2=[initial[order0[i]] for i in range(0,len(g_fit))]
    initial=initial2
    return subregion,g_fit,realvirtual,initial

def get_trace_xyz(j,columnspec_array,apertures_profile_middle,middle_column,trace_shift_max,trace_nlost_max):
    import numpy as np
    mean0=apertures_profile_middle.fit[j].mean.value
    trace_x=[]
    trace_y=[]
    trace_z=[]
    nlost=0
    for k in reversed(range(0,middle_column)):
        if nlost<=trace_nlost_max:
            means=np.array([columnspec_array[k].apertures_profile.fit[q].mean.value for q in range(0,len(columnspec_array[k].apertures_profile.fit))])
            means=means[means==means]#remove any NaNs due to poor fits
            amplitudes=np.array([columnspec_array[k].apertures_profile.fit[q].amplitude.value for q in range(0,len(columnspec_array[k].apertures_profile.fit))])
            amplitudes=amplitudes[means==means]
            dist=(mean0-means)**2
            best=np.where(dist==np.min(dist))[0][0]
            if dist[best]<trace_shift_max:
                nlost=0
                mean0=means[best]
                amplitude0=amplitudes[best]
                trace_x.append(np.median(columnspec_array[k].columns))
                trace_y.append(means[best])
                trace_z.append(amplitudes[best])
            else:
                nlost=nlost+1
    nlost=0
    mean0=apertures_profile_middle.fit[j].mean.value
    for k in range(middle_column,len(columnspec_array)):
        if nlost<trace_nlost_max:
            means=np.array([columnspec_array[k].apertures_profile.fit[q].mean.value for q in range(0,len(columnspec_array[k].apertures_profile.fit))])
            keep=np.where(means==means)[0]
            means=means[keep]#remove any NaNs due to poor fits
            amplitudes=np.array([columnspec_array[k].apertures_profile.fit[q].amplitude.value for q in range(0,len(columnspec_array[k].apertures_profile.fit))])
            amplitudes=amplitudes[keep]
            dist=(mean0-means)**2
            best=np.where(dist==np.min(dist))[0][0]
            if dist[best]<trace_shift_max:
                nlost=0
                mean0=means[best]
                amplitude0=amplitudes[best]
                trace_x.append(np.median(columnspec_array[k].columns))
                trace_y.append(means[best])
                trace_z.append(amplitudes[best])
            else:
                nlost=nlost+1
    trace_x=np.array(trace_x)
    trace_y=np.array(trace_y)
    trace_z=np.array(trace_z)
    order=np.argsort(trace_x)
    trace_x=trace_x[order]
    trace_y=trace_y[order]
    trace_z=trace_z[order]
    return trace_x,trace_y,trace_z

def get_aperture(j,columnspec_array,apertures_profile_middle,middle_column,trace_order,trace_rejection_sigma,trace_rejection_iterations,image_boundary,trace_shift_max,trace_nlost_max,lsf_rejection_iterations,lsf_nsample,lsf_order,window):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models,fitting
    from astropy.modeling import models
    from specutils import Spectrum1D

    trace_x,trace_y,trace_z=get_trace_xyz(j,columnspec_array,apertures_profile_middle,middle_column,trace_shift_max,trace_nlost_max)
    trace_func_init=models.Polynomial1D(degree=trace_order)
    fitter=fitting.LinearLSQFitter()
    y=np.ma.masked_array(trace_y,mask=False)
    z=np.ma.masked_array(trace_z,mask=False)
    sig_y=trace_y#fudge for now, not used for anything

    trace_pix_min=[]
    trace_pix_max=[]
    for i in range(0,len(trace_x)):
        trace_pix_min.append(np.interp(x=trace_y[i],xp=[image_boundary.lower[q][1] for q in range(0,len(image_boundary.lower))],fp=[image_boundary.lower[q][0] for q in range(0,len(image_boundary.lower))]))
        trace_pix_max.append(np.interp(x=trace_y[i],xp=[image_boundary.upper[q][1] for q in range(0,len(image_boundary.upper))],fp=[image_boundary.upper[q][0] for q in range(0,len(image_boundary.upper))]))
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
        pix_min=[np.float(-999.)]
        pix_max=[np.float(-999.)]


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
        pix_min.append(np.float(-999.))
        pix_max.append(np.float(-999.))
    lsf_init=models.Polynomial1D(degree=lsf_order)
    lsf_fitter=fitting.LinearLSQFitter()

    x=np.array([np.median(columnspec_array[k].columns) for k in range(0,len(columnspec_array))])
    if apertures_profile_middle.realvirtual[j]:#if this is a real aperture, fit its LSF as function of pixel
        y0=[]
        for i in range(0,len(x)):#re-fit Gaussian profile at different x values along spectral axis
            center=func[len(func)-1](x[i])
            if((center>0.)&(center<np.max(columnspec_array[i].pixel.value))):#make sure the trace function y(x) makes sense at this x
                spec1d=Spectrum1D(spectral_axis=columnspec_array[i].pixel,flux=columnspec_array[i].spec*u.electron,uncertainty=columnspec_array[i].err,mask=columnspec_array[i].mask)
                subregion0,g_fit0=fit_aperture(spec1d-columnspec_array[i].continuum(columnspec_array[i].pixel.value),window,center)
                y0.append(g_fit0.stddev.value)
            else:#otherwise give a place-holder value and mask it below
                y0.append(np.float(-999.))
        y0=np.array(y0)
        y=np.ma.masked_array(y0,mask=False)
        y.mask[y0<-998.]=True
        y.mask[x<pix_min[len(pix_min)-1]]=True
        y.mask[x>pix_max[len(pix_max)-1]]=True

        lsf_sigma=lsf_init
        lsf_rms=np.sqrt(np.mean((y[use]-lsf_sigma(x[use]))**2))
        for q in range(0,lsf_rejection_iterations):
            use=np.where(y.mask==False)[0]
            if len(use)>3.:
                lsf_sigma=lsf_fitter(lsf_init,x[use],y[use])
                lsf_rms=np.sqrt(np.mean((y[use]-lsf_sigma(x[use]))**2))
                outlier=(np.where(np.abs(y-lsf_sigma(x))>3.*lsf_rms))[0]#reject outliers
                y.mask[outlier]=True
        use=np.where(y.mask==False)[0]
        lsf_npoints=len(use)
#        plt.scatter(x,y0,s=3,color='k')
#        plt.scatter(x[y.mask==True],y0[y.mask==True],s=13,marker='x',color='y')
#        plt.plot(x,lsf_sigma(x),color='r')
#        plt.axvline(x=pix_min[len(pix_min)-1],lw=0.3,linestyle=':',color='k')
#        plt.axvline(x=pix_max[len(pix_max)-1],lw=0.3,linestyle=':',color='k')
#        plt.ylim([1,3])
#        plt.xlabel('x [pixel]')
#        plt.ylabel('LSF sigma [pixel]')
#        plt.show()
#        plt.close()
    else:
        lsf_sigma=-999
        lsf_rms=-999
        lsf_npoints=-999
    aperture0=aperture(trace_aperture=j+1,trace_func=func[len(func)-1],trace_rms=rms[len(rms)-1],trace_npoints=npoints[len(npoints)-1],trace_pixel_min=pix_min[len(pix_min)-1],trace_pixel_max=pix_max[len(pix_max)-1],lsf_sigma=lsf_sigma,lsf_rms=lsf_rms,lsf_npoints=lsf_npoints)
    return aperture0

class image_boundary:
    def __init__(self,lower=None,upper=None):
        self.lower=lower
        self.upper=upper

class linelist:
    def __init__(self,wavelength=None,species=None):
        self.wavelength=wavelength
        self.species=species

class aperture_profile:
    def __init__(self,subregion=None,fit=None,realvirtual=None,initial=None):
        self.subregion=subregion
        self.fit=fit
        self.realvirtual=realvirtual
        self.initial=initial

class aperture:
    def __init__(self,trace_aperture=None,trace_func=None,trace_rms=None,trace_npoints=None,trace_pixel_min=None,trace_pixel_max=None,lsf_sigma=None,lsf_rms=None,lsf_npoints=None):
        self.trace_aperture=trace_aperture
        self.trace_func=trace_func
        self.trace_rms=trace_rms
        self.trace_npoints=trace_npoints
        self.trace_pixel_min=trace_pixel_min
        self.trace_pixel_max=trace_pixel_max
        self.lsf_sigma=lsf_sigma
        self.lsf_rms=lsf_rms
        self.lsf_npoints=lsf_npoints

def get_aperture_profile(apertures_initial,spec1d,continuum,window):
    import numpy as np

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
    return aperture_profile(subregion=subregion,fit=g_fit,realvirtual=realvirtual,initial=initial)

class columnspec:
    def __init__(self,columns=None,spec=None,mask=None,err=None,pixel=None,continuum=None,rms=None,apertures_initial=None,apertures_profile=None):
        self.columns=columns
        self.spec=spec
        self.mask=mask
        self.err=err
        self.pixel=pixel
        self.continuum=continuum
        self.rms=rms
        self.apertures_initial=apertures_initial
        self.apertures_profile=apertures_profile

class extract1d:
    def __init__(self,aperture=None,spec1d_pixel=None,spec1d_flux=None,spec1d_uncertainty=None,spec1d_mask=None):
        self.aperture=aperture
        self.spec1d_pixel=spec1d_pixel
        self.spec1d_flux=spec1d_flux
        self.spec1d_uncertainty=spec1d_uncertainty
        self.spec1d_mask=spec1d_mask

def get_columnspec(data,trace_step,n_lines,continuum_rejection_iterations,threshold_factor,window):
    import numpy as np
    import astropy.units as u
    from specutils.fitting import find_lines_threshold
    from specutils.fitting import find_lines_derivative

    n_cols=np.shape(data)[1]
    trace_n=n_cols/trace_step
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
        print('working on '+str(i+1)+' of '+str(len(trace_cols))+' trace columns')
        col0=np.arange(n_lines)+trace_cols[i]
        spec1d0=column_stack(data,col0)
        continuum0,rms0=get_continuum(spec1d0,continuum_rejection_iterations)
        pixel0=(np.arange(len(spec1d0.data),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
        spec_contsub=spec1d0-continuum0(pixel0.value)
        spec_contsub.uncertainty.quantity.value[:]=rms0
        spec_contsub.mask[:]=False
        apertures_initial0=find_lines_derivative(spec_contsub,flux_threshold=threshold_factor*rms0)#find peaks in continuum-subtracted "spectrum"
#        apertures_profile0=get_aperture_profile(spec1d0,window)

        apertures_profile0=get_aperture_profile(apertures_initial0,spec1d0,continuum0,window)
        columnspec0=columnspec(columns=col0,spec=spec1d0.data,mask=spec1d0.mask,err=spec1d0.uncertainty,pixel=pixel0,continuum=continuum0,rms=rms0,apertures_initial=apertures_initial0,apertures_profile=apertures_profile0)#have to break up spec1d into individual data/mask/uncertainty fields in order to allow pickling of columnspec_array
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
        
#    return columnspec(columns=col,spec1d=spec1d,pixel=pixel,continuum=continuum,rms=rms,apertures_initial=apertures_initial,apertures_profile)
    return columnspec_array

def fiddle_apertures(columnspec_array,column,window,apertures):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models

    subregion,fit,realvirtual,initial=apertures.subregion,apertures.fit,apertures.realvirtual,apertures.initial
    subregion,fit,realvirtual,initial=aperture_order(subregion,fit,realvirtual,apertures.initial)

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
        ax1.text(fit[j].mean.value,0,str(j+1),fontsize=8)
        if not realvirtual[j]:
            ax1.axvline(x=fit[j].mean.value,color='k',lw=0.5,alpha=1,linestyle='--')

    print('press \'d\' to delete aperture nearest cursor \n')
    print('press \'n\' to add new real aperture at cursor position \n')
    print('press \'a\' to add new phantom aperture at cursor position \n')
    print('press \'z\' to return to initial apertures \n')
    print('press \'q\' to quit \n')
    cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_find(event,[columnspec_array,column,subregion,fit,realvirtual,initial,window,fig]))
    plt.show()

    subregion,fit,realvirtual,initial=aperture_order(subregion,fit,realvirtual,initial)
    return aperture_profile(fit=fit,subregion=subregion,realvirtual=realvirtual,initial=initial)


def get_image_boundary(data):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from specutils import SpectralRegion
    import astropy.units as u
    from astropy.modeling import models

    fig=plt.figure(1)
    ax1=fig.add_subplot(111)
    vmin=np.quantile(data.data.flatten(),0.05)
    vmax=np.quantile(data.data.flatten(),0.95)
    ax1.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)

    lower_x=[]
    lower_y=[]
    upper_x=[]
    upper_y=[]
    print('click \'l\' to add a lower boundary point')
    print('click \'u\' to add an upper boundary point')
    print('click \'q\' when finished')
    cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_boundary(event,[data,lower_x,lower_y,upper_x,upper_y,fig]))
    plt.show()

    x1=np.array(lower_x)
    y1=np.array(lower_y)
    order=np.argsort(y1)
    x1=x1[order]
    y1=y1[order]
    x1=np.append(x1[0],x1)
    y1=np.append(0,y1)
    x1=np.append(x1,x1[len(x1)-1])
    y1=np.append(y1,len(data.data))

    x2=np.array(upper_x)
    y2=np.array(upper_y)
    order=np.argsort(y2)
    x2=x2[order]
    y2=y2[order]
    x2=np.append(x2[0],x2)
    y2=np.append(0,y2)
    x2=np.append(x2,x2[len(x2)-1])
    y2=np.append(y2,len(data.data))

    return image_boundary(lower=[(x1[q],y1[q]) for q in range(0,len(x1))],upper=[(x2[q],y2[q]) for q in range(0,len(x2))])


def get_id_lines(extract1d_array,linelist,continuum_rejection_iterations,threshold_factor,window,id_lines_order,id_lines_rejection_iterations,id_lines_rejection_sigma):
    import numpy as np
    import astropy.units as u
    from specutils.fitting import find_lines_threshold
    from specutils.fitting import find_lines_derivative
    from specutils import Spectrum1D
    import matplotlib.pyplot as plt
    from copy import deepcopy

    id_lines_array=[]
    for i in range(0,len(extract1d_array)):
        print('working on aperture ',extract1d_array[i].aperture)

        spec1d0=Spectrum1D(spectral_axis=deepcopy(extract1d_array[i].spec1d_pixel)*u.AA,flux=deepcopy(extract1d_array[i].spec1d_flux),uncertainty=deepcopy(extract1d_array[i].spec1d_uncertainty),mask=deepcopy(extract1d_array[i].spec1d_mask))
        continuum0,rms0=get_continuum(spec1d0,continuum_rejection_iterations)
        pixel0=extract1d_array[i].spec1d_pixel*u.AA

        spec_contsub=spec1d0-continuum0(pixel0.value)
        spec_contsub.uncertainty.quantity.value[:]=rms0
        id_lines_initial0=find_lines_derivative(spec_contsub,flux_threshold=threshold_factor*rms0)#find peaks in continuum-subtracted "spectrum"
        fit_lines=get_aperture_profile(id_lines_initial0,spec1d0,continuum0,window)

        fig=plt.figure(1)
        ax1=fig.add_subplot(111)
        ax1.set_xlim([np.min(extract1d_array[i].spec1d_pixel[extract1d_array[i].spec1d_mask==False]),np.max(extract1d_array[i].spec1d_pixel[extract1d_array[i].spec1d_mask==False])])
        ax1.plot(extract1d_array[i].spec1d_pixel[extract1d_array[i].spec1d_mask==False],extract1d_array[i].spec1d_flux[extract1d_array[i].spec1d_mask==False],color='k',lw=0.5)
        ax1.plot(pixel0.value[extract1d_array[i].spec1d_mask==False],continuum0(pixel0.value)[extract1d_array[i].spec1d_mask==False],color='b',lw=0.3)
        ax1.set_xlabel('pixel')
        ax1.set_ylabel('counts')
        for j in range(0,len(fit_lines.fit)):
            use=np.where((extract1d_array[i].spec1d_pixel>=fit_lines.subregion[j].lower.value)&(extract1d_array[i].spec1d_pixel<=fit_lines.subregion[j].upper.value))[0]
            sub_spectrum_pixel=extract1d_array[i].spec1d_pixel[use]*u.AA
            y_fit=fit_lines.fit[j](sub_spectrum_pixel).value+continuum0(sub_spectrum_pixel.value)
            ax1.plot(sub_spectrum_pixel,y_fit,color='r',lw=0.3)
#            x_center=id_lines_initial0['line_center'][j].value
#            print(fit_lines.fit[j].mean.value)
            plt.axvline(x=fit_lines.fit[j].mean.value,linestyle=':',color='0.3',lw=0.3)

        line_centers=[fit_lines.fit[j].mean for j in range(0,len(fit_lines.fit))]
        id_lines_pix=[]
        id_lines_wav=[]
        id_lines_species=[]
        order=[id_lines_order]
        rejection_iterations=[id_lines_rejection_iterations]
        rejection_sigma=[id_lines_rejection_sigma]
        print('press \'m\' to ID line nearest cursor \n')
        print('press \'d\' to delete ID for line nearest cursor \n')
        print('press \'o\' to change order of polynomial \n')
        print('press \'r\' to change rejection sigma factor \n')
        print('press \'t\' to change number of rejection iterations \n')
        print('press \'q\' to quit \n')
        print(order)
        cid=fig.canvas.mpl_connect('key_press_event',lambda event: on_key_id_lines(event,[deepcopy(extract1d_array[i]),continuum0,fit_lines,linelist,line_centers,id_lines_pix,id_lines_wav,order,rejection_iterations,rejection_sigma,fig]))

        plt.show()
        plt.close()
        np.pause()
#        id_lines0=id_lines(xxx     columns=col0,spec=spec1d0.data,mask=spec1d0.mask,err=spec1d0.uncertainty,pixel=pixel0,continuum=continuum0,rms=rms0,apertures_initial=apertures_initial0,apertures_profile=apertures_profile0)#have to break up spec1d into individual data/mask/uncertainty fields in order to allow pickling of id_lines_array
        id_lines_array.append(id_lines0)
    return id_lines_array

def get_extract1d(j,data,apertures_profile_middle,aperture_array,aperture_peak,pix):
    import numpy as np
    from astropy.nddata import CCDData
    from specutils import Spectrum1D
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty

    below=np.where([aperture_array[q].trace_aperture<aperture_array[j].trace_aperture for q in np.where(apertures_profile_middle.realvirtual)[0]])[0]
    above=np.where([aperture_array[q].trace_aperture>aperture_array[j].trace_aperture for q in np.where(apertures_profile_middle.realvirtual)[0]])[0]
    if len(below)>0:
        below0=aperture_peak[np.max(below)]
    else:
        below0=0.
    if len(above)>0:
        above0=aperture_peak[np.min(above)]
    else:
        above0=np.float(len(data.data))

    spec=[]
    spec_error=[]
    spec_mask=[]
    for x in pix:
        ymid=aperture_array[j].trace_func(x)
        lsf_sigma=aperture_array[j].lsf_sigma(x)
        wing=np.min([3.*lsf_sigma,(above0-below0)/2./2.])
        if wing>0.:
            y1=np.long(ymid-wing)
            y2=np.long(ymid+wing)
            sss=data[y1:y2+1,x]
            sum1=CCDData([0.],unit=data.unit,mask=[False])
            sum2=0.
            for k in range(0,len(sss.data)):
                int1=0.5*np.erf(ymid/lsf_sigma/np.sqrt(2.))-np.erf((ymid-(y1+k))/lsf_sigma/np.sqrt(2.))
                int2=0.5*np.erf(ymid/lsf_sigma/np.sqrt(2.))-np.erf((ymid-(y1+k+1))/lsf_sigma/np.sqrt(2.))
                weight=int2-int1
                sum1=sum1.add((sss[k].multiply(weight)).divide(sss.uncertainty.quantity.value[k]**2))
                sum2+=weight**2/sss.uncertainty.quantity.value[k]**2
            val=sum1.divide(sum2)
#        val=sum1.multiply(sum2)
            spec.append(val.data[0])
            spec_mask.append(val.mask[0])
#        print(x,sum1,sum2,y1,y2,sss.data,val,val.uncertainty._array)
            spec_error.append(val.uncertainty.quantity.value)
        else:
            spec.append(0.)
            spec_mask.append(True)
#        print(x,sum1,sum2,y1,y2,sss.data,val,val.uncertainty._array)
            spec_error.append(999.)
            
    spec=np.array(spec)
    spec_mask=np.array(spec_mask)
    spec_error=np.array(spec_error)

    spec_mask[np.where(((pix<aperture_array[j].trace_pixel_min)|(pix>aperture_array[j].trace_pixel_max)))[0]]=True
    return extract1d(aperture=aperture_array[j].trace_aperture,spec1d_pixel=pix,spec1d_flux=spec*data.unit,spec1d_uncertainty=StdDevUncertainty(spec_error),spec1d_mask=spec_mask)
#spec1d=Spectrum1D(spectral_axis=pix*u.AA,flux=spec*data.unit,uncertainty=StdDevUncertainty(spec_error),mask=spec_mask))

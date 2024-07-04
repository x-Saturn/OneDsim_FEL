import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import stats
from scipy.fftpack import fft


# plot longitudinal distribution
def plot_longitudinal_phase(theta,eta,slice):
    plt.scatter(list(theta[slice,:]/np.pi),list(eta[slice,:]),s=3)
    plt.xlabel(r'$\theta/\pi$')
    plt.ylabel(r'$\eta$')
    plt.title(f'longtitudinal phase at {slice}th slice')
    plt.show()


# plot both pulse and phase space (anim/snap)
def plot_ps_pulse(file,slice,sample=1,z=(0,-1,1),bunching=False):
    f=h5py.File(file,'r')
    delz=f['delz'][0]
    dels=delz
    zsteps=int(f['zsteps'][0])
    ssteps=int(f['ssteps'][0])
    intensity=f['field']['intensity'][:]
    npart=int(f['npart'][0])
    # begin different kinds of slice
    if type(slice)==int and slice>=0 and slice<ssteps:
        case=1
        if f[f'slice{slice:06}']['current'][0]==0:
            raise ValueError(f'!!! No electrons in slice{slice:06} !!!')
        else:
            theta=f[f'slice{slice:06}']['theta'][:]
            eta=f[f'slice{slice:06}']['eta'][:]
        if bunching:
            bunch=abs(f['bunching'][slice])
    elif type(slice)==tuple or type(slice)==list:
        case=2
        deltheta=2*np.pi/sample
        s_start,s_end=slice
        for k in range(s_start,s_end+1):
            if f[f'slice{k:06}']['current'][0]==0:
                raise ValueError(f'!!! No electrons in slice{k:06} !!!')
            else:
                if k==s_start:
                    theta=f[f'slice{k:06}']['theta'][:]+deltheta*k
                    eta=f[f'slice{k:06}']['eta'][:]
                else:
                    theta_new=f[f'slice{k:06}']['theta'][:]+deltheta*k
                    eta_new=f[f'slice{k:06}']['eta'][:]
                    theta=np.concatenate((theta,theta_new),axis=0)
                    eta=np.concatenate((eta,eta_new),axis=0)
        if bunching:
            bunch=abs(f['bunching'][s_start:s_end+1])
    elif type(slice)==int and slice==-1:
        case=3
        # whole bunch
        s_start=0
        while f[f'slice{s_start:06}']['current'][0]==0:
            s_start=s_start+1
        theta=f[f'slice{s_start:06}']['theta'][:]+2*s_start*np.pi
        eta=f[f'slice{s_start:06}']['eta'][:]
        s_end=s_start+1
        try:
            while f[f'slice{s_end:06}']['current'][0]!=0:
                theta_new=f[f'slice{s_end:06}']['theta'][:]+2*s_end*np.pi
                eta_new=f[f'slice{s_end:06}']['eta'][:]
                theta=np.concatenate((theta,theta_new),axis=0)
                eta=np.concatenate((eta,eta_new),axis=0)
                s_end=s_end+1
        except KeyError as e:
            print("Bunch to the end of time window")
        s_end=s_end-1
        print(f'Whole bunch length includes {s_end-s_start+1} slices, from {s_start:06} to {s_end:06}')
        if bunching:
            bunch=abs(f['bunching'][s_start:s_end+1])
    else:
        raise ValueError("!!! Invalid type of slice or slice out of range !!!")
    # end different kinds of slice
    f.close()
    s=np.linspace(0,dels*ssteps,ssteps+1)
    zmax=delz*zsteps
    # begin different kinds of z
    if type(z)==tuple or type(z)==list: # anim
        zstart,zend,zjump=z
        zstart=int(np.round(zstart*zsteps/zmax))
        if zend==-1:
            zend=zsteps
        else:
            zend=int(np.round(zend*zsteps/zmax))
        # begin plot
        plt.ion()
        fig=plt.figure(figsize=(18,6))
        for zz in range(zstart,zend,zjump):
            plt.clf()
            # plot ax1 as pulse
            ax1=fig.add_subplot(121)
            ax1.set_title(f'pulse intensity at z={zz*delz:.3f}')
            ax1.plot(s,intensity[:,zz])
            ax1.set_xlabel(r'$z_1$')
            ax1.set_ylabel(r'$|A|^2$')
            if case==1:
                ax1.axvline(dels*slice,linestyle='--',color='r')
            elif case==2 or case==3:
                ax1.axvline(dels*s_start,linestyle='--',color='r')
                ax1.axvline(dels*s_end,linestyle='--',color='r')
            # plot ax2 as ps phase
            ax2=fig.add_subplot(122)
            ax2.set_title(f'micro particles phase space at z={zz*delz:.3f}')
            if bunching:
                bunch_zz=bunch[:,zz]
                bunch_zz=np.repeat(bunch_zz,npart)
                bunch_max=bunch.max()
                bunch_min=bunch.min()
                im=ax2.scatter(theta[:,zz]/np.pi,eta[:,zz],s=1,c=bunch_zz,cmap='viridis',vmin=bunch_min,vmax=bunch_max)
                fig.colorbar(im,ax=ax2,orientation='vertical')
            else:
                ax2.scatter(theta[:,zz]/np.pi,eta[:,zz],s=1)
            ax2.set_xlabel(r'$\theta/\pi$')
            ax2.set_ylabel(r'$\eta$')
            ax2.axhline(0,linestyle='--',color='r')
            plt.pause(0.1)
            if zz+zjump>=zend:
                plt.ioff()
    elif (type(z)==int or type(z)==float) and z<=zmax: # snap
        if z>=0:
            zz=int(np.round(z*zsteps/zmax))
        elif z==-1:
            zz=zsteps
        else:
            raise ValueError("!!! Invalid type of z or z out of range !!!")
        fig=plt.figure(figsize=(18,6))
        ax1=fig.add_subplot(121)
        ax1.set_title(f'pulse intensity at z={zz*delz:.3f}')
        ax1.plot(s,intensity[:,zz])
        ax1.set_xlabel(r'$z_1$')
        ax1.set_ylabel(r'$|A|^2$')
        if case==1:
            ax1.axvline(dels*slice,linestyle='--',color='r')
        elif case==2 or case==3:
            ax1.axvline(dels*s_start,linestyle='--',color='r')
            ax1.axvline(dels*s_end,linestyle='--',color='r')
        # plot ax2 as ps phase
        ax2=fig.add_subplot(122)
        ax2.set_title(f'micro particles phase space at z={zz*delz:.3f}')
        if bunching:
            bunch_zz=bunch[:,zz]
            bunch_zz=np.repeat(bunch_zz,npart)
            bunch_max=bunch.max()
            bunch_min=bunch.min()
            im=ax2.scatter(theta[:,zz]/np.pi,eta[:,zz],s=1,c=bunch_zz,cmap='viridis',vmin=bunch_min,vmax=bunch_max)
            fig.colorbar(im,ax=ax2,orientation='vertical')
        else:
            ax2.scatter(theta[:,zz]/np.pi,eta[:,zz],s=1)            
        ax2.set_xlabel(r'$\theta/\pi$')
        ax2.set_ylabel(r'$\eta$')
        ax2.axhline(0,linestyle='--',color='r')
        plt.show()
    else:
        raise ValueError("!!! Invalid type of z or z out of range !!!")
        # end different kinds of z


# plot pulse and avgeta (anim/snap)
def plot_avgeta_pulse(file,slice=-1,z=(0,-1,1)):
    f=h5py.File(file,'r')
    delz=f['delz'][0]
    dels=delz
    zsteps=int(f['zsteps'][0])
    ssteps=int(f['ssteps'][0])
    intensity=f['field']['intensity'][:]
    s=np.linspace(0,dels*ssteps,ssteps+1)
    avg_eta=[]
    if type(slice)==int and slice==-1: # whole bunch
        s_start=0
        while f[f'slice{s_start:06}']['current'][0]==0:
            s_start=s_start+1
        eta=f[f'slice{s_start:06}']['eta'][:]
        slice_eta=np.average(eta,axis=0)
        avg_eta.append(slice_eta)
        s_end=s_start+1
        try:
            while f[f'slice{s_end:06}']['current'][0]!=0:
                eta_new=f[f'slice{s_end:06}']['eta'][:]
                slice_eta=np.average(eta_new,axis=0)
                avg_eta.append(slice_eta)
                s_end=s_end+1
        except KeyError as e:
            print("Bunch to the end of time window")
        s_end=s_end-1
        print(f'whole bunch length includes {s_end-s_start+1} slices, from {s_start:06} to {s_end:06}')
    elif type(slice)==tuple or type(slice)==list:
        s_start,s_end=slice
        for k in range(s_start,s_end+1):
            if f[f'slice{k:06}']['current'][0]==0:
                raise ValueError(f'!!! NO ELECTRONS IN THE SLICE{k:06} !!!')
            else:
                if k==s_start:
                    eta=f[f'slice{k:06}']['eta'][:]
                    slice_eta=np.average(eta,axis=0)
                    avg_eta.append(slice_eta)
                else:
                    eta_new=f[f'slice{k:06}']['eta'][:]
                    slice_eta=np.average(eta_new,axis=0)
                    avg_eta.append(slice_eta)
    else:
        raise ValueError("!!! Invalid type of slice or out of range !!!")
    avg_eta=np.array(avg_eta)
    f.close()
    if type(z)==tuple or type(z)==list: # anim
        zstart,zend,zjump=z
        if zend==-1:
            zend=zsteps
        else:
            zmax=delz*zsteps
            zstart=int(np.round(zstart*zsteps/zmax))
            zend=int(np.round(zend*zsteps/zmax))
        # begin plot
        plt.ion()
        fig=plt.figure(figsize=(18,6))
        # ax1=fig.add_subplot(121)
        # ax2=fig.add_subplot(122)
        for zz in range(zstart,zend,zjump):
            plt.clf()
            ax1=fig.add_subplot(121)
            ax2=fig.add_subplot(122)
            ax1.set_title(f'pulse intensity at z={zz*delz:.3f}')
            ax2.set_title(f'avg slice energy at z={zz*delz:.3f}')
            ax1.plot(s,intensity[:,zz],label='intensity')
            ax1.legend()
            ax1.set_xlabel(r'$z_1$')
            ax1.set_ylabel(r'field')
            ax2.plot(s[s_start:s_end+1:],avg_eta[:,zz])
            ax2.set_xlabel(r'$z_1$')
            ax2.set_ylabel(r'$slice\_avg\_\eta$')
            ax1.axvline(dels*s_start,linestyle='--',color='r')
            ax1.axvline(dels*s_end,linestyle='--',color='r')
            ax2.axhline(0,linestyle='--',color='r')
            plt.pause(0.1)
            if zz+zjump>=zend:
                plt.ioff()
    elif (type(z)==int or type(z)==float) and z<=zmax: # snap
        if z>=0:
            zz=int(np.round(z*zsteps/zmax))
        elif z==-1:
            zz=zsteps
        else:
            raise ValueError("!!! Invalid type of z or out of range !!!")
        fig=plt.figure(figsize=(18,6))
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)
        ax1.set_title(f'pulse intensity at z={zz*delz:.3f}')
        ax2.set_title(f'avg slice energy at z={zz*delz:.3f}')            
        ax1.plot(s,intensity[:,zz],label='intensity')
        ax1.legend()
        ax1.set_xlabel(r'$z_1$')
        ax1.set_ylabel(r'field')            
        ax2.plot(s[s_start:s_end+1:],avg_eta[:,zz])
        ax2.set_xlabel(r'$z_1$')
        ax2.set_ylabel(r'$slice\_avg\_\eta$')            
        ax1.axvline(dels*s_start,linestyle='--',color='r')
        ax1.axvline(dels*s_end,linestyle='--',color='r')
        ax2.axhline(0,linestyle='--',color='r')
        plt.show()
    else:
        raise ValueError("!!! Invalid type of z or z out of range !!!")


# plot whole phase trajectory for one electron
def plot_traj(file,slice,num=0):
    f=h5py.File(file,'r')
    zsteps=int(f['zsteps'][0])
    theta=f[f'slice{slice:06}']['theta'][num,0:zsteps]
    eta=f[f'slice{slice:06}']['eta'][num,0:zsteps]
    f.close()
    plt.plot(theta/np.pi,eta)
    plt.scatter(theta[0]/np.pi,eta[0])
    plt.xlabel(r'$\theta/\pi$')
    plt.ylabel(r'$\eta$')
    plt.title(f'one phase trajectory at slice{slice:06}')
    plt.show()


# FWHM
def plot_FWHM(file,zstart=0):
    f=h5py.File(file,'r')
    intens=f['field']['intensity'][:]
    delz=f['delz'][0]
    zsteps=f['zsteps'][0]
    f.close()
    z=np.linspace(0,delz*zsteps,zsteps+1)
    width=np.array([0]*(zsteps+1),dtype='float')
    for zz in range(zsteps+1):
        pulse=intens[:,zz]
        pp=np.max(pulse)
        h_pp=pp/2
        k_start=0
        while pulse[k_start]<h_pp:
            k_start+=1
        k_end=k_start+1
        while pulse[k_end]>h_pp:
            k_end+=1
        width[zz]=(k_end-k_start)*delz
    zmax=delz*zsteps
    zstart=int(np.round(zstart*zsteps/zmax))
    plt.plot(z[zstart::],width[zstart::])
    plt.title('FWHM-z')
    plt.xlabel(r'$z$')
    plt.ylabel(r'$FWHM$')
    plt.show()


# plot EL-z
def plot_EL(file):
    f=h5py.File(file,'r')
    intensity=f['field']['intensity'][:]
    delz=f['delz'][0]
    zsteps=int(f['zsteps'][0])
    s_start=0
    while f[f'slice{s_start:06}']['current'][0]==0:
        s_start=s_start+1
    s_end=s_start+1
    try:
        while f[f'slice{s_end:06}']['current'][0]!=0:
            s_end=s_end+1
    except KeyError as e:
        # do nothing
        print("Electron bunch till the end of time window")
    s_end=s_end-1
    lb=s_end-s_start+1
    # print(k)
    f.close()
    z_plot=np.linspace(0,delz*zsteps,zsteps+1)
    rad_mean=np.zeros(zsteps+1,dtype='float')
    for j in range(zsteps+1):
        EL=np.sum(intensity[:,j])/lb
        rad_mean[j]=EL
    plt.plot(z_plot,rad_mean,label=file)
    '''
    f=h5py.File('short_bunch/steady.hdf5','r')
    intens=f['field']['intensity'][:]
    plt.plot(z_plot,intens,label='steady')'''
    plt.xlabel(r'$z$')
    plt.ylabel(r'$E_L$')
    plt.title('Average energy extraction EL')
    plt.axhline(1,linestyle='--',color='r')
    plt.xlim(0,delz*zsteps)
    plt.legend()
    plt.show()


# plot P_peak-z, E-z and fit curve
def plot_PE(file,z_jump=20,zstart=0,log=False):
    f=h5py.File(file,'r')
    intensity=f['field']['intensity'][:]
    delz=f['delz'][0]
    zsteps=int(f['zsteps'][0])
    f.close()
    zmax=delz*zsteps
    zstart_pos=int(np.round(zstart*zsteps/zmax))
    z_plot=np.linspace(zstart,delz*zsteps,zsteps+1-zstart_pos)
    p_peak=np.array([0]*(zsteps+1-zstart_pos),dtype='float')
    energy=np.array([0]*(zsteps+1-zstart_pos),dtype='float')
    for j in range(zsteps+1-zstart_pos):
        p_peak[j]=np.max(intensity[:,j])
        energy[j]=np.sum(intensity[:,j])
    # fit
    slope_p,inter_p,r_p,p_p,err_p=stats.linregress(np.log((z_plot-zstart)[1::]),np.log(p_peak[1::]))
    slope_e,inter_e,r_e,p_e,err_e=stats.linregress(np.log((z_plot-zstart)[1::]),np.log(energy[1::]))
    fit_p=np.exp(inter_p)*z_plot**slope_p
    fit_e=np.exp(inter_e)*z_plot**slope_e
    # plot
    fig=plt.figure(figsize=(8,6))
    if log:
        ax1=fig.add_subplot(111)
        ax1.loglog(z_plot[::z_jump],p_peak[::z_jump],'^',color='red',label='peak_power')
        ax1.loglog(z_plot,fit_p,'--',color='gray',label=f'{np.exp(inter_p):.2f}*(z-{zstart:.2f})^{slope_p:.2f} (r={r_p:.5f})')
        ax2=ax1.twinx()
        ax2.loglog(z_plot[::z_jump],energy[::z_jump],'.',color='green',label='pulse_energy')
        ax2.loglog(z_plot,fit_e,color='gray',label=f'{np.exp(inter_e):.2f}*(z-{zstart:.2f})^{slope_e:.2f} (r={r_e:.5f})')
    else:
        ax1=fig.add_subplot(111)
        ax1.plot(z_plot[::z_jump],p_peak[::z_jump],'^',color='red',label='peak_power')
        ax1.plot(z_plot,fit_p,'--',color='gray',label=f'{np.exp(inter_p):.2f}*(z-{zstart:.2f})^{slope_p:.2f} (r={r_p:.5f})')
        ax2=ax1.twinx()
        ax2.plot(z_plot[::z_jump],energy[::z_jump],'.',color='green',label='pulse_energy')
        ax2.plot(z_plot,fit_e,color='gray',label=f'{np.exp(inter_e):.2f}*(z-{zstart:.2f})^{slope_e:.2f} (r={r_e:.5f})')
    plt.title('Peak_power & Pulse_energy-z')
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$|A|^2$')
    ax1.tick_params(axis='y',colors='red')
    ax2.set_ylabel(r'$\sum |A|^2$')
    ax2.tick_params(axis='y',colors='green')
    ax1.legend(loc=2)
    ax2.legend(loc=4)
    plt.show()


# get far field(t) and plot it 
def get_far_field(file,lambda_g,lambda_c,lambda_r,plot=True):
    f=h5py.File(file,'r')
    ar=f['field']['ar'][:]
    ai=f['field']['ai'][:]
    delz=f['delz'][0]
    dels=delz
    zsteps=int(f['zsteps'][0])
    ssteps=int(f['ssteps'][0])
    f.close()
    ss=np.linspace(0,ssteps*dels,ssteps+1)
    zz=np.linspace(0,zsteps*delz,zsteps+1)
    k=2*np.pi/lambda_r
    mesh_z,mesh_s=np.meshgrid(zz,ss)
    z=mesh_z/lambda_g
    ct=(-mesh_s*lambda_c+z*(1+lambda_c/lambda_g))
    t=ct/3*1e7 # fs
    A=(ar+1j*ai)
    a=A*np.exp(1j*k*(z-ct))
    if plot==True:
        t_plot=t[:,zsteps]
        a_plot=np.real(a[:,zsteps])
        A_plot=np.abs(A[:,zsteps])
        plt.plot(t_plot,a_plot/np.max(abs(a_plot)),linewidth=0.1)
        plt.plot(t_plot,A_plot/np.max(A_plot),linestyle='--',color='r',linewidth=0.1)
        plt.plot(t_plot,-A_plot/np.max(A_plot),linestyle='--',color='r',linewidth=0.1)
        plt.title(f'pulse at the end of the undulator (z={zz[zsteps]:.3f}m)')
        plt.xlabel(r'$t/fs$')
        plt.ylabel(r'a')
        plt.show()
    return z,t,A,a


# do FFT at a certain point z with a certain time-window
def plot_freq(file,lambda_g,lambda_c,lambda_r,z_snap=-1,t_window=(0,-1)):
    z,t,A,a=get_far_field(file,lambda_g,lambda_c,lambda_r,plot=False)
    if z_snap==-1:
        f=h5py.File(file,'r')
        zsteps=int(f['zsteps'][0])
        f.close()
        signal=a[:,zsteps]
        t_serial=t[:,zsteps]
    else:
        pos=int(np.round(z_snap/z[0,1]))
        signal=a[:,pos]
        t_serial=t[:,pos]
    if t_window[1]!=-1:
        t_end=int(np.round(t_window[1])/(t[0,1]-t[0,0]))
        signal=signal[0:t_end+1:]
        t_serial=t_start[0:t_end+1:] 
    if t_window[0]!=0:
        t_start=int(np.round(t_window[0]/(t[0,1]-t[0,0])))
        signal=signal[t_start::]
        t_serial=t_serial[t_start::]
    # begin FFT
    fs=1/(t_serial[0]-t_serial[1]) # 1e15 Hz
    L=len(signal)
    N=np.power(2,np.ceil(np.log2(L)))
    result=np.abs(fft(x=signal,n=int(N)))
    freq=np.arange(int(N/2))*fs/N
    result=result[range(int(N/2))]
    # find max
    max_result=np.max(result)
    max_index=np.where(result==max_result)[0][0]
    max_freq=freq[max_index]
    max_lambda=300/max_freq # nm
    print(f'Spectrum peaks at {max_lambda:.3f} nm')
    # end FFT
    # begin plot
    fig=plt.figure(figsize=(18,6))
    ax1=fig.add_subplot(121)
    ax1.plot(t_serial,np.real(signal)/np.max(np.real(signal)),linewidth=0.1)
    ax1.set_title('pulse-time')
    ax1.set_xlabel(r'$t/fs$')
    ax1.set_ylabel(r'$a$')
    ax2=fig.add_subplot(122)
    ax2.plot(freq,result/np.max(result))
    ax2.set_title('pulse-freq')
    ax2.set_xlabel(r'$f/10^{15}Hz$')
    ax2.set_ylabel(r'$a$')
    plt.show()



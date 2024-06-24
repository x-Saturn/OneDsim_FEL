import numpy as np
# import matplotlib.pyplot as plt
import time
import h5py


def peak_intensity(y):
    pp=0
    for i in range(len(y)):
        for j in range(len(y[0])):
            if y[i,j]>pp:
                pp=y[i,j]
    return pp 


def SmallGain(theta_init,eta_init,a_init,zsteps,zmax,save=''):
    npart=len(theta_init)
    ar=np.ones(zsteps+1,dtype='float')*a_init
    ai=np.zeros(zsteps+1,dtype='float')
    eta=np.zeros((npart,zsteps+1),dtype='float')
    theta_half=np.zeros((npart,zsteps+1),dtype='float')
    theta=np.zeros((npart,zsteps),dtype='float')
    bunching=np.zeros(zsteps,dtype='complex')
    delz=zmax/zsteps
    print('Progress: Set initial variables done')
    # save h5 file
    if not save=='':
        print('Begin create hdf5 file')
        f=h5py.File(f'{save}.hdf5','w')
        f.create_dataset('zsteps',data=[zsteps])
        f.create_dataset('npart',data=[npart])
        f.create_dataset('delz',data=[delz])
    print('Begin iteration')
    eta[:,0]=eta_init.T
    theta_half[:,0]=theta_init.T-eta[:,0]*delz/2
    if not save=='':
        slice_gr=f.create_group('slice000000')
        slice_gr.create_dataset('current',data=[1.0])
    for j in range(zsteps):
        theta0=theta_half[:,j]+eta[:,j]*delz/2
        theta[:,j]=theta0
        theta_half[:,j+1]=theta_half[:,j]+eta[:,j]*delz
        eta[:,j+1]=eta[:,j]-2*a_init*np.cos(theta_half[:,j+1])*delz
        bunching[j]=np.mean(np.cos(theta0))-1j*np.mean(np.sin(theta0))
    if not save=='':
        slice_gr.create_dataset('theta',data=theta)
        slice_gr.create_dataset('eta',data=eta)
        f.create_dataset('bunching',data=bunching.T)
        field_gr=f.create_group('field')
        field_gr.create_dataset('ar',data=ar.T)
        field_gr.create_dataset('ai',data=ai.T)
        f.close()
    return bunching.T


def LongBunch(theta_init,eta_init,a_init,zsteps,zmax,save=''):
    npart=len(theta_init)
    ar=np.zeros(zsteps+1,dtype='float')
    ai=np.zeros(zsteps+1,dtype='float')
    eta=np.zeros((npart,zsteps+1),dtype='float')
    thetahalf=np.zeros((npart,zsteps+1),dtype='float')
    theta=np.zeros((npart,zsteps),dtype='float')
    bunching=np.zeros(zsteps,dtype='complex')
    delz=zmax/zsteps
    print('Progress: Set initial variables done')
    # save hdf5 file
    if not save=='':
        print('Begin create hdf5 file')
        f=h5py.File(f'{save}.hdf5','w')
        f.create_dataset('zsteps',data=[zsteps])
        f.create_dataset('npart',data=[npart])
        f.create_dataset('delz',data=[delz])
    print('Begin iteration')
    eta[:,0]=eta_init.T
    thetahalf[:,0]=theta_init.T-eta[:,0]*delz/2
    ar[0]=a_init
    if not save=='':
        slice_gr=f.create_group('slice000000')
        slice_gr.create_dataset('current',data=[1.0])
    for j in range(zsteps):
        theta0=thetahalf[:,j]+eta[:,j]*delz/2
        theta[:,j]=theta0
        sinavg=np.sum(np.sin(theta0))/npart
        cosavg=np.sum(np.cos(theta0))/npart
        arhalf=ar[j]+(cosavg)*delz/2
        aihalf=ai[j]+(-sinavg)*delz/2
        thetahalf[:,j+1]=thetahalf[:,j]+eta[:,j]*delz
        eta[:,j+1]=eta[:,j]-2*arhalf*np.cos(thetahalf[:,j+1])*delz+2*aihalf*np.sin(thetahalf[:,j+1])*delz
        sinavg=np.sum(np.sin(thetahalf[:,j+1]))/npart
        cosavg=np.sum(np.cos(thetahalf[:,j+1]))/npart
        ar[j+1]=ar[j]+(cosavg)*delz
        ai[j+1]=ai[j]+(-sinavg)*delz
        bunching[j]=np.mean(np.cos(theta0))-1j*np.mean(np.sin(theta0))
    intensity=ar**2+ai**2
    if not save=='':
        slice_gr.create_dataset('theta',data=theta)
        slice_gr.create_dataset('eta',data=eta)
        f.create_dataset('bunching',data=bunching.T)
        field_gr=f.create_group('field')
        field_gr.create_dataset('ar',data=ar.T)
        field_gr.create_dataset('ai',data=ai.T)
        field_gr.create_dataset('intensity',data=intensity.T)
        f.close()
    return ar.T,ai.T,intensity.T,bunching.T


# optimize for z1>lb
def oneD_FEL(theta_init,eta_init,a_init,shape,zsteps,zmax,save=''):
    # initialize variables
    t_start=time.time()
    ssteps,npart=len(theta_init),len(theta_init[0])
    ar=np.zeros((ssteps+1,zsteps+1),dtype='float')
    ai=np.zeros((ssteps+1,zsteps+1),dtype='float')
    eta=np.zeros((npart,zsteps+1),dtype='float')
    thetahalf=np.zeros((npart,zsteps+1),dtype='float') # half zstep before theta
    theta=np.zeros((npart,zsteps),dtype='float') # theta at full zsteps
    # theta_out=np.zeros((ssteps,1))
    bunching=np.zeros((ssteps,zsteps),dtype='complex') # calculate bunching at each z
    # set grid
    delz=zmax/zsteps
    dels=delz # why?
    print('Progress: Set initial variables done')
    # save h5 file
    if not save=='':
        print('Begin create hdf5 file')
        f = h5py.File(f'{save}.hdf5','w')
        f.create_dataset('ssteps',data=[ssteps])
        f.create_dataset('zsteps',data=[zsteps])
        f.create_dataset('npart',data=[npart])
        f.create_dataset('delz',data=[delz])
    # load initial variables from the tail
    print('Begin iteration from the tail')
    for k in range(ssteps): # the k th slice
        for tim in range(1,11):
            if k==tim*ssteps//10-1:
                print(f'Progess: {tim*10:n}% done, time consumed {time.time()-t_start:.2f}s')
        if not save=='':
            slice_gr=f.create_group(f'slice{k:06}')
            slice_gr.create_dataset('current',data=[shape[k]])
        if shape[k]!=0:
            # do not put all beam phases in ROM
            ar[k,0]=a_init[k].real
            ai[k,0]=a_init[k].imag
            eta[:,0]=eta_init[k].T
            thetahalf[:,0]=theta_init[k].T-eta[:,0]*delz/2 # theta at -0.5 delz
            # set each slice as a group 
            for j in range(zsteps): # the j th zstep
                theta0=thetahalf[:,j]+eta[:,j]*delz/2 # theta go forward half zstep
                theta[:,j]=theta0
                sumsin=np.sum(np.sin(theta0))
                sumcos=np.sum(np.cos(theta0))
                sinavg=shape[k]*sumsin/npart
                cosavg=shape[k]*sumcos/npart
                # shape is a 1*sstep scaled function for electron density
                arhalf=ar[k,j]+(cosavg)*dels/2
                aihalf=ai[k,j]+(-sinavg)*dels/2
                # field go forward half sstep
                thetahalf[:,j+1]=thetahalf[:,j]+eta[:,j]*delz
                eta[:,j+1]=eta[:,j]-2*arhalf*np.cos(thetahalf[:,j+1])*delz+2*aihalf*np.sin(thetahalf[:,j+1])*delz
                # eta go forward one zstep using thetahalf and ahalf
                sumsin=np.sum(np.sin(thetahalf[:,j+1]))
                sumcos=np.sum(np.cos(thetahalf[:,j+1]))
                sinavg=shape[k]*sumsin/npart
                cosavg=shape[k]*sumcos/npart
                ar[k+1,j+1]=ar[k,j]+(cosavg)*dels
                ai[k+1,j+1]=ai[k,j]+(-sinavg)*dels
                # a go forward one zstep and sstep using thetahalf and a
                bunching[k,j]=np.mean(np.cos(theta0))-1j*np.mean(np.sin(theta0))
            if not save=='':
                slice_gr.create_dataset('theta',data=theta)
                slice_gr.create_dataset('eta',data=eta)
        else:
            ar[k,0]=a_init[k].real
            ai[k,0]=a_init[k].imag
            for j in range(zsteps):
                arhalf=ar[k,j]
                aihalf=ai[k,j]
                ar[k+1,j+1]=ar[k,j]
                ai[k+1,j+1]=ai[k,j]
            # theta=np.zeros((npart,zsteps),dtype='float')
            # eta=np.zeros((npart,zsteps+1),dtype='float')
        # set each slice's theta, eta    
    intensity=ar**2+ai**2
    if not save=='':
        f.create_dataset('bunching',data=bunching)
        field_gr=f.create_group('field')
        field_gr.create_dataset('ar',data=ar)
        field_gr.create_dataset('ai',data=ai)
        field_gr.create_dataset('intensity',data=intensity)
        f.close()
    return ar,ai,intensity,bunching
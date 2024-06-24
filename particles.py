import numpy as np
import matplotlib.pyplot as plt

# load one bucket/slice
def load_bucket(rho,delta,delgam,current,lambdaref,b0=0,npart=512,nbins=4,shotnoise=False):
    # rho: Pierce parameter
    # delgam: relative rms energy spread
    # current: current in A
    # lambdaref: slice length in m
    # npart: n-macro particles per bucket/slice
    # nbins: number of beamlets
    eta=np.zeros(npart,dtype='float')
    theta=np.zeros(npart,dtype='float')
    M=int(npart/nbins) # number of particles in each beamlet
    if not shotnoise:
        for i in range(nbins):
            etaj=(delgam*np.random.randn()+np.sqrt(1+2*rho*delta)-1)/rho
            for j in range(M):
                eta[i*M+j]=etaj # the M particles in the same beamlet have the same energy
                if b0==0:
                    theta[i*M+j]=2*np.pi*(j+1)/M # the M particles in the same beamlet is uniformly distributed
                elif b0==1:
                    theta[i*M+j]=np.pi
    else:
        c=3e8 #speed of light
        e=1.6e-19 #electron charge
        Ns=current*(lambdaref/c)/e #real number of electrons per slice, usually >1e6
        effnoise=np.sqrt(3*M/(Ns/nbins))
        for i in range(nbins):
            etaj=(delgam*np.random.randn()+np.sqrt(1+2*rho*delta)-1)/rho
            for j in range(M):
                eta[i*M+j]=etaj
                if b0==0:
                    theta[i*M+j]=2*np.pi*(j+1)/M+2*np.random.rand(1)*effnoise
                elif b0==1:
                    theta[i*M+j]=np.pi
    return theta, eta


# load all buckets/slices
def general_load_bucket_old(rho,delta,delgam,current,lambdaref,b0,ssteps,shape,npart=512,nbins=4,shotnoise=False):
    # load particles in different slices
    theta_init=np.zeros((ssteps,npart),dtype='float')
    eta_init=np.zeros((ssteps,npart),dtype='float')
    for j in range(ssteps):
        # for each row in the length, load theta and eta
        if shape[j]!=0:
            [theta0,eta0]=load_bucket(rho,delta[j],delgam[j],current[j],lambdaref,b0[j],npart,nbins,shotnoise)
            theta_init[j,:]=theta0
            eta_init[j,:]=eta0
    print('Progess: General load bucket done')
    return theta_init,eta_init


# load all buckets/slices with sample>1
def general_load_bucket(rho,delta,delgam,current,lambdaref,b0,ssteps,shape,sample=1,npart=512,nbins=4,shotnoise=False):
    wavelength_num=ssteps//sample
    if wavelength_num*sample!=ssteps:
        raise ValueError("!!! ssteps should be an integral multiplies sample !!!")
    else:
        # load particles in different slices
        theta_init=np.zeros((ssteps,npart),dtype='float')
        eta_init=np.zeros((ssteps,npart),dtype='float')
        for j in range(wavelength_num):
            # for each row in the length, load theta and eta
            if shape[j*sample]!=0:
                [theta0,eta0]=load_bucket(rho,delta[j],delgam[j],current[j],lambdaref,b0[j],npart*sample,nbins,shotnoise)
                for jj in range(sample):
                    theta_init[j*sample+jj,:]=theta0[jj*npart:(jj+1)*npart:]
                    eta_init[j*sample+jj,:]=eta0[jj*npart:(jj+1)*npart:]
        print('Progess: General load bucket done')
        return theta_init,eta_init


# load initial A0 and shape
def general_temporal_profile(ssteps,smax,A0,opt,para:tuple,plot=False):
    s=np.linspace(0,smax,ssteps)
    a_init=np.array([0]*ssteps,dtype='float')
    if opt=='step':
        # (s_start,s_end)
        s_start,s_end=para
        for i in range(ssteps):
            if s[i]>=s_start and s[i]<=s_end:
                a_init[i]=A0
    elif opt=='gaussian':
        # (mu,sigma)
        mu,sig=para
        for i in range(ssteps):
            a_init[i]=A0*np.exp(-(s[i]-mu)**2/(2*sig**2))
    elif opt=='tri':
        # (s_start,s_peak,s_end)
        s_start,s_peak,s_end=para
        for i in range(ssteps):
            if s[i]>=s_start and s[i]<s_peak:
                a_init[i]=A0*(s[i]-s_start)/(s_peak-s_start)
            elif s[i]>=s_peak and s[i]<=s_end:
                a_init[i]=A0*(s_end-s[i])/(s_end-s_peak)
    else:
        print("!!! NO SUCH OPTION !!!")
    if plot:
        plt.plot(s,a_init)
        plt.xlabel(r'$z_1$')
        plt.title('temporal profile')
        plt.show()
    return a_init
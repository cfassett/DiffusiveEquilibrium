import os, optparse, subprocess, sys, tempfile, math
from params import *
from pynverse import inversefunc  #pip install pynverse  in relevant conda environment
import numpy as np
import matplotlib.pyplot as plt

def man(option, opt, value, parser):
    print >>sys.stderr, parser.usage
    print >>sys.stderr, '''\
This program generates craters over the entire surface of the Moon for ejecta hazard assessment.
'''
    sys.exit()

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def numhart(lowdiamkm, updiamkm):
    hartlow=(10.0**-1.14)*lowdiamkm**-1.83
    harthigh=(10.0**-1.14)*updiamkm**-1.83
      
    hart=(hartlow-harthigh)*surfacearea
    
    return hart

def numtrask(lowdiamkm, updiamkm):
    trasklow=(10.0**-1.1)*lowdiamkm**-2.0
    traskhigh=(10.0**-1.1)*updiamkm**-2.0
      
    trask=(trasklow-traskhigh)*surfacearea
    
    return trask
    
def npfcalc(size):
    '''this finds the NPF ==> N(>=size) for crater with -crater size-.  From Neukum and Ivanov/2001.
    sizes in km''' 
       
    
    correctionfrom1millionyrsto1yr=1.0e-6
       
    log10freq=0.0
   
    npfcoeffs=[-6.076755981,-3.557528,0.781027,1.021521,-0.156012,-0.444058,0.019977,0.08685,-0.005874,-0.006809,0.000825,0.0000554]
    for a in range(12):
        log10freq=log10freq+npfcoeffs[a]*((log10(size))**a)
    
    cumfreq=correctionfrom1millionyrsto1yr*surfacearea*(10.0**log10freq)
        
    return cumfreq
    

def diffusetothresh(diam,visibilitythreshold):
    domainsizepx=400
    sscale=diam/100.0
   
    u=np.zeros((domainsizepx,domainsizepx))
    un=np.zeros((domainsizepx,domainsizepx))
    
    
    alpha=1.0
    xcen=domainsizepx/2
    ycen=domainsizepx/2
    xvals=np.linspace(0,sscale*domainsizepx,num=domainsizepx)-(xcen*sscale)
    
    curradius=diam/2.0
    currange=1.5*curradius
    
    px_r = (curradius/sscale)
    x,y=np.ogrid[:domainsizepx,:domainsizepx]
    distfromcen=(((x-xcen)**2.0 + (y-ycen)**2.0)**0.5)*sscale
    rangefromcen=distfromcen/curradius   
    incrat=np.logical_and(rangefromcen>=0.0,rangefromcen<0.98)
    rim=np.logical_and(rangefromcen>=0.98,rangefromcen<1.02)
    outcrat=np.logical_and(rangefromcen>=1.02,rangefromcen<1.5)
    
    sigma=0.0#35
    
    if diam>400:
        dDmax=0.21
    elif diam>200:
        dDmax=0.17
    elif diam>100:
        dDmax=0.15
    elif diam>40:
        dDmax=0.13
    elif diam>10:
        dDmax=0.11           
    else:
        dDmax=0.1    
    dDmax=dDmax++np.random.normal(0,sigma)
     
    interiorfloor=-(dDmax-0.036822095)*diam
        
    # FRESH CRATER MODEL:  Basically what's in Fassett and Thomson, 2014, with a little fix implremented at the rim (whoops)
    # forced to have a flat floor to match dDMax at small sizes

    u[incrat]=(-0.228809953+0.227533882*rangefromcen[incrat]+0.083116795*(rangefromcen[incrat]**2.0)-0.039499407*(rangefromcen[incrat]**3.0))*diam
    u[rim]=0.036822095*diam
    u[outcrat]=(0.188253307-0.187050452*rangefromcen[outcrat]+0.01844746*(rangefromcen[outcrat]**2.0)+0.01505647*(rangefromcen[outcrat]**3.0))*diam
    u[u<interiorfloor]=interiorfloor
        



    # courant criteria and timesteps
    dtst=0.5*(sscale**2.0)/(2*alpha)
    
    r=dtst*alpha/(sscale**2.0)
    
    dD=(np.max(u)-np.min(u))/diam  
    step=0 
    while (dD>=visibilitythreshold):                    #Standard Explicit diffusion equations; modified from my old Matlab code
        un[1:-1, 1:-1] = u[1:-1, 1:-1] + r*((u[2:, 1:-1]+u[:-2, 1:-1])+(u[1:-1, 2:]+u[1:-1, :-2]-4*u[1:-1, 1:-1]))
        u=np.copy(un)
        step=step+1   
        dD=(np.max(u)-np.min(u))/diam  
    
    kappaT=step*dtst
    return kappaT


def makegruns(minlogmassgrams=-18.0,maxlogmassgrams=5.0):
    logstep=1.0
    
    # Constants below equation A2 from Grun
    grun_c_s=[4.00E+29, 1.50E+44, 1.10E-02, 2.20E+03, 1.50E+01]
    grun_y_s=[1.85E+00, 3.70E+00, -5.20E-01, 3.06E-01, -4.38E+00]
    
    grunsize_m=[]
    grunflux=[]
    masslog=minlogmassgrams
    
    while masslog<=maxlogmassgrams:
        massg=10.0**masslog
        
        # converts particle mass to radius 
        grunsize=0.01*(((1.0/(impdensity/1000.0))*((3.0/4.0)*(1.0/math.pi)*massg))**(1.0/3.0))
        
        
        # equation A2 from Grun
        fluxone=(grun_c_s[0]*(massg**grun_y_s[0])+grun_c_s[1]*(massg**grun_y_s[1])+grun_c_s[2])**grun_y_s[2]
        fluxtwo=(grun_c_s[3]*(massg**grun_y_s[3])+grun_c_s[4])**grun_y_s[4]
        flux=fluxone+fluxtwo             
        #fluxes are in per m2 per second
        #per km2 per yeaar
        flux_rescale=flux*86400.0*365.25*1000000.0

        
        grunflux.append(flux_rescale)
        grunsize_m.append(grunsize)
        
        masslog=masslog+logstep
        
    return grunsize_m, grunflux




    
def hohocratersize(impradius):
    # Holsapple theory paper style
    # RELIES ON COMMON SCALING PARAMETERS FROM SCALINGPARAMS.PY  
    
        
    
    impmass=((4.0*math.pi)/3.0)*impdensity*(impradius**3.0)  #impactormass
    pi2=(gravity*impradius)/(effvelocity**2.0)     
    pi3=strength/(targdensity*(effvelocity**2.0)) 
    expone=(6.0*nu-2.0-mu)/(3.0*mu)
    exptwo=(6.0*nu-2.0)/(3.0*mu)
    expthree=(2.0+mu)/2.0
    expfour=(-3.0*mu)/(2.0+mu)  
    piV=K1*(pi2*(densityratio**expone)+(K2*(pi3*(densityratio**exptwo))**expthree)**expfour)
    V=(impmass*piV)/targdensity   #m3 for crater
    craterrimrad=Kr*(V**(1.0/3.0))
    cratersize=2.0*craterrimrad
    
    
    return cratersize

def cratcalc(size_km, uppergrunwarning=False):
    '''this finds the Grun or Neukum ==> N(>=size)  crater with -crater size- size
    number for a surface area set in the scaling params file) (currently 1 km2)
    
    >=10m just calls the NPF    
    ''' 
    
    #these are sizes in m,  assuming hohoscaling, and grunfluxes (per km2 per yr)
    grunimpradii_m, grunfluxes=makegruns()
    grundiameters=[hohocratersize(r) for r in grunimpradii_m]    
    grundiameterslog=[log10(el) for el in grundiameters]
    grunfluxeslog=[log10(el) for el in grunfluxes]

    size=size_km*1000.0
    if size<10.0 or uppergrunwarning:
        slog=log10(size)
        fluxintlog=np.interp(slog, grundiameterslog, grunfluxeslog)        
        fluxint=10.0**fluxintlog
        globalfreq=surfacearea*fluxint
    else:
        globalfreq=npfcalc(size/1000.0)   
    return globalfreq

def neqtime(time):
    timeMa=time/1.0e6
    timeCorMa=(0.0000000000000544*(math.exp(0.00693*timeMa)-1)+0.000000838*timeMa)/0.000000838  
    timeCor=timeCorMa*1.0e6 # convert back to Neukum-eq years (not physical years, for the early chronology!)
    
    return timeCor
    
eqtimeton=inversefunc(neqtime)
   
def main():
    try:
        try:
            usage = "usage: KappaforGrunEquil_changingFlat.py\n"
            parser = optparse.OptionParser(usage=usage)
            (options, inargs) = parser.parse_args()
            #if not inargs: parser.error("Example text")    Currently setup for no arguments so this is pointless.
            #firstarg=inargs[0]
            
        except optparse.OptionError, msg:
            raise Usage(msg)            

        # probably best not to have this bigger than 200m because those aren't in equilibrium
        minsizekmlog10=-3.0
        maxsizekmlog10=-0.75
        lstep=0.05

        size=minsizekmlog10
        sizes=[]
        kHart=[]
        kTrask=[]
        outname="kappa_trask_flat2.csv"
        out=open(outname,'w+')
        while size<maxsizekmlog10:
            sizekm=10.0**size
            sizeupperkm=10.0**(size+lstep)
            
            if (sizeupperkm>=0.01 and sizekm<0.01):
                uppergrunwarning=True
                oneyearfreqatthissize=cratcalc(sizekm)-cratcalc(sizeupperkm,uppergrunwarning)
            else:
                oneyearfreqatthissize=cratcalc(sizekm)-cratcalc(sizeupperkm)
            yearstohart=numhart(sizekm,sizeupperkm)/oneyearfreqatthissize
            yearstotrask=numtrask(sizekm,sizeupperkm)/oneyearfreqatthissize
            sizemid=10.0**(0.5*(log10(sizeupperkm)+log10(sizekm)))
            sizes.append(1000.0*sizemid)  #convert km to m            
            kT_thissize=diffusetothresh(sizekm*1000.0,visibilitythreshold)                
            k_thissizeT=kT_thissize/(eqtimeton(yearstotrask)/1.0e6)
            kTrask.append(k_thissizeT)
            k_thissizeH=kT_thissize/(eqtimeton(yearstohart)/1.0e6)
            kHart.append(k_thissizeH)
            
            #print sizemid, oneyearfreqatthissize, numhart(sizekm,sizeupperkm), eqtimeton(yearstohart), numtrask(sizekm,sizeupperkm), eqtimeton(yearstotrask)
            print sizemid, k_thissizeT, k_thissizeH
            out.write(str(sizemid)+","+str(k_thissizeT)+"\n")
            size=size+lstep



        plt.yscale('log')
        plt.xlim([0.95,250.0])
        plt.ylim([1e-2,1.0])
        
        plt.xscale('log')
        plt.xlabel("Crater Diameter (m)")
        plt.ylabel("Required Kappa $(m^2/My)$")

        plt.scatter(sizes, kTrask, label='Trask')
        plt.scatter(sizes, kHart, label='Hartmann')
        plt.legend()
        plt.show()
        
    except Usage, err:
        print >>sys.stderr, err.msg
        # print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
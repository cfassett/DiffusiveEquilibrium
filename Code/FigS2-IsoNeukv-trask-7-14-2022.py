import os, optparse, subprocess, sys, tempfile, math
from scalingparams import *
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
    
    cumfreq=correctionfrom1millionyrsto1yr*(10.0**log10freq)
        
    return cumfreq
    
    


def makegruns(minlogmassgrams=-18.0,maxlogmassgrams=2.0):
    #Note that the upper valid limit on Grun is 10^2 grams.  Need to solve for bigger particles some other way.
   
    
    logstep=0.1
    
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
        #print massg, grunsize  Useful to print these out as a cross-check.  This is correct, however (sizes in meters).
         
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
    # Fixes error in piV 7/14/22
        
    
    impmass=((4.0*math.pi)/3.0)*impdensity*(impradius**3.0)  #impactormass
    pi2=(gravity*impradius)/(effvelocity**2.0)     
    pi3=strength/(targdensity*(effvelocity**2.0)) 
    expone=(6.0*nu-2.0-mu)/(3.0*mu)
    exptwo=(6.0*nu-2.0)/(3.0*mu)
    expthree=(2.0+mu)/2.0
    expfour=(-3.0*mu)/(2.0+mu)  
    piV=K1*(pi2*(densityratio**expone)+(K2*(pi3*(densityratio**exptwo))**expthree))**expfour
    V=(impmass*piV)/targdensity   #m3 for crater
    craterrimrad=Kr*(V**(1.0/3.0))
    cratersize=2.0*craterrimrad
    
    
    return cratersize


def cratcalc(size_km):
    '''this finds the Grun or Neukum ==> N(>=size)  crater with -crater size- size
    number for a surface area set in the scaling params file) (currently 1 km2)
    These are cumulative crater frequencies (per usual)
    
    >=10m just calls the NPF    
    ''' 
    
    #these are sizes in m,  assuming hohoscaling, and grunfluxes (per km2 per yr)
    grunimpradii_m, grunfluxes=makegruns()
    grundiameters=[hohocratersize(r) for r in grunimpradii_m]    
    grundiameterslog=[log10(el) for el in grundiameters]
    grunfluxeslog=[log10(el) for el in grunfluxes]

    size=size_km*1000.0
    slog=log10(size)
    if size<10.0:
        if slog>grundiameterslog[-1]:   #In range where need to interpolate between Neukum at 10m and whatever the maximum Grun diameter is
            interpdiams=[grundiameterslog[-1], 1]
            interpfreqs=[grunfluxeslog[-1],log10(npfcalc(0.01))]            
            fluxintlog=np.interp(slog, interpdiams, interpfreqs)
            fluxint=10.0**fluxintlog            
        else:          #In range where we draw from the Grun table         
            fluxintlog=np.interp(slog, grundiameterslog, grunfluxeslog)        
            fluxint=10.0**fluxintlog
        globalfreq=fluxint
    else:
        globalfreq=npfcalc(size_km)           
    return surfacearea*globalfreq


def neqtime(time):
    timeMa=time/1.0e6
    timeCorMa=(0.0000000000000544*(exp(0.00693*timeMa)-1)+0.000000838*timeMa)/0.000000838  
    timeCor=timeCorMa*1.0e6 # convert back to Neukum-eq years (not physical years, for the early chronology!)
    
    return timeCor
    
eqtimeton=inversefunc(neqtime)

def cratcalc2(size_km, size_kmupper,lstep):
    epsilon=1.0e-9  #need a small value to get off boundary
    #these are sizes in m,  assuming hohoscaling, and grunfluxes (per km2 per yr)
    grunimpradii_m, grunfluxes=makegruns()
    grundiameters=[hohocratersize(r) for r in grunimpradii_m]    
    grundiameterslog=[log10(el) for el in grundiameters]
    grunfluxeslog=[log10(el) for el in grunfluxes]  
    size=size_km*1000.0
    sizeu=size_kmupper*1000.0
    slog=log10(size)
    sulog=log10(sizeu)
    if (size<10.0):
        if sulog>grundiameterslog[-1]:   #In range where need to interpolate between Neukum at 10m and whatever the maximum Grun diameter is
            interpdiams=[grundiameterslog[-1]-0.5*lstep, 1+0.5*lstep]
            interpfreqs=[cratcalc2(10**(grundiameterslog[-1]-lstep-epsilon-3.0),10**(grundiameterslog[-1]-epsilon-3.0),lstep),(10**log10(npfcalc(0.01))-10**log10(npfcalc(10**(-2.0+lstep))))]
            interpfreqsl=[log10(el) for el in interpfreqs]
            dfluxintlog=np.interp((slog+sulog)/2.0, interpdiams, interpfreqsl)            
            dflux=10**dfluxintlog                       
                       
        elif sulog<=grundiameterslog[-1]:
            fluxintlog=np.interp(slog, grundiameterslog, grunfluxeslog)        
            fluxint=10.0**fluxintlog
            fluxuintlog=np.interp(sulog, grundiameterslog, grunfluxeslog)        
            fluxuint=10.0**fluxuintlog
            dflux=fluxint-fluxuint                     
        else:         
            dflux=np.nan  #this branch only happens if there is a bug.
            
        globalfreq=surfacearea*dflux
    elif size>=10.0:
        globalfreq=surfacearea*(npfcalc(size_km)-npfcalc(size_kmupper))
    else:  
        globalfreq=np.nan #this branch should never happen, unless there is a bug.
    
    return globalfreq

    
def main():
    try:
        try:
            usage = "usage: grunEquil.py\n"
            parser = optparse.OptionParser(usage=usage)
            (options, inargs) = parser.parse_args()
            #if not inargs: parser.error("Example text")    Currently setup for no arguments so this is pointless.
            #firstarg=inargs[0]
            
        except optparse.OptionError, msg:
            raise Usage(msg)            

        esizes=[]
        tfreq=[]
        tenmil=[]
        hunmil=[]
        thomil=[]
        ThrGa=[]
        
        lstep=0.15051499783199/20.0        
        minsizekmlog10=-3-lstep/2.0
        maxsizekmlog10=0.0


        size=minsizekmlog10


        
        while size<maxsizekmlog10:
            sizekm=10.0**size
            sizeupperkm=10.0**(size+lstep)
            esizes.append(1000.0*(10**(size+0.5*lstep)))  #convert km to m        
            oneyearfreqatthissize=cratcalc2(sizekm,sizeupperkm,lstep)
            tenmil.append(oneyearfreqatthissize*1e7)
            hunmil.append(oneyearfreqatthissize*1e8)
            thomil.append(oneyearfreqatthissize*1e9)
            ThrGa.append(oneyearfreqatthissize*3.069396e9)  #three billion year curve has a bit of the exponential
            tfreq.append(numtrask(sizekm,sizeupperkm))
            print 10.0**(size+0.5*lstep)
            
                      
            size=size+lstep
        
        plt.yscale('log')
        plt.xlim([1.0,1000.0])
        plt.ylim([0.01,1e4])
        plt.xscale('log')
        plt.xlabel("Crater Diameter (m)")
        plt.ylabel("Differential freq. (# per $km^2$ per bin)")
        
        plt.scatter(esizes, tenmil, label="10 Ma")
        plt.scatter(esizes, hunmil, label="100 Ma")
        plt.scatter(esizes, thomil, label="1 Ga")
        plt.scatter(esizes, ThrGa, label="3 Ga")
        plt.scatter(esizes, tfreq, label='Trask')
        plt.legend()
        #plt.show()
        plt.savefig('FigS2.png', dpi=300)
        
    except Usage, err:
        print >>sys.stderr, err.msg
        # print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
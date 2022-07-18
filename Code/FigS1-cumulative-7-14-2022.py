import os, optparse, subprocess, sys, tempfile, math
from scalingparams import *
from pynverse import inversefunc  #pip install pynverse  in relevant conda environment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def man(option, opt, value, parser):
    print >>sys.stderr, parser.usage
    print >>sys.stderr, '''\
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
    
    


def makegruns(minlogmassgrams=-18.0,maxlogmassgrams=2.0):
    #Note that the upper valid limit on Grun is 10^2 grams.  Need to solve for bigger particles some other way.
   
    
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
        else:                      
            fluxintlog=np.interp(slog, grundiameterslog, grunfluxeslog)        
            fluxint=10.0**fluxintlog
        globalfreq=surfacearea*fluxint
    else:
        globalfreq=npfcalc(size/1000.0)           
    return globalfreq


def neqtime(time):
    timeMa=time/1.0e6
    timeCorMa=(0.0000000000000544*(exp(0.00693*timeMa)-1)+0.000000838*timeMa)/0.000000838  
    timeCor=timeCorMa*1.0e6 # convert back to Neukum-eq years (not physical years, for the early chronology!)
    
    return timeCor
    
eqtimeton=inversefunc(neqtime)
   
def main():
    try:
        try:
            usage = "usage: gruncheck.py\n"
            parser = optparse.OptionParser(usage=usage)
            (options, inargs) = parser.parse_args()
            #if not inargs: parser.error("Example text")    Currently setup for no arguments so this is pointless.
            #firstarg=inargs[0]
            
        except optparse.OptionError, msg:
            raise Usage(msg)            

        minsizekmlog10=-5
        maxsizekmlog10=2
        lstep=0.01

        size=minsizekmlog10
        sizes=[]
        freqs=[]

        #print "quickcheck:",cratcalc(0.01)  THIS shows the Neukum 10m flux is correct
        grunsize_m, grunflux=makegruns()       
        
        while size<maxsizekmlog10:
            sizekm=10.0**size
            sizeupperkm=10.0**(size+lstep)
            
            sizemid=10.0**(size+lstep/2.0)
            sizes.append(1000.0*sizemid)  #convert km to m
            cumfreq=cratcalc(sizemid)
            freqs.append(cumfreq)            
            size=size+lstep
            print 1000*sizemid, cumfreq
 
        freqsW=[f*37932328.09938 for f in freqs]   
        fig, ax1 = plt.subplots()        
        ax2 = ax1.twinx()
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        plt.xscale('log')
        ax1list=[10.0**i for i in range(-16,7)]
        ax1.set_yticks(ax1list)
        for label in ax1.get_yticklabels()[0::3]:
            label.set_visible(False)
        for label in ax1.get_yticklabels()[2::3]:
            label.set_visible(False)
                

        ax2list=[10.0**i for i in range(-9,15)]
        ax2.set_yticks(ax2list)
        for label in ax2.get_yticklabels()[1::3]:
            label.set_visible(False)
        for label in ax2.get_yticklabels()[2::3]:
            label.set_visible(False)                

        ax1.set_xlabel("Crater Diameter (m)")
        ax1.set_ylabel("Cumulative freq. (# per $km^2$ per yr)")
        ax1.plot(sizes, freqs)
        ax2.plot(sizes, freqsW)
        ax2.set_ylabel('Cumulative freq. (# per Whole Moon per yr)')

        #plt.show()
        plt.savefig('FigS1c.png', dpi=300)
        
    except Usage, err:
        print >>sys.stderr, err.msg
        # print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
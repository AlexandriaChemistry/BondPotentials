#!/usr/bin/env python3

import argparse, copy, glob, json, os, random, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, fmin, root
from diatomics import get_diatomics

from potentials import Kratzer,Harmonic,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Rosen_Morse_try,Linnett,Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun,Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a
#,J1,K1,S0,f_2n

WaveNumber2kJ = 0.0119627

def gen_bounds(pson:dict, diatable:dict, compound:str, func, quiet:bool):
    if not func in pson:
        return None
    mybounds = copy.deepcopy((pson[func]["min"],pson[func]["max"]))
    re   = "re"
    cmpd = compound[:].replace("_", "-")
    if cmpd in diatable and re in pson[func]:
        reval = copy.deepcopy(diatable[cmpd]["distance"])
        reind = copy.deepcopy(int(pson[func][re]))
        # Limit radius but not too much, since there are compounds for
        # which the experimental bond length differs from the QM one.
        mybounds[0][reind] = 0.8*reval
        mybounds[1][reind] = 1.2*reval
    if not quiet:
        print("Bounds for func %s mol %s: %s" % ( func, compound, str(mybounds)))

    return mybounds
    
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def read_potential_parms(filename):

    parms = Vividict()
    # Open the file for reading
    in_file = open(filename,"r")

    # Load the contents from the file, which creates a new dictionary
    parms = json.load(in_file)

    # Close the file... we don't need it anymore  
    in_file.close()

    return parms

def write_json(filename,parms):
    # Open a file for writing
    out_file = open(filename,"w")
    
    # Save the dictionary into this file
    json.dump(parms,out_file, sort_keys=True, indent=2)                                    

    # Close the file
    out_file.close()

def calc_rmsd(func,rkr,popt):
    v = func(rkr[:,0],*popt)
        
    dr = rkr[:,0].max() - rkr[:,0].min()
    rmsd = np.sqrt((((rkr[:,1] - v)**2).sum())/len(rkr[:,1]))

    return(rmsd)
    
    
def calc_Z(func,rkr,popt):
    v = func(rkr[:,0],*popt)
        
    dr = rkr[:,0].max() - rkr[:,0].min()
    Z = ((rkr[:,1] - v)**2).sum() / (rkr[:,1].size * dr)

    return(Z)

def plotit(func, func_name:str, diatomic:str, rkr, 
           popt, porig, bool_output:bool, quiet:bool,
           method:str, minrmsd:float):
    if bool_output:
        output_dir = ( "output/" + method )
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
    v = func(rkr[:,0],*porig)
        
    dr = rkr[:,0].max() - rkr[:,0].min()
    Z = ((rkr[:,1] - v)**2).sum() / (rkr[:,1].size * dr)
    rmsd = calc_rmsd(func,rkr,popt) 
    if rmsd < minrmsd:
        return None
    x0 = np.arange(rkr[:,0].min(), rkr[:,0].max()+0.005, 0.005)
    plt.title(diatomic + " " + func_name + " rmsd=%6.1f\n" % (rmsd))
    plt.plot(rkr[:,0], rkr[:,1], 'ro',label='data')
    plt.plot(rkr[:,0], v, 'g--',label='start')
    
    plt.plot(x0, func(x0,*popt), 'b', label='fit')
    plt.legend()

    output_filename = None
    if bool_output:
        output_filename = output_dir + "/" + diatomic + "_" + func_name + ".pdf"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        if not quiet:
            plt.show(block=False)
            plt.pause(1)
    elif not quiet:
       plt.show()
    plt.close()
    return output_filename

def table(logf, parms, funcs, diatomics):
    parms['Ave'] = {}

    logf.write("%-20s" % ('Species'))
    for func in funcs:
        func_name = func.__name__
        func_name_short = "{0:.15s}".format(func.__name__)
        
        logf.write("%-20s" % (func_name_short))
        parms['Ave'][func_name] = {}
        parms['Ave'][func_name]['Z'] = 0.0
    logf.write("\n")

    logf.write("%-20s" % ('# params'))
    firstkey = diatomics[0]
    for func in funcs:
        func_name = func.__name__

        logf.write("%-20d" % (len(parms[firstkey][func_name]['params'])))
        parms['Ave'][func_name] = {}
        parms['Ave'][func_name]['Z'] = 0.0
    logf.write("\n")

    for diatomic in diatomics:
        logf.write("%-20s" % (diatomic))
        for func in funcs:
            func_name = func.__name__
            
            if diatomic == 'Ave':
                logf.write("%-20.1f" % (parms[diatomic][func_name]['Z']/(len(diatomics)-1),))
            else:
                parms['Ave'][func_name]['Z'] += parms[diatomic][func_name]['Z']
                logf.write("%-20.1f" % (parms[diatomic][func_name]['Z'],))
        logf.write("\n")
    logf.write("\n")

def csv_table(csv:str, parms,funcs,diatomics):
    parms['Ave'] = {}

    outf = open(csv, "w")
    outf.write("%s" % ('Species'))
    for func in funcs:
        func_name = func.__name__
        func_name_short = "{0:.15s}".format(func.__name__)
        
        outf.write(",%s" % (func_name_short))
        parms['Ave'][func_name] = {}
        parms['Ave'][func_name]['Z'] = 0.0
    outf.write("\n")

    outf.write("%s" % ('# params'))
    firstkey = diatomics[0]
    for func in funcs:
        func_name = func.__name__
        outf.write(",%d" % (len(parms[firstkey][func_name]['params'])))
        parms['Ave'][func_name] = {}
        parms['Ave'][func_name]['Z'] = 0.0

    outf.write("\n")

    for diatomic in diatomics:
        outf.write("%s" % (diatomic))
        for func in funcs:
            func_name = func.__name__
            
            if diatomic == 'Ave':
                outf.write(",%16.1f" % (parms[diatomic][func_name]['Z']/(len(diatomics)-1)))
            else:
                parms['Ave'][func_name]['Z'] += parms[diatomic][func_name]['Z']
                outf.write(",%16.1f" % (parms[diatomic][func_name]['Z'],))
        outf.write("\n")
    outf.write("\n")
    outf.close()

def dump_pdfs(pdfs:list, method:str):
    if len(pdfs) == 0:
        return None
    os.chdir("output")
    if method[-4:] == ".pdf":
        method = method[:-4]
    tex = ( "%s.tex" % method )
    with open(tex, "w") as outf:
        outf.write("""\\documentclass{book}
\\usepackage{graphicx}
\\setlength{\\textwidth}{165mm}
\\begin{document}
""")
        for p in range(0, len(pdfs), 9):
            if p % 27 == 0:
                outf.write("\\chapter{Next}\n")
            outf.write("\\begin{figure}[tb]\n")
            for n in range(3):
                n0 = p+3*n
                for m in range(n0, min(n0+3, len(pdfs))):
                    # Dirty hack to strip of the directory name
                    outf.write("\\includegraphics[width=50mm]{%s}" % pdfs[m][7:])
                    if m % 3 == 2:
                        outf.write("\\\\\n")
                    else:
                        outf.write("\n")
            outf.write("\\end{figure}\n\n")
                
        outf.write("\\end{document}\n")
    os.system("pdflatex '%s' > latex.out" % tex)
    pdffn = ("%s.pdf" % method)
    if os.path.exists(pdffn):
        os.system("mv '%s' .." % pdffn)
    os.chdir("..")
    return pdffn
    
def reset_z(parms):
    for comp in parms:
        for function in parms[comp]:
            if "Z" in parms[comp][function]:
                parms[comp][function]["Z"] = 1e20

def get_args(diatomics:list, functions:list):
    parser = argparse.ArgumentParser(
        description='Curvefit 28 potentials to data sets from quantum chemistry and/or experiment.',
    )
    parser.add_argument('-o', action="store_true", default=False,help='Write data and fitted functions to pdf in a directory output')
    parser.add_argument('-s', action="store_true", default=False,help='Plot data and fitted function')
    parser.add_argument('-q','--quiet', action="store_true", help="Do not show graphs on the screen.")
    defpdf = "cfit"
    parser.add_argument("-pdf", type=str, default=defpdf, help="Name for pdf output. This turns on the -o flag. Default "+defpdf)
    parser.add_argument('-ufb', '--use_func_bounds', action="store_true", default=False,help='Use the function bounds in json/potentials.json')
    parser.add_argument('-m', default=False,help='Fit one molecule only, choose one of: ' + ' '.join(diatomics) + ' or choose from those in data/diatomics.csv')
    parser.add_argument('-f', default=False,help='Fit one function only, choose one of: ' + ' '.join([x.__name__ for x in functions]))
    parser.add_argument('-w', dest="filename", metavar='file', required=False, default="curve_fit.json",help='Write json file with parameters and Z values')
    defmeth = "exp"
    parser.add_argument('-method', type=str, default=defmeth, help='Method to use, pick one from exp, MP2, CCSD(T), default '+defmeth)
    parser.add_argument('-rmin', default=False, help='Print the minimum of the potential')
    parser.add_argument('-numax', default=0, type=float, help='Maximum energy (cm^-1) to take into account. 0 means all data')
    parser.add_argument('-random', action="store_true", help="Add a small (5%%) random change to the input parameters")
    defalg = 'Nelder-Mead'
    parser.add_argument('-alg', type=str, default=defalg, help='SciPy algorithm to minimize, default '+defalg)
    minrmsd = 0
    parser.add_argument('-minrmsd', type=float, default=minrmsd, help='Minimum RMSD (wavenumbers) to plot a graph, 0 means plot all, a higher number speeds things up and allows you to check for outliers. Default '+str(minrmsd))
    parser.add_argument('-resetZ', action="store_true", help='Set the Z values after reading to the value corresponding to the input parameters in the json file.')
    return parser.parse_args()

def main():
    diatomics = ['H2','OH','O2','NO','N2+','N2','LiH','Li2','HF','C2','CF','CO','CN', 'CH']
    functions = [Kratzer,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Linnett,Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun,Harmonic,Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a]

    args = get_args(diatomics, functions)
    if args.pdf:
        args.o = True
    diatable = get_diatomics()

    if args.use_func_bounds:
        pfile = "json/potentials.json"
        with open(pfile, "r") as inf:
            pson = json.load(inf)
        print("Successfully read %s with %d potentials" % ( pfile, len(pson.keys())))

    if args.method in [ "MP2", "CCSD(T)" ]:
        os.chdir("data/%s" % args.method)
        dfiles = glob.glob("*.xvg")
        os.chdir("../..")
        jsfile = ("json/%s-numax=%g.json" % (args.method, args.numax))
        if not os.path.exists(jsfile):
            jsfile = ("json/%s-numax=0.json" % args.method)
        parms = read_potential_parms(jsfile)
        diatomics = list(parms.keys())
        for dfile in dfiles:
            ddfile = dfile[:]
            # Replace dash and remove extension
            ddfile = ddfile.replace('-', '_')[:-4]
            if not ddfile in parms.keys():
                cmono = "carbon_monoxide"
                parms[ddfile] = copy.deepcopy(parms[cmono])
                parms[ddfile]['filename'] = dfile
                sys.stderr.write("Copying %s parameters to %s\n" % ( cmono, ddfile ))
    elif args.method == "exp":
        parms = read_potential_parms("json/Royappa2006a-JMS_787_209.json")
    else:
        sys.exit("Unsupported method '%s'" % args.method)
    if args.m:
        diatomics = [ args.m ]

    if len(diatomics) == 0:
        sys.exit("No diatomics, check your command line")
    # This hack could be useful for other future potentials, so leaving it in.
    for d in diatomics:
        if d in parms and "Kratzer" in parms[d] and not "Harmonic" in parms[d]:
            parms[d]["Harmonic"] = parms[d]["Kratzer"]

    if args.f:
        functions = [ eval(args.f) ]

    logfilename = args.method + ".log"
    logf = open(logfilename, "w")
    pdf_files = []
    for diatomic in diatomics:
        if args.quiet:
            print(" %s" % diatomic)
        # Fetch the data
        fnm = 'filename'
        if not fnm in parms[diatomic]:
            sys.stderr.write("No filename for parms[%s]\n" % diatomic)
            continue
        file = ( "data/%s/%s" % ( args.method, parms[diatomic][fnm] ))
        sys.stderr.write("Looking for %s\n" % file)
        if os.path.isfile(file) and os.access(file, os.R_OK):
            rkr = np.loadtxt(file, comments=["@", "#"])
        else:
            sys.stderr.write('[!] data file: %s missing!\n' % file)
            del parms[diatomic]
            diatomics.remove(diatomic)
            continue
        # Check whether we need to skip high energies
        if args.numax > 0:
            # Convert input energy treshold to wavenumbers
            numax = args.numax
            rlen = rkr.shape[0]
            rkr  = np.delete(rkr, np.where(rkr[:,1] > args.numax)[0], 0)
            if rkr.shape[0] < rlen and not args.quiet:
                print("%d/%d data points left for %s below %.1f cm^-1" % ( rkr.shape[0], rlen, diatomic,numax ))
        # Loop over functions to fit
        for func in functions:
            func_name = func.__name__
            func_name_short = "{0:.15s}".format(func.__name__)

            psql = np.array(parms[diatomic][func_name]['params'])
            psql_orig = psql[:]
            if rkr.shape[0] < psql.shape[0]:
                print("Too few data points (%d) left for %s to fit %s with %d parameters" % ( rkr.shape[0], diatomic, func_name, psql.shape[0] ))
                continue
            # Compute in-going Z
            Zold = calc_Z(func,rkr,psql)
            result = ('[+]', diatomic, func_name,'in  [', ','.join(map(str,  psql_orig)), ']', 'Zjson=', parms[diatomic][func_name]["Z"], 'Zcalc=', str(Zold))
            # Now apply bounds
            mybounds = None
            if args.use_func_bounds:
                mybounds = gen_bounds(pson, diatable, diatomic, func_name, args.quiet)
                if mybounds:
                    # Put starting value within bounds
                    if len(psql) != len(mybounds[0]):
                        sys.exit("len(psql) = %d len(mybounds[0]) = %d for %s" % ( len(psql), len(mybounds[0]), func_name))

                    pchanged = False
                    for p in range(len(psql)):
                        if args.random:
                            psql[p] *= 0.95 + 0.1*random.random()
                        paver = (mybounds[0][p]+mybounds[1][p])*0.5
                        if psql[p] < mybounds[0][p]:
                            psql[p] = paver
                            pchanged = True
                        elif psql[p] > mybounds[1][p]:
                            psql[p] = paver
                            pchanged = True
                    if args.random or pchanged:
                        Zbounded = calc_Z(func,rkr,psql)
                        result += ('\n[+]', diatomic, func_name,'bounded [', ','.join(map(str,  psql)), ']', 'Zbounded=', str(Zbounded))
            if Zold < parms[diatomic][func_name]["Z"] and args.resetZ:
                parms[diatomic][func_name]["Z"] = Zold

            for r in result:
                logf.write("%s " % ( r ) )
            logf.write("\n")
            is_z_lower = "N"    

            try:
                sys.stderr.write("Diatomic %s function %s\n" % ( diatomic, func_name ))
                
                if mybounds:
                    popt, pcov = curve_fit(func, rkr[:,0], rkr[:,1], p0=psql,maxfev=1000000, bounds=mybounds)
                else:
                    popt, pcov = curve_fit(func, rkr[:,0], rkr[:,1], p0=psql,maxfev=1000000, ftol=1e-12, xtol=1e-12)

                rmsd  = calc_rmsd(func,rkr,popt)
                Z     = calc_Z(func,rkr,popt)
                del_Z = Z - parms[diatomic][func_name]["Z"]
                if del_Z:
                    parms[diatomic][func_name]["params"] = popt.tolist()
                    parms[diatomic][func_name]["Z"] = Z
                    parms[diatomic][func_name]["RMSD"] = rmsd
                    is_z_lower = "Y"

                kkk = ('[+]', diatomic, func_name, 'opt', popt.tolist(), 'Z=', Z, 'del_Z=', del_Z, 'rmsd=', rmsd, is_z_lower)
                for k in kkk:
                    logf.write("%s " % k)
                logf.write("\n")

                if args.rmin:
                    rm = float(args.rmin)
                    min = minimize(func, rm, args=tuple(popt), method=args.alg, options={ 'disp': False})
                    r0 = min.x[0]
                    lll = ("[+] %s %s rmin: %.5f %.5f (%f)" % (diatomic, func_name, r0, func(r0, *popt), rm))
                    logf.write("%s\n" % lll)
                    
                if args.s or args.o:
                    pdf = plotit(func, func_name, diatomic, rkr, popt, psql_orig, args.o, args.quiet, args.method, args.minrmsd)
                    if None != pdf:
                        pdf_files.append(pdf) 

            except RuntimeError:
                print("curve_fit failed")
    if args.filename:
        jsonname = write_json(args.filename,parms)
    pdfname = ""
    if args.o:
        pdfname = dump_pdfs(pdf_files, args.pdf)

    diatomics.append('Ave')
    func1 = [Kratzer,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Linnett]
    table(logf, parms,func1, diatomics)
    func2 = [Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun]
    table(logf, parms,func2, diatomics)
    func3 = [Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a]
    table(logf, parms,func3, diatomics)
    logf.close()

    csvname = args.method+".csv"
    csv_table(csvname, parms, func1+func2+func3, diatomics)
    
    print("\nPlease check output in %s, %s, %s, %s" %
          ( logfilename, csvname, pdfname, args.filename ) )

if __name__ == "__main__":
    cfe = "cf_errors.txt"
    sys.stderr = open(cfe, "w")
    print("Please check %s if there is no output" % cfe)
    main()
    sys.stderr.close()

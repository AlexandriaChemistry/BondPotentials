#!/usr/bin/env python3

import numpy as np
import math, psi4
import argparse, os, sys, json
from diatomics import get_diatomics,get_moldata
from potentials import AU2INVCM,Kratzer,Harmonic,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Rosen_Morse_try,Linnett,Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun,Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a
from curve_fit import read_potential_parms

verbose  = False
exp      = "Experiment"
ref      = "Reference"
expsum   = "expsum"
exp2sum  = "exp2sum"
ccsdt    = "CCSD(T)"
methods  = [ ccsdt, "MP2" ]
ev2kJ    = 96.485332
cm2kJ    = 0.0119627
    
def to_latex(dd:str, mult:int)->str:
    ndd = ""
    for k in dd:
        if k in '0123456789':
            ndd += "$_" +k + "$"
        elif k in '+-':
            ndd += "$^" +k + "$"
        else:
            ndd += k
#    for m in range(1,mult):
#        ndd += "{\\textbullet}"
    return ndd

def get_range(form:str):
    cfn = ( "data/CCSD(T)/%s.xvg" % form )
    r0  = None
    r1  = None
    n   = 0
    if os.path.exists(cfn):
        with open(cfn, "r") as inf:
            for line in inf:
                if line.startswith("@"):
                    continue
                words = line.split()
                if len(words) == 2:
                    try:
                        x = float(words[0])
                        y = float(words[1])
                        if not r0:
                            r0 = x
                        if not r1:
                            r1 = x
                        r0 = min(x, r0)
                        r1 = max(x, r0)
                        n += 1
                    except ValueError:
                        print("Cannot read line '%s'" % line.strip())
                        continue
    else:
        sys.exit("No such file %s" % cfn)
    return r0, r1, n
                            
def dtable(diat:dict, md:dict, filenm:str):
    ccc = { False: "non-covalent", True: "covalent" }
    zpemax = 0
    with open(filenm, "w") as outf:
        outf.write("\\begin{longtable}{p{40mm}cccccccc}\n")
        outf.write("\\caption{Name and properties (charge q, multiplicity m) of the  diatomics studied. Covalency is indicated by cov. Equilibrium distance r$_{eq}$~\\cite{Huber1979a}, range of the quantum chemical scan (r$_0$, r$_1$ in {\\AA}) and number of points N in the scan.}\\\\\n")
        outf.write("\\hline\n")

        head = ("Compound & Formula & q & m & cov & r$_0$ & r$_{eq}$ & r$_1$ & N \\\\")
        outf.write("%s\n" % head)
        outf.write("\\hline\n")
        outf.write("\\endfirsthead\n")
        outf.write("%s\n" % head)
        outf.write("\\hline\n")
        outf.write("\\endhead\n")
        outf.write("\\hline\n")
        outf.write("\\endfoot\n")
        outf.write("\\hline\n")
        outf.write("\\endlastfoot\n")
        for d in diat.keys():
            dd = d.replace("-", " ")
            r0, r1, N = get_range(diat[d]["formula"])
            req       = 0
            for m in md.keys():
                if (diat[d]["formula"] == m[0] and
                    md[m]["Te cm^{-1}"] == 0.0):
                    req = md[m]["Re \AA"]
            outf.write("%s & %s & %d & %d & %s & %g & %g & %g & %d" % ( dd, to_latex(diat[d]["formula"], diat[d]["mult"]), diat[d]["charge"], diat[d]["mult"], diat[d]["covalent"], r0, req, r1, N ))
            D0 = None
            we = None
            re = None
            outf.write("\\\\\n")
        outf.write("\\end{longtable}\n")
        outf.write("\n")
    print("Largest ZPE %g kJ/mol" % zpemax)

def read_data(filenm:str, numax:float):
    rvals = []
    energies = []
    nskip = 0
    with open(filenm, "r") as inf:
        nline = 0
        for line in inf:
            nline += 1
            nhack = line.find("@")
            if nhack >= 0:
                line = line[:nhack]
            nh2 = line.find("#")
            if nh2 >= 0:
                line = line[:nh2]
            words  = line.strip().split()
            if not len(words) == 2:
                continue
            try:
                rval = float(words[0])
                ener = float(words[1])
                if numax <= 0 or ener < numax:
                    rvals.append(rval)
                    energies.append(ener/AU2INVCM)
                else:
                    nskip += 1
            except ValueError:
                print("Line %d in file %s not comprehensible" % ( nline, filenm ))
                continue
    if nskip > 0 and verbose:
        print("Skipped %d/%d energies from %s" % ( nskip, (nskip+len(rvals)), filenm ))
    return rvals, energies

def compute_energies(method:str, rvals:list, params:list)->list:
    energies = []
    func     = eval(method)
    for r in rvals:
        energies.append(func(r, *params)/AU2INVCM)
    return energies

def compute_freq(rvals:list, energies:list, atoms:list, charge:int, mult:int)->dict:
    rmin = rvals[0]
    emin = energies[0]
    for r in range(len(rvals)):
        if energies[r] < emin:
            emin = energies[r]
            rmin = rvals[r]
    mol_tmpl = ("""
    %d %d
    %s 0 0 0
    %s 0 0 %g
""" % ( charge, mult, atoms[0], atoms[1], rmin) )
    molecule = psi4.geometry(mol_tmpl)
    return psi4.diatomic.anharmonicity(rvals, energies, mol=molecule)

def calc_stats(props:dict, ref:dict, calc:dict, percent:bool, method:str)->dict:
    msd  = {}
    for p in props:
        msd[p]  = 0
        nmsd = 0
        for diat in ref:
            if diat in calc:
                if p in ref[diat] and p in calc[diat]:
                    if math.isfinite(calc[diat][p]) and math.isfinite(ref[diat][p]) and ref[diat][p] > 0:
                        if percent:
                            ratio   = calc[diat][p]/ref[diat][p]
                            msd[p] += (1-ratio)**2
                            if abs(ratio) > 8:
                                print("Suspect compound %s property %s ref %g calc %g ratio %g method %s"
                                      % ( diat, p, ref[diat][p], calc[diat][p], ratio, method ) )
                        else:
                            msd[p] += (ref[diat][p]-calc[diat][p])**2

                        nmsd   += 1

        if nmsd >  1:
            msd[p] = math.sqrt(msd[p]/(nmsd-1))
            if percent:
                msd[p] = min(100, 100*msd[p])

    return msd

def write_qm_freqs(filenm:str, props:dict, mydict:dict, diat:dict):
    with open(filenm, "w") as outf:
        outf.write("\\begin{landscape}\n")
        outf.write("\\begin{longtable}{p{20mm}")
        for p in props:
            outf.write("ccc")
        outf.write("}\n")
        outf.write("\\caption{")
        for p in props:
            outf.write("%s (%s), " % ( props[p]["desc"], props[p]["unit"] ) )
        outf.write(" from experiment~\\cite{Huber1979a} and quantum chemistry (this work).}\\\\\n")
        outf.write("\\hline\n")
        nprops = len(props)
        head = ("Compound ")
        for m in range(3):
            for p in props:
                head += ( "& %s" % props[p]["short"] )
        head += "\\\\"
        outf.write("&\multicolumn{%d}{c}{Experiment}&\multicolumn{%d}{c}{CCSD(T)}&\multicolumn{%d}{c}{MP2}\\\\\n" % ( nprops, nprops, nprops ) )
        outf.write("%s\n" % head)
        outf.write("\\hline\n")
        outf.write("\\endfirsthead\n")
        outf.write("%s\n" % head)
        outf.write("\\hline\n")
        outf.write("\\endhead\n")
        outf.write("\\hline\n")
        outf.write("\\endfoot\n")
        outf.write("\\hline\n")
        outf.write("\\endlastfoot\n")
        for d in sorted(diat.keys()):
            results = []
            count   = 0
            for m in [ ref, methods[0], methods[1] ]:
                if not m in mydict:
                    continue
                count += 1
                for ww in props:
                    if d in mydict[m] and ww in mydict[m][d] and math.isfinite(mydict[m][d][ww]):
                        results.append(props[ww]["format"] % mydict[m][d][ww])
                    else:
                        results.append("")
            outf.write("%s" % to_latex(diat[d]["formula"], diat[d]["mult"]))
            for r in range(len(results)):
                outf.write("& %s" % results[r])
            outf.write("\\\\\n")
        outf.write("\\end{longtable}\n")
        outf.write("\\end{landscape}\n")
        outf.write("\n")

def write_rmsd(outf, props:dict, rmsd:dict):
    for p in props:
        if rmsd[p] == 100:
            outf.write("& 100")
        else:
            outf.write("& %.2f" % ( rmsd[p] ) )

def write_stats(args, mydict:dict, props:dict, methods:list, pots:list):
    fstats = ( "freq_stats_numax=%d.tex" % args.numax )
    if args.numax == 0:
        cutoff = "CCSD(T) without energy threshold"
    elif args.numax == -1:
        cutoff = "experimental data without energy threshold"
    else:
        cutoff = ("CCSD(T) with an energy threshold of %d cm$^{-1}$" % args.numax)
    
    sorting = ""
    if args.numax == 1000:
        sorting = " Potentials sorted according to deviation of $\\omega_e$ from CCSD(T)."
    with open(fstats, "w") as outf:
        outf.write("\\begin{table*}[htb]\n")
        outf.write("\\centering\n")
        if args.percent:
            outf.write("\\caption{Percent deviation for")
        else:
            outf.write("\\caption{Root mean square deviation for")
        for p in props:
            outf.write(" %s, " % props[p]["desc"])
        outf.write(" for different methods and analytical potentials fitted on %s, from the reference indicated below.%s}\n" % ( cutoff, sorting) )
        outf.write("\\label{fstats%d}\n" % args.numax)
        nprops = len(props)
        outf.write("\\begin{tabular}{l")
        if args.numax == -1:
            ncol = 1
            for i in range(nprops):
                outf.write("c")
            outf.write("}\n\\hline\n")
            outf.write("\\multicolumn{1}{r}{Reference:} & \\multicolumn{%d}{c}{%s}\\\\\n" % ( nprops, exp ) )
        else:
            ncol = 2
            for j in range(ncol):
                for i in range(nprops):
                    outf.write("c")
            outf.write("}\n\\hline\n")
            outf.write("\\multicolumn{1}{r}{Reference:} & \\multicolumn{%d}{c}{%s} & \\multicolumn{%d}{c}{CCSD(T)}\\\\\n" % ( nprops, exp, nprops ) )
            ncol = 2
        outf.write("Method ")
        for j in range(ncol):
            for p in props:
                outf.write(" & %s " % props[p]["short"])
        outf.write("\\\\\n")
        outf.write("\\hline\n")
        
        ccsdt = "CCSD(T)"
        mp2   = "MP2"
        mrefs = [ ref ]
        
        rmsd = calc_stats(props, mydict[ref], mydict[ccsdt], args.percent, ccsdt)
        if args.numax >= 0:
            outf.write("%s " % ccsdt)
            write_rmsd(outf, props, rmsd)
            mrefs.append(ccsdt)
            outf.write("\\\\\n")

        if mp2 in mydict:
            outf.write("%s" % mp2)
            for mref in mrefs:
                rmsd = calc_stats(props, mydict[mref], mydict[mp2], args.percent, mp2)
                write_rmsd(outf, props, rmsd)
            outf.write("\\\\\n")
        outf.write("\\hline\n")

        spots = []
        for p in pots:
            if p in mydict:
                rmsd = calc_stats(props, mydict[ccsdt], mydict[p], args.percent, p)
                spots.append( ( p, rmsd ) )
            else:
                print("Function %s missing from mydict" % p)
        for p in sorted(spots, key=lambda tup: tup[1]["we"]):
            m = p[0]
            if args.numax == -1:
                mrefs = [ ref ]
            outf.write("%s " % m.replace("_", "-"))
            for mref in mrefs:
                rmsd = calc_stats(props, mydict[mref], mydict[m], args.percent, m)
                if m == "Harmonic":
                    for p in props:
                        if p == "we":
                            outf.write("& %.2f" % ( rmsd[p] ) )
                        else:
                            outf.write(" & -" )
                else:
                    write_rmsd(outf, props, rmsd)
            if len(mrefs) == 1 and args.numax >= 0:
                outf.write("&&")
            outf.write("\\\\\n")
        outf.write("\\hline\n")
        outf.write("\\end{tabular}\n")
        outf.write("\\end{table*}\n")

    return fstats

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute frequencies, make plots and tables.',
    )
    parser.add_argument('-v', "--verbose", action="store_true", default=False, help='Write info to the screen')
    parser.add_argument('-numax', default=0, type=int, help='Maximum energy (cm^-1) to take into account. 0 means all data. -1 means experimental data.')
    parser.add_argument('-p', "--percent", action="store_true", default=False, help='Print RMSD tables in percentage')

    return parser.parse_args()

def write_csv(freqcsv:str, verbose:bool, ddd:dict, mydict:dict)->dict:
    stats = {}
    with open(freqcsv, "w") as outf:
        outf.write("Compound")
        for prop in props.keys():
            outf.write(",%s(Exper)" % ( prop ) )
            stats[prop] = {}
        for method in methods+pots:
            for prop in props.keys():
                outf.write(",%s(%s),%s(%s)-%s(Exper)" % ( prop, method, prop, method, prop ))
        outf.write("\n")

        for diat in ddd.keys():
            outf.write("%s" % diat)
            if verbose:
                print("Printing for %s : %s" % ( diat, str(mydict[ref][diat]) ) )
            for m in [ exp ] + methods + pots:
                for prop in props.keys():
                    if not m in stats[prop]:
                        stats[prop][m] = { chi2: [] }
                    wexp = None
                    if diat in mydict[ref] and prop in mydict[ref][diat]:
                        wexp = mydict[ref][diat][prop]
                        if not math.isfinite(wexp):
                            outf.write(",")
                            wexp = None
                        else:
                            outf.write(",%g" % ( wexp ))

                    if not m == exp:
                        if m in mydict and diat in mydict[m]:
                            outf.write(",%g" % ( mydict[m][diat][prop] ) )
                            if wexp:
                                xvg[prop][m].write("%10g  %10g\n" % ( wexp, mydict[m][diat][prop] ))
                                diff = mydict[m][diat][prop]-wexp
                                stats[prop][m][chi2].append(diff**2)
                                outf.write(",%g" % ( diff ))
                            else:
                                outf.write(",")
                        else:
                            outf.write(",,")

            outf.write("\n")
    return stats
    
def write_exptab(props:dict, ddd:dict, mydict:dict):
    outf = open("exper_freq.tex", "w")
    outf.write("\\begin{table}[ht]\n")
    outf.write("\\centering\n")
    outf.write("\\caption{Vibrational properties:")
    for p in props:
        outf.write(" %s (%s)" % ( props[p]["desc"], props[p]["unit"] ) )
    outf.write(" from experiment~\cite{Huber1979a} or computed using Psi4~\cite{psi4} based on experimental data. Root Mean Square Deviation (RMSD) and Mean Signed Error (MSE) are given at the bottom.}\n")
    outf.write("\\label{expfreqs}\n")
    nprops = len(props)
    outf.write("\\begin{tabular}{l")
    for i in range(2*nprops):
        outf.write("c")
    outf.write("}\n\\hline\n")
    outf.write("Molecule ")
    for p in props:
        outf.write("& \multicolumn{2}{c}{%s}" % ( props[p]["short"] ) )
    outf.write("\\\\\n")
                    
    for i in range(nprops):
        outf.write("& Exper & Psi4 ")
    outf.write("\\\\\n")
    outf.write("\\hline\n")

    nexpsum = {}
    stats   = {}
    for prop in props.keys():
        stats[prop]   = { expsum: 0.0, exp2sum: 0.0 }
        nexpsum[prop] = 0

    for diat in ddd:
        if (diat in mydict[method] and diat in mydict[ref] and
            prop in mydict[method][diat] and prop in mydict[ref][diat]):
            ddname = to_latex(ddd[diat]["formula"], ddd[diat]["mult"])
            outf.write(ddname)
            for prop in props:
                expval  = mydict[ref][diat][prop]
                calcval = mydict[method][diat][prop]
                if math.isfinite(calcval) and math.isfinite(expval):
                    diff   = calcval - expval
                    stats[prop][expsum]  += diff
                    stats[prop][exp2sum] += diff**2
                    nexpsum[prop] += 1
                for val in [ expval, calcval ]:
                    if math.isfinite(val):
                        format = "&" + props[prop]["format"]
                        outf.write(format % val)
                    else:
                        outf.write("&")
            outf.write("\\\\\n")
    outf.write("\\hline\n")
    
    outf.write("RMSD  ")
    for ww in props.keys():
        if nexpsum[ww] > 0:
            rmsd = (stats[ww][exp2sum]/nexpsum[ww]-(stats[ww][expsum]/nexpsum[ww])**2)**(0.5)
            format = (" & & %s " % props[ww]["format"] )
            outf.write(format % rmsd)
        else:
            outf.write(" & & - ")
    outf.write("\\\\\n")
    outf.write("MSE  ")
    for ww in props.keys():
        if nexpsum[ww] > 0:
            format = (" & & %s " % props[ww]["format"] )
            outf.write(format % ( stats[ww][expsum]/nexpsum[ww] ) )
        else:
            outf.write(" & & - ")
    outf.write("\\\\\n")
    outf.write("\\hline\n")
    outf.write("\\end{tabular}\n\\end{table}\n")
    outf.close()

def fetch_reference(ddd:dict, props:dict, verbose:bool, dtex:str)->dict:
    mdfile  = "data/Diatomic_Moleculedata.csv"
    md      = get_moldata(mdfile)
    mdfile2 = "data/Diatomic_Moleculedata_updates.csv"
    dscdm   = get_moldata(mdfile2)
    md.update(dscdm)
    print("There are %d entries in %s and %s" % ( len(md), mdfile, mdfile2 ))
    dtable(ddd, md, dtex)
    mydict = { ref: {} }
    for diat in ddd.keys():
        mydict[ref][diat] = {}
        for m in md:
            if (ddd[diat]["formula"] == m[0] and
                md[m]["Te cm^{-1}"] == 0.0):
                for prop in props:
                    key = props[prop]["key"]
                    if key in md[m]:
                        mydict[ref][diat][prop] = md[m][key]
                        if "factor" in props[prop]:
                            mydict[ref][diat][prop] *= props[prop]["factor"]
                    else:
                        mydict[ref][diat][prop] = None
                    if verbose:
                        print("Found experiment for %s = %s" % ( diat, str(mydict[ref][diat]) ) )
    return mydict

if __name__ == "__main__":
    args = parse_args()

    ddd = get_diatomics()
    print("There are %d diatomics" % ( len(ddd) ))

    plog = "psi4.log"
    psi4.core.set_output_file(plog, False)
    pots    = []
    for p in [ Kratzer,Harmonic,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Linnett,Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun,Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a ]:
        pots.append(str(p.__name__))

    chi2    = "chi2"
    props = { "we":   { "key": "omega_e cm^{-1}", "format": "%.1f", "unit": "1/cm",
                        "desc": "Ground state frequency $\\omega_e$", "short": "$\\omega_e$" },
              "wexe": { "key": "omega_ex_e cm^{-1}", "format": "%.2f", "unit": "1/cm",
                        "desc": "Anharmonic frequency $\\omega_e$x$_e$", "short": "$\\omega_e$x$_e$" },
              "Be":   { "key": "Be cm^{-1}", "format": "%.2f", "unit": "1/cm",
                        "desc": "Rotational constant in equilibrium position B$_e$", "short": "B$_e$" },
              "ae":   { "key": "alpha_e cm^{-1}", "format": "%.2e", "unit": "1/cm",
                        "desc": "Rotational constant - first term $\\alpha_e$", "short": "$\\alpha_e$" },
              "De":   { "key": "De 10^{-7}cm^{-1}", "format": "%.2e", "unit": "1/cm",
                        "desc": "Centrifugal distortion constant D$_e$", "short": "D$_e$", "factor": 1e-7 }
             }
             
    xvg     = {}
    for prop in props.keys():
        xvg[prop]     = {}
        for m in methods+pots:
            fnm = ( "%s-%s.xvg" % ( prop, m ))
            xvg[prop][m] = open(fnm, "w")
            xvg[prop][m].write("@xaxis label \"%s Experiment\"\n" % prop)
            xvg[prop][m].write("@yaxis label \"QM - Experiment\"\n")

    # Look up experimental reference data. Note that it needs to be indexed with "ref", not "exp".
    # The latter is for numbers generated by Psi4 based on the experimental data.
    dtex   = "diat_table.tex"
    mydict = fetch_reference(ddd, props, args.verbose, dtex)

    # Fetch or compute theoretical data
    jparms = {}
    for method in methods:
        # Read parameters for this method
        if args.numax == -1:
            filename = "json/exp.json"
        else:
            filename = ("json/%s-numax=%d.json" % ( method, args.numax ))
        mydict[method] = {}
        jparms[method] = read_potential_parms(filename)
        for diat in ddd.keys():
            atoms  = [ ddd[diat]["ai"], ddd[diat]["aj"] ]
            charge = ddd[diat]["charge"]
            mult   = ddd[diat]["mult"]
            rvals    = None
            energies = None
            # Read existing file
            compound = ddd[diat]['formula']
            if not compound in jparms[method]:
                continue
            filenm = ("data/%s/%s" % ( method, jparms[method][compound]['filename']))
            if not os.path.exists(filenm):
                print("Cannot find %s" % filenm)
                continue

            rvals, energies = read_data(filenm, args.numax)
            # Now try and compute frequencies, need at least 5 for Psi4 algorithm.
            if len(rvals) > 4:
                try:
                    mydict[method][diat] = compute_freq(rvals, energies, atoms, charge, mult)
                except:
                    print("Problem computing frequencies")
                    continue

                if method == ccsdt or args.numax == -1:                          
                    for pot in pots:
                        # Compute them
                        energies = compute_energies(pot, rvals, jparms[method][compound][pot]['params'])
                        if args.verbose:
                            print("Computed %d energies for %s using %s and %s. First %g" % ( len(energies), diat, method, pot, energies[0] ))
                        try:
                            if not pot in mydict:
                                mydict[pot] = {}
                            mydict[pot][diat] = compute_freq(rvals, energies, atoms, charge, mult)
                            if args.verbose and pot == "Wei_Hua":
                                print("%s %s %s" % ( diat, pot, str(mydict[pot][diat]) ) )
                        except:
                            print("Something weird with %s for %s with %d data points" % ( pot, compound, len(rvals) ))
            else:
                print("Not enough data points (%d) in %s" % ( len(rvals), filenm ))
        if args.numax == -1:
            break

    # Collected all data
    # Check energies at minimum. TODO: does this work?
    for m in mydict:
        for d in mydict[m]:
            ere = "E(re)"
            if ere in mydict[m][d] and mydict[m][d][ere] < -0.01:
                print("Energy at minimum %g/cm for %s and %s " % (mydict[m][d][ere]*AU2INVCM, m, d))

    freqcsv  = "frequencies.csv"
    stats = write_csv(freqcsv, args.verbose, ddd, mydict)
    write_qm_freqs("qm_freqs.tex", props, mydict, ddd)
    if args.numax == -1:
        write_exptab(props, ddd, mydict)
    fstats = write_stats(args, mydict, props, methods, pots)

    for md in props.keys():
        xvg[md] = {}
        for m in methods+pots:
            if m in xvg[md]:
                xvg[md][m].close()
    
    print("Please check %s, %s, %s and %s" % ( freqcsv, dtex, fstats, plog ))

#!/usr/bin/env python3

import numpy as np
import math, psi4
import argparse, os, sys, json
from diatomics import get_diatomics,get_moldata
from potentials import AU2INVCM,Kratzer,Harmonic,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Rosen_Morse_try,Linnett,Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun,Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a
from curve_fit import read_potential_parms

verbose  = False
wetex    = "$\\omega_e$"
wexetex  = "$\\omega_e$x$_e$"
exp      = "Experiment"
methods  = [ "CCSD(T)", "MP2" ]
ev2kJ    = 96.485332
cm2kJ    = 0.0119627
    
def to_latex(dd:str, radical:bool)->str:
    ndd = ""
    for k in dd:
        if k in '0123456789':
            ndd += "$_" +k + "$"
        else:
            ndd += k
    if radical:
        ndd += "{$\\cdot$}"
    return ndd

def dtable(diat:dict, md:dict, filenm:str):
    ccc = { False: "non-covalent", True: "covalent" }
    zpemax = 0
    with open(filenm, "w") as outf:
        outf.write("\\begin{longtable}{p{40mm}cccc}\n")
        outf.write("\\caption{Name and properties (charge q, multiplicity m) of the  diatomics studied. Covalency is indicated by cov.}\\\\\n")# Experimental dissociation energy $D_0$ (cm$^{-1}$) and zero point energy ZPE (cm$^{-1}$) and bond distance r$_e$ (\AA)~\\cite{Huber1979a}.}\\\\\n")
        outf.write("\\hline\n")
        #& D$_0$ & ZPE & r$_e$
        head = ("Compound & Formula & q & m & cov \\\\")
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
            radical = diat[d]["mult"] > 1
            outf.write("%s & %s & %d & %d & %s" % ( d, to_latex(diat[d]["formula"], radical), diat[d]["charge"], diat[d]["mult"], diat[d]["covalent"] ))
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

def compute_them(method:str, rvals:list, params:list)->list:
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

def calc_stats(ref:dict, calc:dict):
    msd_we   = 0
    msd_wexe = 0
    nwe      = 0
    nwexe    = 0
    for i in ref:
        if not i in calc:
            continue
        if "we" in ref[i] and "we" in calc[i]:
            if math.isfinite(calc[i]["we"]):
                msd_we   += (ref[i]["we"]-calc[i]["we"])**2
                nwe      += 1
        if "wexe" in ref[i] and "wexe" in calc[i]:
            if math.isfinite(ref[i]["wexe"]) and math.isfinite(calc[i]["wexe"]):
                msd_wexe += (ref[i]["wexe"]-calc[i]["wexe"])**2
                nwexe    += 1
    rmsd_we = 0
    if nwe > 1:
        rmsd_we = math.sqrt(msd_we/(nwe-1))
    rmsd_wexe = 0
    if nwexe > 1:
        rmsd_wexe = math.sqrt(msd_wexe/(nwexe-1))
    return rmsd_we, rmsd_wexe, nwe, nwexe 

def write_qm_freqs(filenm:str, mydict:dict, diat:dict):
    with open(filenm, "w") as outf:
        outf.write("\\begin{longtable}{p{20mm}cccccc}\n")
        outf.write("\\caption{First (%s) and second (%s) vibrational frequencies (cm$^{-1}$) from experiment~\\cite{Huber1979a} and quantum chemistry.}\\\\\n" % ( wetex, wexetex ))
        outf.write("\\hline\n")
        head = ("Compound & %s & %s & %s & %s & %s & %s \\\\\n" % ( wetex, wexetex, wetex, wexetex,wetex, wexetex ))
        outf.write("&\multicolumn{2}{c}{Experiment}&\multicolumn{2}{c}{CCSD(T)}&\multicolumn{2}{c}{MP2}\\\\\n")
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
            we = "we"
            wexe = "wexe"
            format = "%.1f"
            results = []
            count   = 0
            for m in [ exp, methods[0], methods[1] ]:
                if not m in mydict:
                    continue
                count += 1
                for w in [ "we", "wexe" ]:
                    if d in mydict[m] and w in mydict[m][d] and math.isfinite(mydict[m][d][w]):
                        results.append(format % mydict[m][d][w])
                    else:
                        results.append("")
            outf.write("%s" % to_latex(diat[d]["formula"], diat[d]["mult"] > 1))
            for r in range(len(results)):
                outf.write("& %s" % results[r])
            outf.write("\\\\\n")
        outf.write("\\end{longtable}\n")
        outf.write("\n")

def write_stats(args, mydict:dict, methods:list, pots:list):
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
        outf.write("\\begin{table}[htb]\n")
        outf.write("\\centering\n")
        outf.write("\\caption{Root mean square deviation for ground state vibrational frequencies %s and first anharmonic component %s (both in cm$^{-1}$) for different methods and analytical potentials fitted on %s, from the reference indicated below.%s}\n" % ( wetex, wexetex, cutoff, sorting) )
        outf.write("\\label{fstats%d}\n" % args.numax)
        if args.numax == -1:
            outf.write("\\begin{tabular}{lcc}\n")
            outf.write("\\hline\n")
            outf.write("\\multicolumn{1}{r}{Reference:} & \\multicolumn{2}{c}{%s}\\\\\n" % exp)
            outf.write("Method & %s & %s \\\\\n" % ( wetex, wexetex ))
        else:
            outf.write("\\begin{tabular}{lcccc}\n")
            outf.write("\\hline\n")
            outf.write("\\multicolumn{1}{r}{Reference:} & \\multicolumn{2}{c}{%s} & \\multicolumn{2}{c}{CCSD(T)}\\\\\n" % exp)
            outf.write("Method & %s & %s & %s & %s \\\\\n" % ( wetex, wexetex, wetex, wexetex ))
        outf.write("\\hline\n")
        
        ccsdt = "CCSD(T)"
        mp2   = "MP2"
        mrefs = [ exp ]
        
        rmsd_we, rmsd_wexe, nwe, nwexe = calc_stats(mydict[exp], mydict[ccsdt])
        outf.write("%s & %.2f & %.2f" % ( ccsdt, rmsd_we, rmsd_wexe ))
        if args.numax >= 0:
            outf.write(" & & ")
            mrefs.append(ccsdt)
        outf.write("\\\\\n")

        if mp2 in mydict:
            outf.write("%s" % mp2)
            for mref in mrefs:
                rmsd_we, rmsd_wexe, nwe, nwexe = calc_stats(mydict[mref], mydict[mp2])
                outf.write(" & %.2f & %.2f" % ( rmsd_we, rmsd_wexe ))
            outf.write("\\\\\n")

        spots = []
        for p in pots:
            if p in mydict:
                rmsd_we, rmsd_wexe, nwe, nwexe = calc_stats(mydict[ccsdt], mydict[p])
                spots.append( ( p, rmsd_we ) )
            else:
                print("Function %s missing from mydict" % p)
        for p in sorted(spots, key=lambda tup: tup[1]):
            m = p[0]
            if args.numax == -1:
                mrefs = [ exp ]
            outf.write("%s " % m.replace("_", "-"))
            for mref in mrefs:
                rmsd_we, rmsd_wexe, nwe, nwexe = calc_stats(mydict[mref], mydict[m])
                print("RMSD for %s from %s we %g (N=%d) wexe %g (N=%d)" % ( m, mref, rmsd_we, nwe, rmsd_wexe, nwexe ))
                if m == "Harmonic":
                    outf.write("& %.2f & -" % ( rmsd_we ))
                else:
                    outf.write("& %.2f & %.2f" % ( rmsd_we, rmsd_wexe ))
            if len(mrefs) == 1 and args.numax >= 0:
                outf.write("&&")
            outf.write("\\\\\n")
        outf.write("\\hline\n")
        outf.write("\\end{tabular}\n")
        outf.write("\\end{table}\n")

    return fstats

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute frequencies, make plots and tables.',
    )
    parser.add_argument('-v', "--verbose", action="store_true", default=False,help='Write info to the screen')
    parser.add_argument('-numax', default=0, type=int, help='Maximum energy (cm^-1) to take into account. 0 means all data. -1 means experimental data.')

    return parser.parse_args()

def exptab_header(outf):
    outf.write("\\begin{table}[ht]\n")
    outf.write("\\centering\n")
    outf.write("\\caption{First and second vibrational frequencies  either from experiment~\cite{Huber1979a} or computed using Psi4~\cite{psi4} based on experimental data.}\n")
    outf.write("\\label{expfreqs}\n")
    outf.write("\\begin{tabular}{lcccc}\n")
    outf.write("\\hline\n")
    outf.write("Molecule & \multicolumn{2}{c}{$\omega_e$} & \multicolumn{2}{c}{$\omega_w$x$_e$} \\\\\n")
    outf.write("& Exper & Psi4 & Exper & Psi4 \\\\\n")
    outf.write("\\hline\n")
    
if __name__ == "__main__":
    args = parse_args()

    ddd = get_diatomics()
    print("There are %d diatomics" % ( len(ddd) ))

    mdfile = "data/Diatomic_Moleculedata.csv"
    md = get_moldata(mdfile)
    mdfile2 = "data/Diatomic_Moleculedata_dscdm.csv"
    dscdm = get_moldata(mdfile2)
    md.update(dscdm)
    print("There are %d entries in %s and %s" % ( len(md), mdfile, mdfile2 ))
    dtex = "diat_table.tex"
    dtable(ddd, md, dtex)

    plog = "psi4.log"
    psi4.core.set_output_file(plog, False)
    pots    = []
    for p in [ Kratzer,Harmonic,Lippincott,Deng_Fan,Pseudo_Gaussian,Rydberg,Varshni,Morse,Valence_State,Rosen_Morse,Linnett,Poschl_Teller,Frost_Musulin,Levine,Wei_Hua,Tietz_I,Rafi,Noorizadeh,Tietz_II,Hulburt_Hirschfelder,Murrell_Sorbie,Sun,Lennard_Jones,Buckingham,Wang_Buckingham,Cahill,Xie2005a,Tang2003a ]:
        pots.append(str(p.__name__))

    xvg_we   = {}
    xvg_wexe = {}
    for m in methods+pots:
        xvg_we[m] = open("we-%s.xvg" % m, "w")
        xvg_wexe[m] = open("wexe-%s.xvg" % m, "w")
        xvg_we[m].write("@xaxis label \"Experimental Frequency $\omega_e$ (1/cm)\"\n")
        xvg_wexe[m].write("@xaxis label \"Experimental Frequency $\omega_e$x$_e$ (1/cm)\"\n")
        xvg_we[m].write("@yaxis label \"QM - Experiment\"\n")
        xvg_wexe[m].write("@yaxis label \"QM - Experiment\"\n")

    chi2 = {}
    for m in methods+pots:
        chi2[m] = {}
        for ww in [ "we", "wexe" ]:
            chi2[m][ww] = []

    mdwe   = "omega_e cm^{-1}"
    mdwexe = "omega_ex_e cm^{-1}"
    freqcsv  = "frequencies.csv"
    if args.numax == -1:
        explog  = open("exper_freq.log", "w")
        exptab  = open("exper_freq.tex", "w")
        exptab_header(exptab)
        expsum  = { "we": 0, "wexe": 0 }
        exp2sum = { "we": 0, "wexe": 0 }
        nexpsum = 0
    with open(freqcsv, "w") as outf:
        outf.write("Compound,we-Exper,wexe-Exper")
        for method in methods+pots:
            outf.write(",we-%s,wexe-%s,we-%s-Exp,wexe-%s-Exp" % ( method, method, method, method ))
        outf.write("\n")
        # Look up experimental data
        mydict = { exp: {} }
        for diat in ddd.keys():
            mydict[exp][diat] = {}
            for m in md.keys():
                if (ddd[diat]["formula"] == m[0] and
                    md[m]["Te cm^{-1}"] == 0.0):
                    mydict[exp][diat]["we"]   = md[m][mdwe]
                    mydict[exp][diat]["wexe"] = md[m][mdwexe]
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
                        if args.numax == -1:
                            explog.write("********** %s **********\n" % diat)
                            explog.write("Psi4: %s\n" % str(mydict[method][diat]))
                            compound = ddd[diat]['formula']
                            for m in md.keys():
                                if (ddd[diat]["formula"] == m[0] and
                                    md[m]["Te cm^{-1}"] == 0.0):
                                    explog.write("Experiment: %s %s = %g\n" % ( diat, mdwe, md[m][mdwe] )) 
                                    explog.write("Experiment: %s %s = %g\n" % ( diat, mdwexe, md[m][mdwexe] )) 
                                    exptab.write("%s & %.1f & %.1f & %.1f & %.1f\\\\\n" % 
                                                 ( ddd[diat]["formula"], md[m][mdwe], mydict[method][diat]["we"],
                                                   md[m][mdwexe], mydict[method][diat]["wexe"] ))
                                    diff = ( mydict[method][diat]["we"] - md[m][mdwe])
                                    expsum["we"]  += diff
                                    exp2sum["we"] += diff**2
                                    diff = ( mydict[method][diat]["wexe"] - md[m][mdwexe])
                                    expsum["wexe"]  += diff
                                    exp2sum["wexe"] += diff**2
                                    nexpsum += 1
                                    
                            explog.write("\n\n")
                            
                        if method == "CCSD(T)" or args.numax == -1:
                            for pot in pots:
                                # Compute them
                                ddd2 = compound
                                energies = compute_them(pot, rvals, jparms[method][ddd2][pot]['params'])
                                if args.verbose:
                                    print("Computed %d energies for %s using %s. First %g" % ( len(energies), diat, method, energies[0] ))
                                try:
                                    if not pot in mydict:
                                        mydict[pot] = {}
                                    mydict[pot][diat] = compute_freq(rvals, energies, atoms, charge, mult)
                                except:
                                    print("Something weird with %s for %s with %d data points" % ( pot, compound, len(rvals) ))
                    except:
                        print("Something weird with %s" % filenm)
                else:
                    print("Not enough data points (%d) in %s" % ( len(rvals), filenm ))
            if args.numax == -1:
                break

        # Collected all data
        # Check energies at minimum
        for m in mydict:
            for d in mydict[m]:
                ere = "E(re)"
                if ere in mydict[m][d] and mydict[m][d][ere] < -0.01:
                    print("Energy at minimum %g/cm for %s and %s " % (mydict[m][d][ere]*AU2INVCM, m, d))
        mwe = "we"
        mwexe = "wexe"
        for diat in ddd.keys():
            outf.write("%s" % diat)
            if mwe in mydict[exp][diat]:
                we = mydict[exp][diat][mwe]
                outf.write(",%g," % ( we ))
                wexe = None
                if mwexe in mydict[exp][diat]:
                    wexe = mydict[exp][diat]["wexe"]
                    if math.isfinite(wexe):
                        outf.write("%g" %  wexe)

                for m in methods+pots:
                    if m in mydict:
                        if not diat in mydict[m]:
                            if args.verbose:
                                print("No data for %s %s" % ( method, diat ))
                            continue
                        if mwe in mydict[m][diat]:
                            xvg_we[m].write("%10g  %10g\n" % ( we, mydict[m][diat][mwe] ))
                        if mwexe in mydict[m][diat] and math.isfinite(wexe):
                            xvg_wexe[m].write("%10g  %10g\n" % ( wexe, mydict[m][diat][mwexe] ))
                        
                        outf.write(",%g,%g" % ( mydict[m][diat]["we"], mydict[m][diat]["wexe"] ))
                        if we:
                            diff = mydict[m][diat]["we"]-we
                            chi2[m]["we"].append(diff**2)
                            outf.write(",%g," % ( diff ))
                            if math.isfinite(wexe):
                                diff = mydict[m][diat]["wexe"]-wexe
                                chi2[m]["wexe"].append(diff)
                                outf.write("%g" % ( diff ))
                        else:
                            outf.write(",,")
                    else:
                        outf.write(",,,,")
            outf.write("\n")
        write_qm_freqs("qm_freqs.tex", mydict, ddd)
        fstats = write_stats(args, mydict, methods, pots)
    if args.numax == -1:
        explog.close()
        if nexpsum > 0:
            rmsd = {}
            for ww in [ "we", "wexe" ]:
                rmsd[ww] = (exp2sum[ww]/nexpsum-(expsum[ww]/nexpsum)**2)**(0.5)
            exptab.write("\\hline\n")
            exptab.write("RMSD &  & %.1f &  & %.1f\\\\\n" % 
                         ( rmsd["we"], rmsd["wexe"] ))
            exptab.write("MSE &  & %.1f &  & %.1f\\\\\n" % 
                         ( expsum["we"]/nexpsum, expsum["wexe"]/nexpsum ))
            exptab.write("\\hline\n")
            exptab.write("\\end{tabular}\n\\end{table}\n")
        exptab.close()
    for m in methods+pots:
        xvg_we[m].close()
        xvg_wexe[m].close()

    
    print("Please check %s, %s, %s and %s" % ( freqcsv, dtex, fstats, plog ))

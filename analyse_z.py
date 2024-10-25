#!/usr/bin/env python3

import json, glob,  math, os, sys
import pandas as pd
import importlib  
foobar = importlib.import_module("curve_fit")
from diatomics import get_diatomics

debug   = False
verbose = False

covs = [ "Non-covalent", "Covalent" ]

fdata = { 
         "Buckingham": { "ref": "Buckingham1938a", "np": 3, "nmolsim": 3 },
         "Cahill": { "ref": "Cahill2004a", "np": 6, "nmolsim": 6 },
         "Deng_Fan": { "ref": "Deng1957a_potential", "np": 3, "nmolsim": 3 },
         "Frost_Musulin": { "ref": "Frost1954a", "np": 4, "nmolsim": 3 },
         "Harmonic": { "np": 3, "nmolsim": 2 },
         "Hulburt_Hirschfelder": { "ref": "Hulburt1941a", "np": 5, "nmolsim": 5 },
         "Kratzer": { "ref": "Kratzer1920a", "np": 2, "nmolsim": 2 },
         "Lennard_Jones": { "ref": "Lennard-Jones1924b", "np": 2, "nmolsim": 2 },
         "Levine": { "ref": "Levine1966a", "np": 4, "nmolsim": 4 },
         "Linnett": { "ref": "Linnett1940a", "np": 4, "nmolsim": 3 },
         "Lippincott": { "ref": "Lippincott1953a", "np": 3, "nmolsim": 3 },
         "Morse": { "ref": "Morse1929a", "np": 3, "nmolsim": 3 },
         "Murrell_Sorbie": { "ref": "Murrell1974a", "np": 5, "nmolsim": 5 },
         "Noorizadeh": { "ref": "Noorizadeh2004a_empirical", "np": 5, "nmolsim": 4 },
         "Poschl_Teller": { "ref": "Poeschl1933a", "np": 4, "nmolsim": 3, "label": "P{\\\"o}schl-Teller" },
         "Pseudo_Gaussian": { "ref": "Sage1984a", "np": 3, "nmolsim": 3 },
         "Rafi": { "ref": "Rafi1995a", "np": 5, "nmolsim": 4 },
         "Rosen_Morse": { "ref": "Rosen1932a", "np": 4, "nmolsim": 3 },
         "Rydberg": { "ref": "Rydberg1932a", "np": 3, "nmolsim": 3 },
         "Sun": { "ref": "Sun1997a_potential", "np": 8, "nmolsim": 8 },
         "Tang2003a": { "ref": "Tang2003a", "np": 6, "nmolsim": 6 },
         "Tietz_I": { "ref": "Tietz1971a", "np": 5, "nmolsim": 4, "label": "Tietz I" },
         "Tietz_II": { "ref": "Tietz1971a", "np": 5, "nmolsim": 4, "label": "Tietz II" },
         "Valence_State": { "ref": "Gardner1999a", "np": 4, "nmolsim": 4 },
         "Varshni": { "ref": "Varshni1988a", "np": 3, "nmolsim": 3 },
         "Wang_Buckingham": { "ref": "Wang2012a", "np": 3, "nmolsim": 3 },
         "Wei_Hua": { "ref": "Hua1990a", "np": 4, "nmolsim": 4, "label": "Wei Hua" },
         "Xie2005a": { "ref": "Xie2005a", "np": 4, "nmolsim": 3 }
         }

def write_header(outf, table:str, caption:str, label:str):
    outf.write("\\begin{%s}[ht]\n" % table)
    outf.write("\\centering\n")
    outf.write("\\caption{%s}\n" % caption)
    outf.write("\\label{%s}\n" % label)
    
def write_footer(outf, table:str):
    outf.write("\\hline\n")
    outf.write("\\end{tabular}\n")
    outf.write("\\end{%s}\n" % table)

def calc_aver(mysum:dict)->dict:
    av = { "Z": 0, "RMSD": 0 }
    for res in av.keys():
        N = len(mysum[res])
        av[res] = 0.0
        for k in range(N):
            val = mysum[res][k]
            if "Z" == res:
                av[res] += val
            else:
                av[res] += val**2
        if N > 0:
            av[res] /= N
        if "RMSD" == res:
            av[res] = math.sqrt(av[res])
    return av

def print_exper_table(filenm:str, mysum:dict, df, functions:dict):
    with open(filenm, "w") as outf:
        caption = "Statistics per function for fits to experimental data for the 14 compounds used by Royappa {\\em et al.}~\\cite{Royappa2006a}.  M is the number of parameters, Z is the average Z-score (cm$^-2$/{\\AA}), $\\Delta$Z indicates the difference between the Z calculated here and that by Royappa, and RMSD (J/mol) is the root mean signed error from experimental data without any energy cut-off. Table is sorted after Z-score."
        write_header(outf, "table", caption, "experiment")

        # Number of covalency columns
        ncov = 1
        cov = covs[1]
        outf.write("\\begin{tabular}{lccccc}\n")
        outf.write("\\hline\n")
        ncol = 2
        outf.write("\\\\\n")
        outf.write("Function & M & Z & $\\Delta$Z & RMSD\\\\\n")
        outf.write("\\hline\n")
        fkeys = {}
        for func in functions:
            fkeys[func] = 0
            for z in mysum[func][cov]["Z"]:
                fkeys[func] += z
            fkeys[func] /= len(mysum[func][cov]["Z"])
        fkeys = list(sorted(fkeys.items(), key=lambda kv:kv[1]))
        if debug:
            print(fkeys)
        for fff in fkeys:
            func = fff[0]
            if not func in df.loc['Avg.'].index:
                continue
            myfuncname = func[:].replace("_", "-")
            if func in fdata:
                lab = "label"
                if lab in fdata[func]:
                    myfuncname = fdata[func][lab]
                if "ref" in fdata[func]:
                    myfuncname += ("\\cite{%s} " % ( fdata[func]["ref"] ))
                myfuncname += (" & %d " % ( fdata[func]["np"]))
            else:
                myfuncname += " & N/A "
            outf.write("%s " % myfuncname)
            factor = 1
            if func in mysum:
                av = calc_aver(mysum[func][cov])
                if func in df.loc['Avg.'].index:
                    dz = ("%.0f" % ( av["Z"] - df.loc['Avg.'][func] ))
                    outf.write("& %.0f & %s & %.0f" %
                               ( av["Z"], dz, av["RMSD"] ))
            outf.write("\\\\\n")
        write_footer(outf, "table")

def mystr(f:float)->str:
    if f < 10:
        return ("%.1f" % f)
    else:
        return ("%.0f" % f)

def print_theory_table(filenm:str, jsons:list, mysum:dict, df, numax:int, functions:dict):
    with open(filenm, "w") as outf:
        ecut = ""
        if numax > 0:
            ecut = ("An energy cut-off of %d cm$^{-1}$ was applied." % numax)
        else:
            ecut = ("No energy cut-off was applied.")
        caption = ("Statistics per function for quantum chemistry results. M$_f$ is the number of parameters used for fitting, M$_{sim}$ the number of parameters if the minimum is not fixed at zero and when redundancies are removed (see text). N is the number of compounds, Z is the average Z-score (cm$^-2$/{\\AA}), and RMSD (J/mol) is the root mean signed error from quantum chemical results.  %s Table is sorted after Z-score for covalent compounds computed at the CCSD(T) level of theory." % ecut)
        table = "table*"
        write_header(outf, table, caption, ("theory%d" % numax))

        # Number of covalency columns
        ncov = 2
        outf.write("\\begin{tabular}{lcc")
        for js in jsons:
            for c in range(ncov):
                ncol = "ccc"
                outf.write(ncol)
        outf.write("}\n")
        outf.write("\\hline\n")
        ncol = 2*ncov
        printHeader = False
        for js in jsons:
            if not printHeader:
                outf.write(" & & ")
            jjjs = js.split("-")[0].upper()
            outf.write("& \\multicolumn{%d}{c}{%s}" % ( ncol, jjjs ))
            printHeader = True

        if printHeader:
            outf.write("\\\\\n")
        myfunc = list(functions.keys())[0]
        printcov = False
        for js in jsons:
            if not printcov:
                outf.write(" & & ")
            printcov = True
            for c in range(ncov):
                covnum = len(mysum[js][myfunc][covs[c]]["Z"])
                outf.write(" & \\multicolumn{2}{c}{%s (%d)}" % ( covs[c], covnum ) )
        if printcov:
            outf.write("\\\\\n")
        outf.write("Function & M$_f$ & M$_{sim}$ ")
        for js in jsons:
            for c in range(ncov):
                outf.write("& Z & RMSD")
        outf.write("\\\\\n")
        outf.write("\\hline\n")
        fkeys = {}
        jsccsd = jsons[0]
        if verbose:
            print("jsccsd has %d keys" % ( len(mysum[jsccsd].keys())))
            if debug:
                for k in mysum[jsccsd].keys():
                    print(k)
        for func in functions:
            fkeys[func] = 0
            for z in mysum[jsccsd][func]["Covalent"]["Z"]:
                fkeys[func] += z
            N = len(mysum[jsccsd][func]["Covalent"]["Z"])
            if N == 0:
                print("No data for %s %s" % ( jsccsd, func ))
            else:
                fkeys[func] /= N
        fkeys = list(sorted(fkeys.items(), key=lambda kv:kv[1]))
        if debug:
            print(fkeys)
        for fff in fkeys:
            func = fff[0]
            myfuncname = func[:].replace("_", "-")
            if func in fdata:
                lab = "label"
                if lab in fdata[func]:
                    myfuncname = fdata[func][lab]
                if "ref" in fdata[func]:
                    myfuncname += ("\\cite{%s}" % ( fdata[func]["ref"] ))
                myfuncname += ( " & %d & %d " % ( fdata[func]["np"], fdata[func]["nmolsim"]))
            else:
                myfuncname += " & N/A "
            outf.write("%s " % myfuncname)
            factor = 1000*foobar.WaveNumber2kJ
            for js in jsons:
                if func in mysum[js]:
                    for cov in covs:
                        av = calc_aver(mysum[js][func][cov])
                        if js.find("exp") >= 0:
                            if covs[0] == cov:
                                if func in df.loc['Avg.'].index:
                                    dz = ("%.0f" % ( av["Z"] - df.loc['Avg.'][func] ))
                                    outf.write("& %s & %s & %s" %
                                               ( mystr(av["Z"]), dz, 
                                                 mystr(av["RMSD"] )))
                        else:
                            outf.write("& %s & %s" %
                                       ( mystr(av["Z"]), mystr(factor*av["RMSD"] )))
                else:
                    outf.write(" & & & ")
            outf.write("\\\\\n")
                
        outf.write("\\hline\n")
        outf.write("\\end{tabular}\n")
        outf.write("\\end{%s}\n" % table)

def calc_mysum(jsonfile:str, diatomics:dict, covalent, useFormula:bool)->dict:
    mysum     = {}
    functions = {}
    results = foobar.read_potential_parms("json/"+jsonfile)
    for kk in results.keys():
        found = False
        for d in diatomics:
            if diatomics[d]["formula"] == kk:
                found = True
        if not found:
            if verbose:
                print("Did not find %s in diatomics database" % ( kk ) )
            continue
        elif debug:
            print("Found %s with %d keys" % ( kk, len(results[kk].keys()) ))
        if kk in covalent:
            cc = covs[1] 
        else:
            cc = covs[0]
        for func in results[kk].keys():
            if func in [ "filename", "E[re]", "re" ]:
                continue
            functions[func] = 1
            for res in [ "Z", "RMSD" ]:
                if res in results[kk][func]:
                    xx = results[kk][func][res]
                    if not func in mysum:
                        mysum[func] = { covs[0]: {}, covs[1]: {} }
                    for c in covs:
                        if not res in mysum[func][c]:
                            mysum[func][c][res] = []
                    mysum[func][cc][res].append(xx)
        
    return results, mysum, functions

if __name__ == "__main__":
    print("Will analyse json file produced by the curve_fit.py script.")
    csv_file = 'data/exp/Royappa2006a-JMS_787_209-table-I.csv'
    df = pd.read_csv(csv_file)
    df.set_index('Species', inplace=True)

    covalent = []
    diatomics = get_diatomics()
    for d in diatomics:
        if diatomics[d]["covalent"]:
            covalent.append(diatomics[d]["formula"])
    print("There are %d covalent compounds out of %d" % ( len(covalent), len(diatomics)))
    jsroy, msroy, functions = calc_mysum("exp.json", diatomics, covalent, True)
    tables = "experiment.tex"
    print_exper_table(tables, msroy, df, functions)

    for numax in [ 0, 1000, 5000 ]:
        mysum   = {}
        results = []

        os.chdir("json")
        jsfiles = glob.glob("*-numax=%d.json" % numax)
        os.chdir("..")
        for jjs in range(len(jsfiles)):
            js = jsfiles[jjs]
            rrr, mysum[js], ff = calc_mysum(js, diatomics, covalent, True)
            if verbose:
                print("Read %s with %d functions" % (js, len(mysum[js].keys())))
            results.append(js[:])
            functions = functions | ff

        ttab = ("theory%d.tex" % numax)
        print_theory_table(ttab, results, mysum, df, numax, functions)
        tables += " " + ttab

    print("Please check output in %s" % (tables))

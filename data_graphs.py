#!/usr/bin/env python3

import argparse, copy, glob, os, sys
import json, pandas
import numpy as np
import matplotlib.pyplot as plt
from diatomics import get_diatomics
from curve_fit import read_potential_parms
from frequencies import to_latex

def get_refs(filenm:str)->dict:
    refs = {}
    df = pandas.read_csv(filenm, comment='#', sep=",")
    cpd = "Formula"
    rrr = "Reference"
    nrows = len(df[cpd])
    for row in range(nrows):
        compound = df[cpd][row][:]
        refs[compound] = df[rrr][row][:]
    return refs

if __name__ == "__main__":
    diatomics = get_diatomics()
    methods = { "exp": { "fn": "exp" },
                "MP2": { "fn": "MP2-numax=1000" },
                "CCSD(T)": { "fn": "CCSD(T)-numax=1000" }
               }
    for m in methods:
        jfn = "json/" + methods[m]["fn"] + ".json"
        methods[m]["parms"] = read_potential_parms(jfn)
        print("There are %d compounds in %s" % ( len(methods[m]["parms"]), jfn))

    refs = get_refs("data/reference.csv")
    dgdir = "output/data_graphs"
    os.makedirs(dgdir, exist_ok=True)
    os.chdir(dgdir)
    tex = open("graphs.tex", "w")
    data = "../../data/"
    ncomp = 1
    for comp in methods["MP2"]["parms"]:
        found  = False
        ddd    = None
        for d in diatomics:
            if comp == diatomics[d]["formula"]:
                ddd   = d
                found = True
        if not found:
            continue
        fns   = ""
        label = ""
        nmeth = 0
        for m in methods:
            mycomp = comp
            if mycomp in methods[m]["parms"]:
                filenm = data + m + "/" + methods[m]["parms"][mycomp]["filename"]
                if os.path.exists(filenm):
                    fns   += (" '%s'" % filenm)
                    label += (" '%s'" % m )
                    nmeth += 1
                else:
                    print("No such file %s" % filenm)
        addflag = ""
        if nmeth == 3:
            addflag = " -colors black red blue -ls None solid dashed -mk o None None "
        else:
            addflag = " -colors red blue -ls solid dashed "
        pdf = ("%s.pdf" % comp)
        os.system("viewxvg -f %s -label %s -pdf %s -noshow -legend_x 0.3 %s" % ( fns, label, pdf, addflag ))
        tex.write("\\begin{figure*}[ht]\n")
        tex.write("\\centering\n")
        tex.write("\\includegraphics[width=12cm]{graphs/%s}\n" % pdf)
        extra  = ""
        if comp in refs:
            extra = (" Experimental data from ref.\\citenum{%s}." % ( refs[comp]))
        radical = diatomics[ddd]["mult"] > 1
        ccc     = ddd.replace("-", " ")
        tex.write("\\caption{Compound %s (%s). %s}\n" % ( ccc, to_latex(comp, radical), extra ))
        tex.write("\\end{figure*}\n")
        if ncomp % 15 == 0:
            tex.write("\\cleardoublepage\n")
        ncomp += 1
        
    tex.close()

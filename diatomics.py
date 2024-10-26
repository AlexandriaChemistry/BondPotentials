#!/usr/bin/env python3

import copy, pandas, sys

def get_diatomics(ddfile:str = "data/diatomics.csv", exclude:list=["boron-nitride"]) -> dict:
    diatomics = {}
    df = pandas.read_csv(ddfile, comment='#', sep=",")
    cpd = "Compound"
    nrows = len(df[cpd])
    for row in range(nrows):
        compound = df[cpd][row][:]
        if compound in exclude:
            continue
        diatomics[compound] = {}
        for d in df:
            if cpd == d:
                continue
            diatomics[compound][d] = df[d][row]
        if "chargei" in df and "chargej" in df:
            diatomics[compound]["charge"] = df["chargei"][row] + df["chargej"][row]
        else:
            sys.exit("Cannot find chargei or chargej in %s" % ddfile)
    return diatomics

def get_moldata(mdfile:str) -> dict:
    moldata = {}
    df = pandas.read_csv(mdfile,  comment='#', sep=",")
    cpd = "Molecule name"
    est = "Electronic state"
    nrows = len(df[cpd])
    for row in range(nrows):
        # Only store the ground level
        if df["Te cm^{-1}"][row] == 0.0:
            formula = (df[cpd][row], df[est][row])
            moldata[formula] = {}
            for d in df:
                if cpd == d or est == d:
                    continue
                moldata[formula][d] = df[d][row]
    return moldata
        
if __name__ == "__main__":
    verbose = False
    #verbose = True
    ddfile = "data/diatomics.csv"
    ddd = get_diatomics(ddfile)
    print("There are %d entries in %s" % ( len(ddd), ddfile ))
    if verbose:
        for d in ddd:
            print(ddd[d])
    
    mdfile = "data/Diatomic_Moleculedata.csv"
    md = get_moldata(mdfile)
    print("There are %d entries in %s" % ( len(md), mdfile ))
    if verbose:
        for d in md:
            print(d,md[d])
            
    mdfile = "data/Diatomic_Moleculedata_updates.csv"
    dscdm = get_moldata(mdfile)
    print("There are %d entries in %s" % ( len(dscdm), mdfile ))
    md.update(dscdm)
    
    if verbose:
        for d in md:
            print(d,md[d])

            
    if verbose:
        print("Found the following diatomics:")
    hits = {}
    for m in md.keys():
        (formula, est) = m
        for d in ddd:
            if ddd[d]["formula"] == formula and md[m]["Te cm^{-1}"] == 0.0:
                hits[d] = 1
                if verbose:
                    print(formula, md[m]['Re \\AA'], md[m]['D0 eV'] * 219474.6313702 / 27.21138602)
    print("There are %d hits for formula/electronic state in our dataset" % len(hits.keys()))
    for d in ddd:
        if not d in hits:
            print("No support for %s in %s" % ( d, mdfile ))

#!/usr/bin/env python3

import json, sys
from curve_fit import read_potential_parms

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Please give me the input file name, e.g. json/MP2-numax=50.json")
    pfile = "json/potentials.json"
    with open(pfile, "r") as inf:
        pson = json.load(inf)
    print("Successfully read %s with %d potentials" % ( pfile, len(pson.keys())))
    for func in pson.keys():
        if not "n" in pson[func]:
            print("Strange func %s" % func)
        else:
            pson[func]["min"] = [0.0] * pson[func]["n"]
            pson[func]["max"] = [0.0] * pson[func]["n"]

    for infile in sys.argv[1:]:
        parms = read_potential_parms(infile)
        for cmp in parms:
            for func in parms[cmp]:
                for m in pson.keys():
                    if m == func:
                        for p in range(pson[m]["n"]):
                            pson[m]["min"][p] = min(float(pson[m]["min"][p]), float(parms[cmp][func]["params"][p]))
                            pson[m]["max"][p] = max(float(pson[m]["max"][p]), float(parms[cmp][func]["params"][p]))

    p2file = "potentials.json"
    with open(p2file, "w") as outf:
        json.dump(pson, outf, sort_keys=True, indent=2)
    print("Please check %s" % p2file)

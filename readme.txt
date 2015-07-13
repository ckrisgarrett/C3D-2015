
Please note, this code has only been tested for the problems in
"Optimization and Large Scale Computation of an Entropy-Based Moment Closure"
by Garrett, Hauck, and Hill (2015)
We make no guarantee about its fidelity in general.
This code is supplied for the purposes of reproducibility.



////  Note  ////
It is believed that for the M_N solver, there is an addition that should be 
made for robustness.  In the code, you start with u^n, then find alpha^n and 
finally update for u^{n+1}.  This note has to do with the fact that alpha^n is
only an approximation within some prescribed tolerance, i.e. 
u^n \approx <m exp(m^T alpha^n)>.  One should calculate a new u^n so that
u^n = <m exp(m^T alpha^n)> exactly.  But this may not preserve particle number.
So then u^n and alpha^n should be scaled appropriately to represent the original
particle number, i.e. u_0^n should not change.



////  Files  ////
input.deck          Contains parameters for the program
license.txt         Open source license allowing use of this program
Makefile.local      Makefile for a workstation
Makefile.titan      Makefile for Titan
opt_statistics.py   Python script to output certain statistics
                    Must set OUTPUT_OPT_STATS to true in input.deck to get data
plot.py             Plots density
statistics.py       Python script to get regularization statistics
timings.py          Python script to get timing data for nodes
                    Must set OUTPUT_TIMINGS to true in input.deck to get data
timings_1node.py    Python script to get timing data for one node
                    Must set OUTPUT_TIMINGS to true in input.deck to get data
titan.run           A script to run code on Titan
titan.setup         A script to setup the environment on Titan
src                 Folder containing all the source files
output              An empty directory initially
                    Necessary for output data files

/*
    File:   opt.h
    Author: Kris Garrett
    Date:   February 13, 2013
*/

#ifndef __OPT_H
#define __OPT_H

#include <stack>


// Input to opt
struct OPTIONS
{
    int maxIter;
    double tolAbs;
    double tolRel;
    double tolGamma;
    double condHMax;
    bool useGaunt;
    bool useGauntSparse;
};

// Output from opt
struct OUTS
{
    double r;
    int iter;
    int iterGamma;
    double gamma;
    double normG;
};


// Regularization stuff.
static const int NUM_REGULARIZATIONS = 5;
static const double REGULARIZATIONS[NUM_REGULARIZATIONS] = {0.0, 1e-3, 1e-2, 1e-1, 5e-1};


// Global function declarations.
void opt(int nm, int nm2, int nq, double *u, double *alpha, OPTIONS options, double *w, double *p, 
         double *p2, OUTS *outs);
void fobj_init(int nm2);
double fobjMn(int nm, int nm2, int nq, double *alpha, double *u, double *w, double *p, double *p2, 
              double *g, double *h, bool useGaunt, bool useGauntSparse);
double fobjMnUseH(int nm, double *alpha, double *u, double *p, double *g, double *h);
void solveHMn(int nm, int nm2, int nq, double *alpha, double *w, double *p, double *p2, 
              double *h, bool useGaunt, bool useGauntSparse);
void linesearch_init(int nm);
double linesearch(int n1, int n2, int nq, double *alpha, double *d, double f, double *g, 
                  double *u, double *w, double *p);


// Global variable for Gaunt coefficients.
extern double *g_beta;
extern double *g_betaVal;
extern int *g_betaColInd;
extern int *g_betaRowPtr;


// Optimization class for batch Mn.
class Optimization
{
public:
    Optimization(int i, int j, int k, int n, double *u, double *alpha);
    void preProcess(int n);
    void postProcess(int n);
    
    static void opt(int nm, int nm2, int nq);
    
    static std::stack<Optimization*> *c_stack1;
    static std::stack<Optimization*> *c_stack2;
    static std::stack<Optimization*> *c_stack3;
    
    double *c_u;
    double *c_alpha;
    double *c_h;
    double *c_alpha0;
    double *c_uOriginal;
    double c_t;
    double c_r;
    double c_err;
    int c_iter;
    int c_iterGamma;
    int c_totalIter;
    int c_i, c_j, c_k;
    int c_regIndex;
    double c_u0;
    bool c_preProcessed;
};

#endif

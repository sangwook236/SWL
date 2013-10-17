/*
 * Branches 0 and -1 of the Lambert W function.
 *
 * This file should be compiled with the preprocessor symbol FUNCTION defined to
 * one of loglambertwn1, lambertwn1, loglambertw0, and lambertw0.
 *
 * $Id: lambert.c,v 1.3 1998/09/02 18:59:10 shan Exp $
 */

// [ref]
//	http://www.comp.nus.edu.sg/~bioinfo/bayesprot/Entropic/Brand/
//	https://meteo.unican.es/trac/MLToolbox/browser/MLToolbox/MeteoLab/BayesNets/BNT/Entropic/Brand?rev=1
//	http://www.merl.com/people/brand/

#include <stdio.h>
#include <ctype.h>
#include <math.h>
/* #include <strings.h>*/
#include <stdlib.h>
#include <float.h>

#ifdef MATLAB_MEX_FILE
/* #define V4_COMPAT */
#include <matrix.h>  /* Matlab matrices */
#else
#include <errno.h>
//#include <sys/errno.h>
#endif

#ifndef M_E
#define M_E (2.7182818284590452354) /* e */
#endif
#define M_E_INV (-0.36787944117144232159552377016) /* -e^-1 */
#define W01 (0.56714329040978387299996866221)
#define EPSILON 1e-20

#define inline __inline

static double outofrange(const char *func, double input)
{
#ifdef MATLAB_MEX_FILE
    char buf[200];
    sprintf(buf, "Argument to %s out of range: %.12g", func, input);
    mexWarnMsgTxt(buf);
    return mxGetNaN();
#else
    errno = EDOM;
    return HUGE;
#endif
}

/*
 * Halley's method -- iteratively solve x * e^x = y, i.e.,
 * log(x) + x = log(y).
 */

inline static double halley(double x, double y)
{
    register double oldp = DBL_MAX;
    while (1)
    {
        register const double ex = exp(x);
        register const double p = x * ex - y;
        if (0 == p || fabs(p) >= fabs(oldp)) break;
        x -= (oldp=p) / (ex*(x+1.0) - (x+2.0)*p/(x+x+2.0));
    }
    return x;
}

extern double loglambertwn1(double y);
extern double lambertwn1(double y);
extern double loglambertw0(double y);
extern double lambertw0(double y);

double lambertwn1(double y)
{
    if (y < 0.0)
    {
        if (y > M_E_INV)
        {
            register double x;
            /* Make a first guess */
            if (y < -0.02)
            {
                const double x2 = 2.0 * (1.0 + M_E*y);
                x = sqrt(x2);
                x = - (5.0/3.0) - (2.0/3.0)*M_E*y
                    - x * (1 + x2 * ((11.0/72.0) + (43.0/540.0)*x));
            }
            else
            {
                const double z = log(-y);
                /*
                 * We check here for z close to or less than -log(realmax).
                 * Since halley() fails by overflow, we use loglambertwn1()
                 * instead.
                 */
                if (z < -700) return -loglambertwn1(-z);
                x = z + log(-z);
            }
            /* Halley's method */
            return halley(x, y);
        }
        else if (y == M_E_INV)
        {
            /* Tip of branch */
            return -1.0;
        }
        else
        {
            /* Argument out of range */
            return outofrange("lambertwn1", y);
        }
    }
    else if (y == 0.0)
    {
        /* -Inf */
      /* return -1.0/0.0; -- won't compile on microsoft */
      return -1e100;
    }
    else
    {
        /* Argument out of range */
        return outofrange("lambertwn1", y);
    }
}

double loglambertwn1(double y)
{
    if (y > 1.0)
    {
        if (y > 6e+17)
        {
            /* For very large y, exp dominates */
            return y;
        }
        else if (y < 20.0)
        {
            /* For moderately small y, just use lambertwn1 */
            return -lambertwn1(-exp(-y));
        }
        else
        {
            /* Newton-Rhapson */
            register double oldx = y + log(y);
            register double x = oldx * (1 - log(oldx) - y) / (1 - oldx);
            /*
             * For y >= 20, the initial guess for x above is guaranteed to be
             * too small.  By considering the signs of the first and second
             * derivatives, we know we've run out of numeric precision when
             * x >= oldx.
             */
            do
            {
                oldx = x;
                x = x * (1 - log(x) - y) / (1 - x);
            }
            while (oldx - x > EPSILON);
            return x;
        }
    }
    else if (y == 1.0)
    {
        /* Tip of branch */
        return 1.0;
    }
    else
    {
        /* Argument out of range */
        return outofrange("loglambertwn1", y);
    }
}

double lambertw0(double y)
{
    if (y == 0.0)
    {
        return 0.0;
    }
    else if (y > M_E_INV)
    {
        register double x;
        /* Make a first guess */
        if (y < 0.0)
        {
            x = 2.0 * (1.0 + M_E*y);
            x = -1.0 - x/3.0 + sqrt(x) * (1 + x * (11.0/72.0));
        }
        else if (y < 1.0)
        {
            x = y;
        }
        else if (y == 1.0)
        {
            return W01;
        }
        else
        {
            const double logy = log(y);
            /*
             * For very large y, halley() can fail by overflow, so use the
             * Newton-Rhapson part in loglambertw0() instead.  Note that the
             * +Inf case is handled here.
             */
            if (logy > 40) return loglambertw0(-logy);
            x = fabs(log(logy));
        }
        /* Halley's method */
        return halley(x, y);
    }
    else if (y == M_E_INV)
    {
        /* Tip of branch */
        return -1.0;
    }
    else
    {
        /* Argument out of range */
        return outofrange("lambertw0", y);
    }
}

double loglambertw0(double y)
{
    if (y >= 37.0)
    {
        /* For very large y (i.e., W_0(small positive), exp dominates */
        return exp(-y);
    }
    else if (y > -3.0)
    {
        if (y == 0.0)
        {
            return W01;
        }
        else
        {
            /* For moderately large y, just use lambertw0 */
            return lambertw0(exp(-y));
        }
    }
    else
    {
        /* Newton-Rhapson */
        register double x = -y - log(-y), oldx = 0.0;
        /*
         * For y < -1, the initial guess for x above is guaranteed to be too
         * small.  By considering the signs of the first and second derivatives,
         * we know we've run out of numeric precision when x <= oldx.
         */
        while (x - oldx > EPSILON)
        {
            oldx = x;
            x = x * (1 - log(x) - y) / (x + 1);
        }
        return x;
    }
}

#ifdef MATLAB_MEX_FILE

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *arg;
    const double *a;
    double *r;
    int i;

    if (nrhs != 1) mexErrMsgTxt("takes 1 input argument.");
    if (nlhs > 1) mexErrMsgTxt("outputs one result.");

    arg = prhs[0];
    if (!mxIsDouble(arg) || mxIsSparse(arg) || mxIsComplex(arg))
        mexErrMsgTxt("1st arg must be a real non-sparse matrix.");
    a = mxGetPr(arg);

    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(arg),
        mxGetDimensions(arg), mxDOUBLE_CLASS, mxREAL);
    r = mxGetPr(plhs[0]);

    for (i = mxGetNumberOfElements(arg); i > 0; i--)
        *r++ = FUNCTION(*a++);
}

#else

//--S [] 2013/10/16: Sang-Wook Lee
/*
int main()
{
    int i;
    for (i = -100; i <= 100; i++)
    {
        const double y = (double)i/100;
        printf("%g %g\n", y, FUNCTION(y));
    }

    return 0;
}
*/
//--E [] 2013/10/16: Sang-Wook Lee

#endif

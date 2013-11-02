#include "AutoRegression.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

// [ref] http://paulbourke.net/miscellaneous/ar/.

#define MAXENTROPY      0
#define LEASTSQUARES    1
#define MAXENTROPY2     2  // NOTICE [caution] >> maybe, this is not correctly working.

#define GENERATE_INTERMEDIATES

bool ARMaxEntropy(double *,int,int,double **,double *,double *,double *,double *);
bool ARLeastSquare(double *,int,int,double *);
bool SolveLE(double **,double *,unsigned int);

//--S [] 2013/11/02: Sang-Wook Lee
//void ARMaxEntropy (double *inputseries, int length, int degree,
void ARMaxEntropy2 (double *inputseries, int length, int degree,
    #ifdef GENERATE_INTERMEDIATES
        double **ar,
    #else
        double *coef,
    #endif
        double *per, double *pef, double *h, double *g);
//--E [] 2013/11/02: Sang-Wook Lee

#if 0
// an example of AutoRegression.

#include "stdafx.h"
#include "ar.h"
#include <iostream>
#include <string>
#include <cstdlib>


int main(int argc, char* argv[])
{
	const int MAXCOEFF = 100;

	const int method = MAXENTROPY;  // MAXENTROPY, LEASTSQUARES, MAXENTROPY2.

#if 0
	const std::string data_filename("test1.dat");
	const int degree = 5;  // Maximum degree. degree < MAXCOEFF.
#elif 0
	const std::string data_filename("test2.dat");
	const int degree = 7;  // Maximum degree. degree < MAXCOEFF.
#elif 1
	const std::string data_filename("test3.dat");
	const int degree = 2;  // Maximum degree. degree < MAXCOEFF.
#endif

	// Open the data file.
	FILE *fptr = NULL;
	if (NULL == (fptr = fopen(data_filename.c_str(), "r")))
	{
		std::cerr << "Unable to open data file" << std::endl;
		return EXIT_FAILURE;
	}

	// Read as many points as we can.
	double d, *data = NULL;
	int length = 0;
	while (fscanf(fptr, "%lf", &d) == 1)
	{
		if (NULL == (data = (double *)realloc(data, (length + 1) * sizeof(double))))
		{
			std::cerr << "Memory allocation for data failed" << std::endl;
			return EXIT_FAILURE;
		}

		data[length] = d;
		++length;
	}

	fclose(fptr);
	fptr = NULL;

	std::cout << "Read " << length << " points" << std::endl;

	// allocate for coefficients.
	double *coefficients = NULL;
	if (NULL == (coefficients = (double *)malloc(degree * sizeof(double))))
	{
		std::cerr << "Failed to allocate space for coefficients" << std::endl;
		return EXIT_FAILURE;
	}

	// Calculate and print the coefficients.
	if (!AutoRegression(data, length, degree, coefficients, method))
	{
		std::cerr << "AR routine failed" << std::endl;
		return EXIT_FAILURE;
	}
	for (int i = 0; i < degree; ++i)
		std::cout << coefficients[i] << std::endl;

	free(coefficients);
	coefficients = NULL;
	free(data);
	data = NULL;

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return EXIT_SUCCESS;
}
#endif

bool AutoRegression(
   double   *inputseries,
   int      length,
   int      degree,
   double   *coefficients,
   int      method)
{
   double mean;
   int i, t;            
   double *w=NULL;      /* Input series - mean */
   double *h=NULL; 
   double *g=NULL;      /* Used by mempar() */
   double *per=NULL; 
   double *pef=NULL;      /* Used by mempar() */
   double **ar=NULL;      /* AR coefficients, all degrees */

   /* Allocate space for working variables */
   if ((w = (double *)malloc(length*sizeof(double))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   }
   if ((h = (double *)malloc((degree+1)*sizeof(double))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   }
   if ((g = (double *)malloc((degree+2)*sizeof(double))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   }
   if ((per = (double *)malloc((length+1)*sizeof(double))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   }
   if ((pef = (double *)malloc((length+1)*sizeof(double))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   }

   if ((ar = (double **)malloc((degree+1)*sizeof(double*))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   }
   for (i=0;i<degree+1;i++) {
      if ((ar[i] = (double *)malloc((degree+1)*sizeof(double))) == NULL) {
      	//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
      }
   }

   /* Determine and subtract the mean from the input series */
   mean = 0.0;
   for (t=0;t<length;t++) 
      mean += inputseries[t];
   mean /= (double)length;
   for (t=0;t<length;t++)
      w[t] = inputseries[t] - mean;

   /* Perform the appropriate AR calculation */
   if (method == MAXENTROPY) {

      if (!ARMaxEntropy(w,length,degree,ar,per,pef,h,g)) {
      	//fprintf(stderr,"Max entropy failed - fatal!\n");
		//exit(-1);
		std::cerr << "Max entropy failed - fatal" << std::endl;
		return false;
		}
      for (i=1;i<=degree;i++)
         coefficients[i-1] = -ar[degree][i];

   } else if (method == LEASTSQUARES) {

      if (!ARLeastSquare(w,length,degree,coefficients)) {
      	//fprintf(stderr,"Least squares failed - fatal!\n");
		//exit(-1);
		std::cerr << "Least squares failed - fatal" << std::endl;
		return false;
		}

//--S [] 2013/11/02: Sang-Wook Lee
   } else if (method == MAXENTROPY2) {

      ARMaxEntropy2(w,length,degree,ar,per,pef,h,g);
      //for (i=1;i<=degree;i++)
      //   coefficients[i-1] = -ar[degree][i];
      for (i=0;i<degree;i++)
         coefficients[i] = -ar[degree][i];
//--E [] 2013/11/02: Sang-Wook Lee

   } else {

	     fprintf(stderr,"Unknown method\n");
		//exit(-1);
		std::cerr << "Unknown method" << std::endl;
		return false;

   }

   if (w != NULL)
      free(w);
   if (h != NULL)
      free(h);
   if (g != NULL)
      free(g);
   if (per != NULL)
      free(per);
   if (pef != NULL)
      free(pef);
   if (ar != NULL) {
      for (i=0;i<degree+1;i++)
         if (ar[i] != NULL)
            free(ar[i]);
      free(ar);
   }
      
   return true;
}

/*   
   Previously called mempar()
   Originally in FORTRAN, hence the array offsets of 1, Yuk.
   Original code from Kay, 1988, appendix 8D.
   
   Perform Burg's Maximum Entropy AR parameter estimation
   outputting successive models en passant. Sourced from Alex Sergejew
 
   Two small changes made by NH in November 1998:
   tstarz.h no longer included, just say "typedef double REAL" instead
   Declare ar by "REAL **ar" instead of "REAL ar[MAXA][MAXA]
   
   Further "cleaning" by Paul Bourke.....for personal style only.
*/

bool ARMaxEntropy(
   double *inputseries,int length,int degree,double **ar,
   double *per,double *pef,double *h,double *g)
{
   int j,n,nn,jj;
   double sn,sd;
   double t1,t2;

   for (j=1;j<=length;j++) {
      pef[j] = 0;
      per[j] = 0;
   }
      
   for (nn=2;nn<=degree+1;nn++) {
      n  = nn - 2;
      sn = 0.0;
      sd = 0.0;
      jj = length - n - 1;
      for (j=1;j<=jj;j++) {
         t1 = inputseries[j+n] + pef[j];
         t2 = inputseries[j-1] + per[j];
         sn -= 2.0 * t1 * t2;
         sd += (t1 * t1) + (t2 * t2);
      }
      g[nn] = sn / sd;
      t1 = g[nn];
      if (n != 0) {
         for (j=2;j<nn;j++) 
            h[j] = g[j] + (t1 * g[n - j + 3]);
         for (j=2;j<nn;j++)
            g[j] = h[j];
         jj--;
      }
      for (j=1;j<=jj;j++) {
         per[j] += (t1 * pef[j]) + (t1 * inputseries[j+nn-2]);
         pef[j]  = pef[j+1] + (t1 * per[j+1]) + (t1 * inputseries[j]);
      }

      for (j=2;j<=nn;j++)
         ar[nn-1][j-1] = g[j];
   }
   
   return true;
}

/*
   Least squares method
   Original from Rainer Hegger, Last modified: Aug 13th, 1998
   Modified (for personal style and context) by Paul Bourke
*/
bool ARLeastSquare(
   double   *inputseries,
   int      length,
   int      degree,
   double   *coefficients)
{
   int i,j,k,hj,hi;
   double **mat;

   if ((mat = (double **)malloc(degree*sizeof(double *))) == NULL) {
		//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
	}
   for (i=0;i<degree;i++) {
      if ((mat[i] = (double *)malloc(degree*sizeof(double))) == NULL) {
      	//fprintf(stderr,"Unable to malloc memory - fatal!\n");
		//exit(-1);
		std::cerr << "Unable to malloc memory - fatal" << std::endl;
		return false;
   	}
	}

   for (i=0;i<degree;i++) {
      coefficients[i] = 0.0;
      for (j=0;j<degree;j++)
         mat[i][j] = 0.0;
   }
   for (i=degree-1;i<length-1;i++) {
      hi = i + 1;
      for (j=0;j<degree;j++) {
         hj = i - j;
         coefficients[j] += (inputseries[hi] * inputseries[hj]);
         for (k=j;k<degree;k++)
            mat[j][k] += (inputseries[hj] * inputseries[i-k]);
      }
   }
   for (i=0;i<degree;i++) {
      coefficients[i] /= (length - degree);
      for (j=i;j<degree;j++) {
         mat[i][j] /= (length - degree);
         mat[j][i] = mat[i][j];
      }
   }

   /* Solve the linear equations */
   if (!SolveLE(mat,coefficients,degree)) {
		//fprintf(stderr,"Linear solver failed - fatal!\n");
		std::cerr << "Linear solver failed - fatal" << std::endl;
		//exit(-1);
		return false;
	}
     
   for (i=0;i<degree;i++)
      if (mat[i] != NULL)
         free(mat[i]);
   if (mat != NULL)
        free(mat);

   return true;
}

/*
   Gaussian elimination solver
   Author: Rainer Hegger Last modified: Aug 14th, 1998
   Modified (for personal style and context) by Paul Bourke
*/
bool SolveLE(double **mat,double *vec,unsigned int n)
{
   int i,j,k,maxi;
   double vswap,*mswap,*hvec,max,h,pivot,q;
  
   for (i=0;i<n-1;i++) {
      max = fabs(mat[i][i]);
      maxi = i;
      for (j=i+1;j<n;j++) {
         if ((h = fabs(mat[j][i])) > max) {
            max = h;
            maxi = j;
         }
      }
      if (maxi != i) {
         mswap     = mat[i];
         mat[i]    = mat[maxi];
         mat[maxi] = mswap;
         vswap     = vec[i];
         vec[i]    = vec[maxi];
         vec[maxi] = vswap;
      }
    
      hvec = mat[i];
      pivot = hvec[i];
      if (fabs(pivot) == 0.0) {
         //fprintf(stderr,"Singular matrix - fatal!\n");
		 std::cerr << "Singular matrix - fatal" << std::endl;
         return false;
      }
      for (j=i+1;j<n;j++) {
         q = - mat[j][i] / pivot;
         mat[j][i] = 0.0;
         for (k=i+1;k<n;k++)
            mat[j][k] += q * hvec[k];
         vec[j] += (q * vec[i]);
      }
   }
   vec[n-1] /= mat[n-1][n-1];
   for (i=n-2;i>=0;i--) {
      hvec = mat[i];
      for (j=n-1;j>i;j--)
         vec[i] -= (hvec[j] * vec[j]);
      vec[i] /= hvec[i];
   }
   
   return true;
}

/*   
   Previously called mempar()
   Originally in FORTRAN, hence the array offsets of 1, Yuk.
   Original code from Kay, 1988, appendix 8D.
   
   Perform Burg's Maximum Entropy AR parameter estimation
   outputting (or not) successive models en passant. Sourced from Alex Sergejew
 
   Two small changes made by NH in November 1998:
   tstarz.h no longer included, just say "typedef double REAL" instead
   Declare ar by "REAL **ar" instead of "REAL ar[MAXA][MAXA]
   
   Further "cleaning" by Paul Bourke.....for personal style only.

   Converted to zero-based arrays by Paul Sanders, June 2007
*/

//--S [] 2013/11/02: Sang-Wook Lee
//#define GENERATE_INTERMEDIATES
//--E [] 2013/11/02: Sang-Wook Lee

//--S [] 2013/11/02: Sang-Wook Lee
//void ARMaxEntropy (double *inputseries, int length, int degree,
void ARMaxEntropy2 (double *inputseries, int length, int degree,
//--E [] 2013/11/02: Sang-Wook Lee
    #ifdef GENERATE_INTERMEDIATES
        double **ar,
    #else
        double *coef,
    #endif
        double *per, double *pef, double *h, double *g)
{
    double t1, t2;
    int n;

    for (n = 1; n <= degree; n++)
    {
        double sn = 0.0;
        double sd = 0.0;
        int j;
        int jj = length - n;

        for (j = 0; j < jj; j++)
        {
            t1 = inputseries [j + n] + pef [j];
            t2 = inputseries [j] + per [j];
            sn -= 2.0 * t1 * t2;
            sd += (t1 * t1) + (t2 * t2);
        }

        t1 = g [n] = sn / sd;
        if (n != 1)
        {
            for (j = 1; j < n; j++) 
                h [j] = g [j] + t1 * g [n - j];
            for (j = 1; j < n; j++)
                g [j] = h [j];
            jj--;
        }

        for (j = 0; j < jj; j++)
        {
            per [j] += t1 * pef [j] + t1 * inputseries [j + n];
            pef [j] = pef [j + 1] + t1 * per [j + 1] + t1 * inputseries [j + 1];
        }

#ifdef GENERATE_INTERMEDIATES
        for (j = 0; j < n; j++)
           ar [n][j] = g [j + 1];
#endif
    }

#ifndef GENERATE_INTERMEDIATES
    for (n = 0; n < degree; n++)
        coef [n] = g [n + 1];
#endif
}

#if !defined(__SWL_RND_UTIL__AUTOREGRESSION__H_)
#define __SWL_RND_UTIL__AUTOREGRESSION__H_


// [ref] http://paulbourke.net/miscellaneous/ar/.

bool AutoRegression(double *inputseries, int length, int degree, double *coefficients, int method);


#endif  // __SWL_RND_UTIL__AUTOREGRESSION__H_

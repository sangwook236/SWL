#if !defined(__SWL_RND_UTIL__AUTOREGRESSION__H_)
#define __SWL_RND_UTIL__AUTOREGRESSION__H_


namespace swl {

// [ref] http://paulbourke.net/miscellaneous/ar/.

bool computeAutoRegression(double *inputseries, int length, int degree, double *coefficients, int method);

}  // namespace swl


#endif  // __SWL_RND_UTIL__AUTOREGRESSION__H_

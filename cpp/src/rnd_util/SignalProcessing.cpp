#include "swl/Config.h"
#include "swl/rnd_util/SignalProcessing.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// Signal processing.

// REF [function] >> filter() in Matlab.
/*static*/ void SignalProcessing::filter(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &x, std::vector<double> &y)
{
	// a[0] * y[n] = b[0] * x[n] + b[1] * x[n - 1] + ... + b[nb] * x[n - nb] - a[1] * y[n - 1] - ... - a[na] * y[n - na].

	const size_t na = a.size();
	const size_t nb = b.size();
	const size_t nx = x.size();

	y.reserve(nx);
	if (na == nb)
	{
		for (size_t n = 0; n < nx; ++n)
		{
			double sum = b[0] * x[n];
			for (size_t i = 1; i <= nb && i <= n; ++i)
				sum += b[i] * x[n - i] - a[i] * y[n - i];

			y.push_back(sum / a[0]);
		}
	}
	else
	{
		for (size_t n = 0; n < nx; ++n)
		{
			double sum = 0.0;
			for (size_t i = 0; i <= nb && i <= n; ++i)
				sum += b[i] * x[n - i];
			for (size_t i = 1; i <= na && i <= n; ++i)
				sum -= a[i] * y[n - i];

			y.push_back(sum / a[0]);
		}
	}
}

}  // namespace swl

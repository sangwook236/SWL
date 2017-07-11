#include "swl/Config.h"
#include "swl/math/Statistic.h"
#include <boost/timer/timer.hpp>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void statistic()
{
	{
		//const std::vector<double> sample({ 1., -3., 2., 6., 7., 13., -37., 56. });
		const std::vector<double> sample({ 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999 });

		boost::timer::auto_cpu_timer timer;
		const double &mean = swl::Statistic::mean(sample);
		std::cout << "Mean = " << mean << std::endl;

		const double &sd = swl::Statistic::standardDeviation(sample, mean);
		std::cout << "SD = " << sd << std::endl;

		const double &ssd = swl::Statistic::sampleStandardDeviation(sample, mean);
		std::cout << "Sample SD = " << ssd << std::endl;

		const double &skewness = swl::Statistic::skewness(sample, mean, sd);
		std::cout << "skewness = " << skewness << std::endl;

		const double &kurtosis = swl::Statistic::kurtosis(sample, mean, sd);
		std::cout << "kurtosis = " << kurtosis << std::endl;
	}
}

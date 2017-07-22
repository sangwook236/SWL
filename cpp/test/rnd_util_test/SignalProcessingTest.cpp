//#include "stdafx.h"
#include "swl/Config.h"
//#include "swl/rnd_util/SignalProcessing.h"
#include "swl/math/Statistic.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cassert>
#if defined(_WIN64) || defined(_WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// Signal-to-noise ratio (SNR).
// REF [site] >> https://kr.mathworks.com/help/signal/ref/snr.html
void snr()
{
	const double Fs = 48.0e3;  // Sampling frequency [Hz].
	const double Ts = 1.0 / Fs;  // Sampling interal [sec].
	const size_t numSamples = (size_t)std::floor(Fs * 1.0 + 0.5);
	const double A = 1.0;
	const double a = 0.4;
	const double s = 0.1;

	// The theoretical average power (mean-square) of each complex sinusoid = A^2 / 4.
	// Accounting for the power in the positive and negative frequencies results in an average power of (A^2 / 4) * 2.
	const double powerFundamental = A * A / 2.0;
	const double powerHarmonic = a * a / 2.0;
	const double varNoiseTrue = s * s;

	typedef boost::minstd_rand base_generator_type;
	//typedef boost::mt19937 base_generator_type;
	base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	const double meanGen = 0.0;
	const double sigmaGen = 1.0;
	generator_type normal_gen(baseGenerator, distribution_type(meanGen, sigmaGen));

	baseGenerator.seed(static_cast<unsigned int>(std::time(NULL)));
	generator_type uni_gen(baseGenerator, distribution_type(0, 1));

	std::vector<double> x, fundamentals, harmonics, signals, noises;
	x.reserve(numSamples);
	fundamentals.reserve(numSamples);
	harmonics.reserve(numSamples);
	signals.reserve(numSamples);
	noises.reserve(numSamples);
	for (size_t idx = 0; idx < numSamples; ++idx)
	{
		const double t = Ts * idx;
		const double fundamental = A * std::cos(2.0 * M_PI * 1000.0 * t);
		const double harmonic = a * std::sin(2.0 * M_PI * 2000.0 * t);
		const double noise = s * normal_gen();
		fundamentals.push_back(fundamental);
		harmonics.push_back(harmonic);
		signals.push_back(fundamental + harmonic);
		noises.push_back(noise);
		x.push_back(fundamental + harmonic + noise);
	}

	const double meanX = swl::Statistic::mean(x);
	const double varX = swl::Statistic::variance(x, meanX);
	const double meanFundamental = swl::Statistic::mean(fundamentals);
	const double varFundamental = swl::Statistic::variance(fundamentals, meanFundamental);
	const double meanHarmonic = swl::Statistic::mean(harmonics);
	const double varHarmonic = swl::Statistic::variance(harmonics, meanHarmonic);
	const double meanSignal = swl::Statistic::mean(signals);
	const double varSignal = swl::Statistic::variance(signals, meanSignal);
	const double meanNoise = swl::Statistic::mean(noises);
	const double varNoise = swl::Statistic::variance(noises, meanNoise);

	// Signal-to-noise ratio (SNR).
	//const double SNR = 10.0 * std::log10(varX / varNoise);
	//const double SNR = 10.0 * std::log10(varSignal / varNoise);
	const double SNR = 10.0 * std::log10(varFundamental / varNoise);
	const double defSNR = 10.0 * std::log10(powerFundamental / varNoiseTrue);  // By definition.

	std::cout << "Signal-to-noise ratio (SNR) = " << SNR << std::endl;
	std::cout << "Signal-to-noise ratio (SNR) = " << defSNR << std::endl;

	// Total harmonic distortion (THD).
	const double THD = 10.0 * std::log10(varHarmonic / varFundamental);
	const double defTHD = 10.0 * std::log10(powerHarmonic / powerFundamental);  // By definition.

	std::cout << "Total harmonic distortion (THD) = " << THD << std::endl;
	std::cout << "Total harmonic distortion (THD) = " << defTHD << std::endl;

	// Signal to noise and distortion ratio (SINAD).
	const double SINAD = 10.0 * std::log10(varFundamental / (varHarmonic + varNoise));
	const double defSINAD = 10.0 * std::log10(powerFundamental / (powerHarmonic + varNoiseTrue));  // By definition.

	std::cout << "Signal to noise and distortion ratio (SINAD) = " << SINAD << std::endl;
	std::cout << "Signal to noise and distortion ratio (SINAD) = " << defSINAD << std::endl;
}

}  // namespace local
}  // unnamed namespace

void signal_processing()
{
	// Signal-to-noise ratio (SNR).
	local::snr();
}

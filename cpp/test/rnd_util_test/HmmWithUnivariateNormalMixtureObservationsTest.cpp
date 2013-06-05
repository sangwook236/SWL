//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/HmmWithUnivariateNormalMixtureObservations.h"
#include <boost/smart_ptr.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <ctime>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#define __TEST_HMM_MODEL 0
//#define __TEST_HMM_MODEL 1
//#define __TEST_HMM_MODEL 2
//#define __USE_SPECIFIED_VALUE_FOR_RANDOM_SEED 1


namespace {
namespace local {

void model_reading_and_writing()
{
	// reading a model
	{
		boost::scoped_ptr<swl::CDHMM> cdhmm;

#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		const bool retval = cdhmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi, A, & alpha
		cdhmm->normalizeModelParameters();

		cdhmm->writeModel(std::cout);
	}

	// writing a model
	{
		boost::scoped_ptr<swl::CDHMM> cdhmm;

#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		const double arrPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double arrA[] = {
			0.2, 0.5, 0.3,
			0.45, 0.3, 0.25,
			0.5, 0.15, 0.35
		};
		const double arrAlpha[] = {
			0.4, 0.6,
			0.35, 0.65,
			0.7, 0.3
		};
		const double arrMu[] = {
			0.0, -200.0,
			1500.0, 2000.0,
			-2000.0, -4000.0
		};
		const double arrSigma[] = {
			5.0, 3.0,
			2.0, 10.0,
			6.0, 8.5
		};

		//
		std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test0_writing.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		const double arrPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double arrA[] = {
			0.9,  0.05, 0.05,
			0.45, 0.1,  0.45,
			0.45, 0.45, 0.1
		};
		const double arrAlpha[] = {
			0.7, 0.3,
			0.2, 0.8,
			0.5, 0.5
		};
		const double arrMu[] = {
			0.0, 5.0,
			30.0, 40.0,
			-20.0, -25.0
		};
		const double arrSigma[] = {
			3.0, 1.0,
			2.0, 2.0,
			1.5, 2.5
		};

		//
		std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test1_writing.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		const double arrPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double arrA[] = {
			0.5, 0.2,  0.3,
			0.2, 0.4,  0.4,
			0.1, 0.45, 0.45
		};
		const double arrAlpha[] = {
			0.2, 0.8,
			0.6, 0.4,
			0.75, 0.25
		};
		const double arrMu[] = {
			0.0, -5.0,
			-30.0, -35.0,
			20.0, 15.0
		};
		const double arrSigma[] = {
			1.0, 2.0,
			2.0, 4.0,
			0.5, 1.5
		};

		//
		std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test2_writing.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		swl::HmmWithUnivariateNormalMixtureObservations::dvector_type pi(boost::numeric::ublas::vector<double, std::vector<double> >(K, std::vector<double>(arrPi, arrPi + K)));
		swl::HmmWithUnivariateNormalMixtureObservations::dmatrix_type A(boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major, std::vector<double> >(K, K, std::vector<double>(arrA, arrA + K * K)));
		swl::HmmWithUnivariateNormalMixtureObservations::dmatrix_type alphas(boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major, std::vector<double> >(K, C, std::vector<double>(arrAlpha, arrAlpha + K * C)));
		swl::HmmWithUnivariateNormalMixtureObservations::dmatrix_type mus(boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major, std::vector<double> >(K, C, std::vector<double>(arrMu, arrMu + K * C)));
		swl::HmmWithUnivariateNormalMixtureObservations::dmatrix_type sigmas(boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major, std::vector<double> >(K, C, std::vector<double>(arrSigma, arrSigma + K * C)));
		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C, pi, A, alphas, mus, sigmas));

		const bool retval = cdhmm->writeModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model writing error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}
	}
}

void observation_sequence_generation(const bool outputToFile)
{
	boost::scoped_ptr<swl::CDHMM> cdhmm;

	// read a model
	{
#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		const bool retval = cdhmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi, A, & alpha
		cdhmm->normalizeModelParameters();

		//cdhmm->writeModel(std::cout);
	}

	// generate a sample sequence
	{
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
		const unsigned int seed = 34586u;
#else
		const unsigned int seed = (unsigned int)std::time(NULL);
#endif
		std::srand(seed);
		std::cout << "random seed: " << seed << std::endl;

		if (outputToFile)
		{
#if __TEST_HMM_MODEL == 0

#if 1
			const size_t N = 50;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test0_50.seq");
#elif 0
			const size_t N = 100;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test0_100.seq");
#elif 0
			const size_t N = 1500;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test0_1500.seq");
#endif

#elif __TEST_HMM_MODEL == 1

#if 1
			const size_t N = 50;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test1_50.seq");
#elif 0
			const size_t N = 100;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test1_100.seq");
#elif 0
			const size_t N = 1500;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test1_1500.seq");
#endif

#elif __TEST_HMM_MODEL == 2

#if 1
			const size_t N = 50;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test2_50.seq");
#elif 0
			const size_t N = 100;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test2_100.seq");
#elif 0
			const size_t N = 1500;
			std::ofstream stream("..\\data\\hmm\\uni_normal_mixture_test2_1500.seq");
#endif

#endif
			if (!stream)
			{
				std::ostringstream stream;
				stream << "file not found at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}

			swl::CDHMM::dmatrix_type observations(N, cdhmm->getObservationDim(), 0.0);
			swl::CDHMM::uivector_type states(N, (unsigned int)-1);
			cdhmm->generateSample(N, observations, states, seed);

#if 0
			// output states
			for (size_t n = 0; n < N; ++n)
				std::cout << states[n] << ' ';
			std::cout << std::endl;
#endif

			// write a sample sequence
			swl::CDHMM::writeSequence(stream, observations);
		}
		else
		{
			const size_t N = 100;

			swl::CDHMM::dmatrix_type observations(N, cdhmm->getObservationDim(), 0.0);
			swl::CDHMM::uivector_type states(N, (unsigned int)-1);
			cdhmm->generateSample(N, observations, states, seed);

#if 0
			// output states
			for (size_t n = 0; n < N; ++n)
				std::cout << states[n] << ' ';
			std::cout << std::endl;
#endif

			// write a sample sequence
			swl::CDHMM::writeSequence(std::cout, observations);
		}
	}
}

void observation_sequence_reading_and_writing()
{
	swl::CDHMM::dmatrix_type observations;
	size_t N = 0;  // length of observation sequence, N

#if __TEST_HMM_MODEL == 0

#if 1
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_50.seq");
#elif 0
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_100.seq");
#elif 0
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_1500.seq");
#else
	std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 1

#if 1
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_50.seq");
#elif 0
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_100.seq");
#elif 0
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_1500.seq");
#else
	std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 2

#if 1
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_50.seq");
#elif 0
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_100.seq");
#elif 0
	std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_1500.seq");
#else
	std::istream stream = std::cin;
#endif

#endif
	if (!stream)
	{
		std::ostringstream stream;
		stream << "file not found at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// read a observation sequence
	size_t D = 0;
	const bool retval = swl::CDHMM::readSequence(stream, N, D, observations);
	if (!retval)
	{
		std::ostringstream stream;
		stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// write a observation sequence
	swl::CDHMM::writeSequence(std::cout, observations);
}

void forward_algorithm()
{
	boost::scoped_ptr<swl::CDHMM> cdhmm;

	// read a model
	{
#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		const bool retval = cdhmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi, A, & alpha
		cdhmm->normalizeModelParameters();

		//cdhmm->writeModel(std::cout);
	}

	// read a observation sequence
	swl::CDHMM::dmatrix_type observations;
	size_t N = 0;  // length of observation sequence, N
	{
#if __TEST_HMM_MODEL == 0

#if 1
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_50.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_100.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_1500.seq");
#else
		std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 1

#if 1
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_50.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_100.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_1500.seq");
#else
		std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 2

#if 1
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_50.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_100.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_1500.seq");
#else
		std::istream stream = std::cin;
#endif

#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		size_t D = 0;
		const bool retval = swl::CDHMM::readSequence(stream, N, D, observations);
		if (!retval || cdhmm->getObservationDim() != D)
		{
			std::ostringstream stream;
			stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}
	}

	const size_t K = cdhmm->getStateDim();

	// forward algorithm without scaling
	{
		swl::CDHMM::dmatrix_type alpha(N, K, 0.0);
		double probability = 0.0;
		cdhmm->runForwardAlgorithm(N, observations, alpha, probability);

		//
		std::cout << "------------------------------------" << std::endl;
		std::cout << "forward algorithm without scaling" << std::endl;
		std::cout << "\tlog prob(observations | model) = " << std::scientific << std::log(probability) << std::endl;
	}

	// forward algorithm with scaling
	{
		swl::CDHMM::dvector_type scale(N, 0.0);
		swl::CDHMM::dmatrix_type alpha(N, K, 0.0);
		double logProbability = 0.0;
		cdhmm->runForwardAlgorithm(N, observations, scale, alpha, logProbability);

		//
		std::cout << "------------------------------------" << std::endl;
		std::cout << "forward algorithm with scaling" << std::endl;
		std::cout << "\tlog prob(observations | model) = " << std::scientific << logProbability << std::endl;
	}
}

void backward_algorithm()
{
	throw std::runtime_error("not yet implemented");
}

void viterbi_algorithm()
{
	boost::scoped_ptr<swl::CDHMM> cdhmm;

	// read a model
	{
#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		const bool retval = cdhmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi, A, & alpha
		cdhmm->normalizeModelParameters();

		//cdhmm->writeModel(std::cout);
	}

	// read a observation sequence
	swl::CDHMM::dmatrix_type observations;
	size_t N = 0;  // length of observation sequence, N
	{
#if __TEST_HMM_MODEL == 0

#if 1
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_50.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_100.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_1500.seq");
#else
		std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 1

#if 1
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_50.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_100.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_1500.seq");
#else
		std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 2

#if 1
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_50.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_100.seq");
#elif 0
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_1500.seq");
#else
		std::istream stream = std::cin;
#endif

#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		size_t D = 0;
		const bool retval = swl::CDHMM::readSequence(stream, N, D, observations);
		if (!retval || cdhmm->getObservationDim() != D)
		{
			std::ostringstream stream;
			stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}
	}

	const size_t K = cdhmm->getStateDim();

	// Viterbi algorithm using direct probabilities
	{
		swl::CDHMM::dmatrix_type delta(N, K, 0.0);
		swl::CDHMM::uimatrix_type psi(N, K, (unsigned int)-1);
		swl::CDHMM::uivector_type states(N, (unsigned int)-1);
		double probability = 0.0;
		cdhmm->runViterbiAlgorithm(N, observations, delta, psi, states, probability, false);

		//
		std::cout << "------------------------------------" << std::endl;
		std::cout << "Viterbi algorithm using direct probabilities" << std::endl;
		std::cout << "\tViterbi MLE log prob = " << std::scientific << std::log(probability) << std::endl;
		std::cout << "\toptimal state sequence:" << std::endl;
		for (size_t n = 0; n < N; ++n)
			std::cout << states[n] << ' ';
		std::cout << std::endl;
	}

	// Viterbi algorithm using log probabilities
	{
		swl::CDHMM::dmatrix_type delta(N, K, 0.0);
		swl::CDHMM::uimatrix_type psi(N, K, (unsigned int)-1);
		swl::CDHMM::uivector_type states(N, (unsigned int)-1);
		double logProbability = 0.0;
		cdhmm->runViterbiAlgorithm(N, observations, delta, psi, states, logProbability, true);

		//
		std::cout << "------------------------------------" << std::endl;
		std::cout << "Viterbi algorithm using log probabilities" << std::endl;
		std::cout << "\tViterbi MLE log prob = " << std::scientific << logProbability << std::endl;
		std::cout << "\toptimal state sequence:" << std::endl;
		for (size_t n = 0; n < N; ++n)
			std::cout << states[n] << ' ';
		std::cout << std::endl;
	}
}

void em_learning_by_mle()
{
	boost::scoped_ptr<swl::CDHMM> cdhmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed.
*/

	// initialize a model
	const int initialization_mode = 1;
	if (1 == initialization_mode)
	{
#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		const bool retval = cdhmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi, A, & alpha
		cdhmm->normalizeModelParameters();

		//cdhmm->writeModel(std::cout);
	}
	else if (2 == initialization_mode)
	{
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
		const unsigned int seed = 34586u;
#else
		const unsigned int seed = (unsigned int)std::time(NULL);
#endif
		std::srand(seed);
		std::cout << "random seed: " << seed << std::endl;

		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		// the total number of parameters of observation density = K * C * D * 2
		std::vector<double> lowerBounds, upperBounds;
		const size_t numParameters = K * C * 1 * 2;
		lowerBounds.reserve(numParameters);
		upperBounds.reserve(numParameters);
		// means
		for (size_t i = 0; i < K * C; ++i)
		{
			lowerBounds.push_back(-10000.0);
			upperBounds.push_back(10000.0);
		}
		// standard deviations: sigma > 0
		const double small = 1.0e-10;
		for (size_t i = K * C; i < numParameters; ++i)
		{
			lowerBounds.push_back(small);
			upperBounds.push_back(10000.0);
		}
		cdhmm->initializeModel(lowerBounds, upperBounds);
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	// for a single observation sequence
	{
		// read a observation sequence
		swl::CDHMM::dmatrix_type observations;
		size_t N = 0;  // length of observation sequence, N
		{
#if __TEST_HMM_MODEL == 0

#if 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_50.seq");
#elif 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_100.seq");
#elif 1
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 1

#if 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_50.seq");
#elif 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_100.seq");
#elif 1
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 2

#if 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_50.seq");
#elif 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_100.seq");
#elif 1
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#endif
			if (!stream)
			{
				std::ostringstream stream;
				stream << "file not found at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}

			size_t D = 0;
			const bool retval = swl::CDHMM::readSequence(stream, N, D, observations);
			if (!retval || cdhmm->getObservationDim() != D)
			{
				std::ostringstream stream;
				stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}
		}

		// Baum-Welch algorithm
		{
			const double terminationTolerance = 0.001;
			const size_t maxIteration = 1000;
			size_t numIteration = (size_t)-1;
			double initLogProbability = 0.0, finalLogProbability = 0.0;
			cdhmm->estimateParametersByML(N, observations, terminationTolerance, maxIteration, numIteration, initLogProbability, finalLogProbability);

			// normalize pi, A, & alpha
			//cdhmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "Baum-Welch algorithm for a single observation sequence" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observations | initial model) = " << std::scientific << initLogProbability << std::endl;
			std::cout << "\tlog prob(observations | estimated model) = " << std::scientific << finalLogProbability << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdhmm->writeModel(std::cout);
		}
	}

	// for multiple independent observation sequences
	{
		// read a observation sequence
		std::vector<swl::CDHMM::dmatrix_type> observationSequences;
		std::vector<size_t> Ns;  // lengths of observation sequences
		{
#if __TEST_HMM_MODEL == 0
			const size_t R = 3;  // number of observations sequences
			const std::string observationSequenceFiles[] = {
				"..\\data\\hmm\\uni_normal_mixture_test0_50.seq",
				"..\\data\\hmm\\uni_normal_mixture_test0_100.seq",
				"..\\data\\hmm\\uni_normal_mixture_test0_1500.seq"
			};
#elif __TEST_HMM_MODEL == 1
			const size_t R = 3;  // number of observations sequences
			const std::string observationSequenceFiles[] = {
				"..\\data\\hmm\\uni_normal_mixture_test1_50.seq",
				"..\\data\\hmm\\uni_normal_mixture_test1_100.seq",
				"..\\data\\hmm\\uni_normal_mixture_test1_1500.seq"
			};
#elif __TEST_HMM_MODEL == 2
			const size_t R = 3;  // number of observations sequences
			const std::string observationSequenceFiles[] = {
				"..\\data\\hmm\\uni_normal_mixture_test2_50.seq",
				"..\\data\\hmm\\uni_normal_mixture_test2_100.seq",
				"..\\data\\hmm\\uni_normal_mixture_test2_1500.seq"
			};
#endif
			observationSequences.resize(R);
			Ns.resize(R);
			for (size_t r = 0; r < R; ++r)
			{
				std::ifstream stream(observationSequenceFiles[r].c_str());
				if (!stream)
				{
					std::ostringstream stream;
					stream << "file not found at " << __LINE__ << " in " << __FILE__;
					throw std::runtime_error(stream.str().c_str());
					return;
				}

				size_t D = 0;
				const bool retval = swl::CDHMM::readSequence(stream, Ns[r], D, observationSequences[r]);
				if (!retval || cdhmm->getObservationDim() != D)
				{
					std::ostringstream stream;
					stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
					throw std::runtime_error(stream.str().c_str());
					return;
				}
			}
		}

		const size_t R = observationSequences.size();  // number of observations sequences

		// Baum-Welch algorithm
		{
			const double terminationTolerance = 0.001;
			const size_t maxIteration = 1000;
			size_t numIteration = (size_t)-1;
			std::vector<double> initLogProbabilities(R, 0.0), finalLogProbabilities(R, 0.0);
			cdhmm->estimateParametersByML(Ns, observationSequences, terminationTolerance, maxIteration, numIteration, initLogProbabilities, finalLogProbabilities);

			// normalize pi, A, & alpha
			//cdhmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "Baum-Welch algorithm for multiple independent observation sequences" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observation sequences | initial model):" << std::endl;
			std::cout << "\t\t";
			for (size_t r = 0; r < R; ++r)
				std::cout << std::scientific << initLogProbabilities[r] << ' ';
			std::cout << std::endl;
			std::cout << "\tlog prob(observation sequences | estimated model):" << std::endl;
			std::cout << "\t\t";
			for (size_t r = 0; r < R; ++r)
				std::cout << std::scientific << finalLogProbabilities[r] << ' ';
			std::cout << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdhmm->writeModel(std::cout);
		}
	}
}

void em_learning_by_map()
{
	boost::scoped_ptr<swl::CDHMM> cdhmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed.
*/

	// initialize a model
	const int initialization_mode = 1;
	if (1 == initialization_mode)
	{
#if __TEST_HMM_MODEL == 0
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0.cdhmm");
#elif __TEST_HMM_MODEL == 1
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1.cdhmm");
#elif __TEST_HMM_MODEL == 2
		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		//
		std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2.cdhmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		const bool retval = cdhmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi, A, & alpha
		cdhmm->normalizeModelParameters();

		//cdhmm->writeModel(std::cout);
	}
	else if (2 == initialization_mode)
	{
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
		const unsigned int seed = 34586u;
#else
		const unsigned int seed = (unsigned int)std::time(NULL);
#endif
		std::srand(seed);
		std::cout << "random seed: " << seed << std::endl;

		const size_t K = 3;  // the dimension of hidden states
		//const size_t D = 1;  // the dimension of observation symbols
		const size_t C = 2;  // the number of mixture components

		cdhmm.reset(new swl::HmmWithUnivariateNormalMixtureObservations(K, C));

		// the total number of parameters of observation density = K * C * D * 2
		std::vector<double> lowerBounds, upperBounds;
		const size_t numParameters = K * C * 1 * 2;
		lowerBounds.reserve(numParameters);
		upperBounds.reserve(numParameters);
		// means
		for (size_t i = 0; i < K * C; ++i)
		{
			lowerBounds.push_back(-10000.0);
			upperBounds.push_back(10000.0);
		}
		// standard deviations: sigma > 0
		const double small = 1.0e-10;
		for (size_t i = K * C; i < numParameters; ++i)
		{
			lowerBounds.push_back(small);
			upperBounds.push_back(10000.0);
		}
		cdhmm->initializeModel(lowerBounds, upperBounds);
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	// for a single observation sequence
	{
		// read a observation sequence
		swl::CDHMM::dmatrix_type observations;
		size_t N = 0;  // length of observation sequence, N
		{
#if __TEST_HMM_MODEL == 0

#if 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_50.seq");
#elif 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_100.seq");
#elif 1
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test0_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 1

#if 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_50.seq");
#elif 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_100.seq");
#elif 1
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test1_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_HMM_MODEL == 2

#if 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_50.seq");
#elif 0
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_100.seq");
#elif 1
			std::ifstream stream("..\\data\\hmm\\uni_normal_mixture_test2_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#endif
			if (!stream)
			{
				std::ostringstream stream;
				stream << "file not found at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}

			size_t D = 0;
			const bool retval = swl::CDHMM::readSequence(stream, N, D, observations);
			if (!retval || cdhmm->getObservationDim() != D)
			{
				std::ostringstream stream;
				stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}
		}

		// Baum-Welch algorithm
		{
			const double terminationTolerance = 0.001;
			const size_t maxIteration = 1000;
			size_t numIteration = (size_t)-1;
			double initLogProbability = 0.0, finalLogProbability = 0.0;
			cdhmm->estimateParametersByML(N, observations, terminationTolerance, maxIteration, numIteration, initLogProbability, finalLogProbability);

			// normalize pi, A, & alpha
			//cdhmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "Baum-Welch algorithm for a single observation sequence" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observations | initial model) = " << std::scientific << initLogProbability << std::endl;
			std::cout << "\tlog prob(observations | estimated model) = " << std::scientific << finalLogProbability << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdhmm->writeModel(std::cout);
		}
	}

	// for multiple independent observation sequences
	{
		// read a observation sequence
		std::vector<swl::CDHMM::dmatrix_type> observationSequences;
		std::vector<size_t> Ns;  // lengths of observation sequences
		{
#if __TEST_HMM_MODEL == 0
			const size_t R = 3;  // number of observations sequences
			const std::string observationSequenceFiles[] = {
				"..\\data\\hmm\\uni_normal_mixture_test0_50.seq",
				"..\\data\\hmm\\uni_normal_mixture_test0_100.seq",
				"..\\data\\hmm\\uni_normal_mixture_test0_1500.seq"
			};
#elif __TEST_HMM_MODEL == 1
			const size_t R = 3;  // number of observations sequences
			const std::string observationSequenceFiles[] = {
				"..\\data\\hmm\\uni_normal_mixture_test1_50.seq",
				"..\\data\\hmm\\uni_normal_mixture_test1_100.seq",
				"..\\data\\hmm\\uni_normal_mixture_test1_1500.seq"
			};
#elif __TEST_HMM_MODEL == 2
			const size_t R = 3;  // number of observations sequences
			const std::string observationSequenceFiles[] = {
				"..\\data\\hmm\\uni_normal_mixture_test2_50.seq",
				"..\\data\\hmm\\uni_normal_mixture_test2_100.seq",
				"..\\data\\hmm\\uni_normal_mixture_test2_1500.seq"
			};
#endif
			observationSequences.resize(R);
			Ns.resize(R);
			for (size_t r = 0; r < R; ++r)
			{
				std::ifstream stream(observationSequenceFiles[r].c_str());
				if (!stream)
				{
					std::ostringstream stream;
					stream << "file not found at " << __LINE__ << " in " << __FILE__;
					throw std::runtime_error(stream.str().c_str());
					return;
				}

				size_t D = 0;
				const bool retval = swl::CDHMM::readSequence(stream, Ns[r], D, observationSequences[r]);
				if (!retval || cdhmm->getObservationDim() != D)
				{
					std::ostringstream stream;
					stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
					throw std::runtime_error(stream.str().c_str());
					return;
				}
			}
		}

		const size_t R = observationSequences.size();  // number of observations sequences

		// Baum-Welch algorithm
		{
			const double terminationTolerance = 0.001;
			const size_t maxIteration = 1000;
			size_t numIteration = (size_t)-1;
			std::vector<double> initLogProbabilities(R, 0.0), finalLogProbabilities(R, 0.0);
			cdhmm->estimateParametersByML(Ns, observationSequences, terminationTolerance, maxIteration, numIteration, initLogProbabilities, finalLogProbabilities);

			// normalize pi, A, & alpha
			//cdhmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "Baum-Welch algorithm for multiple independent observation sequences" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observation sequences | initial model):" << std::endl;
			std::cout << "\t\t";
			for (size_t r = 0; r < R; ++r)
				std::cout << std::scientific << initLogProbabilities[r] << ' ';
			std::cout << std::endl;
			std::cout << "\tlog prob(observation sequences | estimated model):" << std::endl;
			std::cout << "\t\t";
			for (size_t r = 0; r < R; ++r)
				std::cout << std::scientific << finalLogProbabilities[r] << ' ';
			std::cout << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdhmm->writeModel(std::cout);
		}
	}
}

}  // namespace local
}  // unnamed namespace

void hmm_with_univariate_normal_mixture_observation_densities()
{
	std::cout << "===== CDHMM w/ univariate normal mixture observation densities =====" << std::endl;

	//local::model_reading_and_writing();
	//const bool outputToFile = false;
	//local::observation_sequence_generation(outputToFile);
	//local::observation_sequence_reading_and_writing();

	//local::forward_algorithm();
	//local::backward_algorithm();  // not yet implemented
	//local::viterbi_algorithm();

	//local::em_learning_by_mle();
	local::em_learning_by_map();
}

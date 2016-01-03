//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/MultivariateNormalMixtureModel.h"
#include <gsl/gsl_rng.h>
#include <boost/smart_ptr.hpp>
#include <fstream>
#include <iostream>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


//#define __TEST_MIXTURE_MODEL 1
#define __TEST_MIXTURE_MODEL 2

//#define __USE_SPECIFIED_VALUE_FOR_RANDOM_SEED 1


namespace {
namespace local {

void model_reading_and_writing()
{
	// reading a model.
	{
		boost::scoped_ptr<swl::ContinuousDensityMixtureModel> cdmm;

#if __TEST_MIXTURE_MODEL == 1
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1.cdmm");
#elif __TEST_MIXTURE_MODEL == 2
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2.cdmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D));

		const bool retval = cdmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi.
		cdmm->normalizeModelParameters();

		cdmm->writeModel(std::cout);
	}

	// writing a model.
	{
		boost::scoped_ptr<swl::ContinuousDensityMixtureModel> cdmm;

#if __TEST_MIXTURE_MODEL == 1
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		const double arrPi[] = {
			0.25, 0.60, 0.15
		};
		const double arrMu[] = {
			0.0, 5.0,
			30.0, 40.0,
			-20.0, -25.0
		};
		const double arrCov[] = {
			8.0, 0.0,  0.0, 10.0,
			12.0, 3.0,  3.0, 4.0,
			11.5, 6.0,  6.0, 12.5
		};

		//
		std::ofstream stream("./data/mixture_model/bi_normal_mixture_test1_writing.cdmm");
#elif __TEST_MIXTURE_MODEL == 2
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		const double arrPi[] = {
			0.25, 0.60, 0.15
		};
		const double arrMu[] = {
			0.0, -5.0,
			-30.0, -35.0,
			20.0, 15.0
		};
		const double arrCov[] = {
			18.0, 0.0,  0.0, 12.0,
			6.0, 6.0,  6.0, 14.0,
			13.5, 3.0,  3.0, 8.5
		};

		//
		std::ofstream stream("./data/mixture_model/bi_normal_mixture_test2_writing.cdmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		std::vector<double> pi(arrPi, arrPi + K);
		std::vector<swl::MultivariateNormalMixtureModel::dvector_type> mus(K, swl::MultivariateNormalMixtureModel::dvector_type(D, 0.0));
		std::vector<swl::MultivariateNormalMixtureModel::dmatrix_type> covs(K, swl::MultivariateNormalMixtureModel::dmatrix_type(D, D, 1.0));
		std::size_t mm = 0, cc = 0;
		for (std::size_t k = 0; k < K; ++k)
		{
			swl::MultivariateNormalMixtureModel::dvector_type &mu = mus[k];
			swl::MultivariateNormalMixtureModel::dmatrix_type &cov = covs[k];
			for (std::size_t d = 0; d < D; ++d, ++mm)
			{
				mu[d] = arrMu[mm];
				for (std::size_t i = 0; i < D; ++i, ++cc)
					cov(d, i) = arrCov[cc];
			}
		}
		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D, pi, mus, covs));

		const bool retval = cdmm->writeModel(stream);
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
	boost::scoped_ptr<swl::ContinuousDensityMixtureModel> cdmm;

	// read a model.
	{
#if __TEST_MIXTURE_MODEL == 1
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1.cdmm");
#elif __TEST_MIXTURE_MODEL == 2
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2.cdmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D));

		const bool retval = cdmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi.
		cdmm->normalizeModelParameters();

		//cdmm->writeModel(std::cout);
	}

	// generate a sample sequence.
	{
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
		const unsigned int seed = 34586u;
#else
		const unsigned int seed = (unsigned int)std::time(NULL);
#endif
		std::srand(seed);
		gsl_rng_env_setup();
		std::cout << "random seed: " << seed << std::endl;

		if (outputToFile)
		{
#if __TEST_MIXTURE_MODEL == 1

#if 1
			const std::size_t N = 50;
			std::ofstream stream("./data/mixture_model/bi_normal_mixture_test1_50.seq");
#elif 0
			const std::size_t N = 100;
			std::ofstream stream("./data/mixture_model/bi_normal_mixture_test1_100.seq");
#elif 0
			const std::size_t N = 1500;
			std::ofstream stream("./data/mixture_model/bi_normal_mixture_test1_1500.seq");
#endif

#elif __TEST_MIXTURE_MODEL == 2

#if 1
			const std::size_t N = 50;
			std::ofstream stream("./data/mixture_model/bi_normal_mixture_test2_50.seq");
#elif 0
			const std::size_t N = 100;
			std::ofstream stream("./data/mixture_model/bi_normal_mixture_test2_100.seq");
#elif 0
			const std::size_t N = 1500;
			std::ofstream stream("./data/mixture_model/bi_normal_mixture_test2_1500.seq");
#endif

#endif
			if (!stream)
			{
				std::ostringstream stream;
				stream << "file not found at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}

			swl::ContinuousDensityMixtureModel::dmatrix_type observations(N, cdmm->getObservationDim(), 0.0);
			std::vector<unsigned int> states(N, (unsigned int)-1);
			cdmm->generateSample(N, observations, states, seed);

#if 0
			// output states.
			for (std::size_t n = 0; n < N; ++n)
				std::cout << states[n] << ' ';
			std::cout << std::endl;
#endif

			// write a sample sequence.
			swl::ContinuousDensityMixtureModel::writeSequence(stream, observations);
		}
		else
		{
			const std::size_t N = 100;

			swl::ContinuousDensityMixtureModel::dmatrix_type observations(N, cdmm->getObservationDim(), 0.0);
			std::vector<unsigned int> states(N, (unsigned int)-1);
			cdmm->generateSample(N, observations, states, seed);

#if 0
			// output states.
			for (std::size_t n = 0; n < N; ++n)
				std::cout << states[n] << ' ';
			std::cout << std::endl;
#endif

			// write a sample sequence.
			swl::ContinuousDensityMixtureModel::writeSequence(std::cout, observations);
		}
	}
}

void observation_sequence_reading_and_writing()
{
	swl::ContinuousDensityMixtureModel::dmatrix_type observations;
	std::size_t N = 0;  // length of observation sequence, N.

#if __TEST_MIXTURE_MODEL == 1

#if 1
	std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_50.seq");
#elif 0
	std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_100.seq");
#elif 0
	std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_1500.seq");
#else
	std::istream stream = std::cin;
#endif

#elif __TEST_MIXTURE_MODEL == 2

#if 1
	std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_50.seq");
#elif 0
	std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_100.seq");
#elif 0
	std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_1500.seq");
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

	// read a observation sequence.
	std::size_t D = 0;
	const bool retval = swl::ContinuousDensityMixtureModel::readSequence(stream, N, D, observations);
	if (!retval)
	{
		std::ostringstream stream;
		stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
		throw std::runtime_error(stream.str().c_str());
		return;
	}

	// write a observation sequence.
	swl::ContinuousDensityMixtureModel::writeSequence(std::cout, observations);
}

void ml_learning_by_em()
{
	boost::scoped_ptr<swl::ContinuousDensityMixtureModel> cdmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed.
*/

	// initialize a model.
	const int initialization_mode = 1;
	if (1 == initialization_mode)
	{
#if __TEST_MIXTURE_MODEL == 1
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1.cdmm");
#elif __TEST_MIXTURE_MODEL == 2
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2.cdmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D));

		const bool retval = cdmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi.
		cdmm->normalizeModelParameters();

		//cdmm->writeModel(std::cout);
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

		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D));

		// the total number of parameters of observation density = K * (D + D * D).
		std::vector<double> lowerBounds, upperBounds;
		const std::size_t numParameters = K * (D + D * D);
		lowerBounds.reserve(numParameters);
		upperBounds.reserve(numParameters);
		// mean vector & convariance matrix.
		for (std::size_t i = 0; i < numParameters; ++i)
		{
			lowerBounds.push_back(-1000.0);
			upperBounds.push_back(1000.0);
		}
		cdmm->initializeModel(lowerBounds, upperBounds);
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	//
	{
		// read a observation sequence.
		swl::ContinuousDensityMixtureModel::dmatrix_type observations;
		std::size_t N = 0;  // length of observation sequence, N.
		{
#if __TEST_MIXTURE_MODEL == 1

#if 0
			std::ifstream stream("./data/mixture_model/ni_normal_mixture_test1_50.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_100.seq");
#elif 1
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_MIXTURE_MODEL == 2

#if 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_50.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_100.seq");
#elif 1
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_1500.seq");
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

			std::size_t D = 0;
			const bool retval = swl::ContinuousDensityMixtureModel::readSequence(stream, N, D, observations);
			if (!retval || cdmm->getObservationDim() != D)
			{
				std::ostringstream stream;
				stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}
		}

		// EM algorithm.
		{
			const double terminationTolerance = 0.001;
			const std::size_t maxIteration = 1000;
			std::size_t numIteration = (std::size_t)-1;
			double initLogProbability = 0.0, finalLogProbability = 0.0;
			cdmm->trainByML(N, observations, terminationTolerance, maxIteration, numIteration, initLogProbability, finalLogProbability);

			// normalize pi.
			//cdmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "EM algorithm" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observations | initial model) = " << std::scientific << initLogProbability << std::endl;
			std::cout << "\tlog prob(observations | estimated model) = " << std::scientific << finalLogProbability << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdmm->writeModel(std::cout);
		}
	}
}

void map_learning_by_em_using_conjugate_prior()
{
	boost::scoped_ptr<swl::ContinuousDensityMixtureModel> cdmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed.
*/

	// initialize a model.
	const int initialization_mode = 1;
	if (1 == initialization_mode)
	{
#if __TEST_MIXTURE_MODEL == 1
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1.cdmm");
#elif __TEST_MIXTURE_MODEL == 2
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2.cdmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		std::srand((unsigned int)std::time(NULL));

		// hyperparameters for the conjugate prior.
		// FIXME [check] >> hyperparameters for multivariate normal mixture distributions.
		std::vector<double> *pi_conj = new std::vector<double>(K, 1.0);
		std::vector<swl::ContinuousDensityMixtureModel::dvector_type> *mus_conj = new std::vector<swl::ContinuousDensityMixtureModel::dvector_type>(K, swl::ContinuousDensityMixtureModel::dvector_type(D, 0.0));
		swl::ContinuousDensityMixtureModel::dvector_type *betas_conj = new swl::ContinuousDensityMixtureModel::dvector_type(K, 1.0);  // beta > 0.
		std::vector<swl::ContinuousDensityMixtureModel::dmatrix_type> *covs_conj = new std::vector<swl::ContinuousDensityMixtureModel::dmatrix_type>(K, swl::ContinuousDensityMixtureModel::dmatrix_type(D, D, 1.0));
		swl::ContinuousDensityMixtureModel::dvector_type *nus_conj = new swl::ContinuousDensityMixtureModel::dvector_type(K, 1.0);  // nu > D - 1.
		for (std::size_t k = 0; k < K; ++k)
		{
			//(*betas_conj)[k] = (std::rand() / RAND_MAX + 1.0) * 10.0;
			//(*nus_conj)[k] = ???;

			swl::MultivariateNormalMixtureModel::dvector_type &mu_conj = (*mus_conj)[k];
			//swl::MultivariateNormalMixtureModel::dmatrix_type &cov_conj = (*covs_conj)[k];
			for (std::size_t d = 0; d < D; ++d)
			{
				mu_conj[d] = (std::rand() / RAND_MAX) * 10.0 - 5.0;
				for (std::size_t i = 0; i < D; ++i)
				{
					//cov_conj(d, i) = ???;
				}
			}
		}

		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D, pi_conj, mus_conj, betas_conj, covs_conj, nus_conj));

		const bool retval = cdmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi.
		cdmm->normalizeModelParameters();

		//cdmm->writeModel(std::cout);
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

		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		// hyperparameters for the conjugate prior.
		// FIXME [check] >> hyperparameters for multivariate normal mixture distributions.
		std::vector<double> *pi_conj = new std::vector<double>(K, 1.0);
		std::vector<swl::ContinuousDensityMixtureModel::dvector_type> *mus_conj = new std::vector<swl::ContinuousDensityMixtureModel::dvector_type>(K, swl::ContinuousDensityMixtureModel::dvector_type(D, 0.0));
		swl::ContinuousDensityMixtureModel::dvector_type *betas_conj = new swl::ContinuousDensityMixtureModel::dvector_type(K, 1.0);  // beta > 0.
		std::vector<swl::ContinuousDensityMixtureModel::dmatrix_type> *covs_conj = new std::vector<swl::ContinuousDensityMixtureModel::dmatrix_type>(K, swl::ContinuousDensityMixtureModel::dmatrix_type(D, D, 1.0));
		swl::ContinuousDensityMixtureModel::dvector_type *nus_conj = new swl::ContinuousDensityMixtureModel::dvector_type(K, 1.0);  // nu > D - 1.
		for (std::size_t k = 0; k < K; ++k)
		{
			//(*betas_conj)[k] = (std::rand() / RAND_MAX + 1.0) * 10.0;
			//(*nus_conj)[k] = ???;

			swl::MultivariateNormalMixtureModel::dvector_type &mu_conj = (*mus_conj)[k];
			//swl::MultivariateNormalMixtureModel::dmatrix_type &cov_conj = (*covs_conj)[k];
			for (std::size_t d = 0; d < D; ++d)
			{
				mu_conj[d] = (std::rand() / RAND_MAX) * 10.0 - 5.0;
				for (std::size_t i = 0; i < D; ++i)
				{
					//cov_conj(d, i) = ???;
				}
			}
		}

		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D, pi_conj, mus_conj, betas_conj, covs_conj, nus_conj));

		// the total number of parameters of observation density = K * (D + D * D).
		std::vector<double> lowerBounds, upperBounds;
		const std::size_t numParameters = K * (D + D * D);
		lowerBounds.reserve(numParameters);
		upperBounds.reserve(numParameters);
		// mean vector & convariance matrix.
		for (std::size_t i = 0; i < numParameters; ++i)
		{
			lowerBounds.push_back(-1000.0);
			upperBounds.push_back(1000.0);
		}
		cdmm->initializeModel(lowerBounds, upperBounds);
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	//
	{
		// read a observation sequence.
		swl::ContinuousDensityMixtureModel::dmatrix_type observations;
		std::size_t N = 0;  // length of observation sequence, N.
		{
#if __TEST_MIXTURE_MODEL == 1

#if 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_50.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_100.seq");
#elif 1
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_1500.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_MIXTURE_MODEL == 2

#if 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_50.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_100.seq");
#elif 1
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_1500.seq");
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

			std::size_t D = 0;
			const bool retval = swl::ContinuousDensityMixtureModel::readSequence(stream, N, D, observations);
			if (!retval || cdmm->getObservationDim() != D)
			{
				std::ostringstream stream;
				stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}
		}

		// EM algorithm.
		{
			const double terminationTolerance = 0.001;
			const std::size_t maxIteration = 1000;
			std::size_t numIteration = (std::size_t)-1;
			double initLogProbability = 0.0, finalLogProbability = 0.0;
			cdmm->trainByMAPUsingConjugatePrior(N, observations, terminationTolerance, maxIteration, numIteration, initLogProbability, finalLogProbability);

			// normalize pi.
			//cdmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "EM algorithm" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observations | initial model) = " << std::scientific << initLogProbability << std::endl;
			std::cout << "\tlog prob(observations | estimated model) = " << std::scientific << finalLogProbability << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdmm->writeModel(std::cout);
		}
	}
}

void map_learning_by_em_using_entropic_prior()
{
	boost::scoped_ptr<swl::ContinuousDensityMixtureModel> cdmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed.
*/

	// initialize a model.
	const int initialization_mode = 2;
	if (1 == initialization_mode)
	{
#if __TEST_MIXTURE_MODEL == 1
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1.cdmm");
#elif __TEST_MIXTURE_MODEL == 2
		const std::size_t K = 3;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		//
		std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2.cdmm");
#endif
		if (!stream)
		{
			std::ostringstream stream;
			stream << "file not found at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// hyperparameters for the entropic prior.
		//	don't need.

		//cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D, mus_conj, betas_conj, sigmas_conj, nus_conj));
		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D));

		const bool retval = cdmm->readModel(stream);
		if (!retval)
		{
			std::ostringstream stream;
			stream << "model reading error at " << __LINE__ << " in " << __FILE__;
			throw std::runtime_error(stream.str().c_str());
			return;
		}

		// normalize pi.
		cdmm->normalizeModelParameters();

		//cdmm->writeModel(std::cout);
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

		const std::size_t K = 30;  // the number of mixture components.
		const std::size_t D = 2;  // the dimension of observation symbols.

		// hyperparameters for the entropic prior.
		//	don't need.

		//cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D, mus_conj, betas_conj, sigmas_conj, nus_conj));
		cdmm.reset(new swl::MultivariateNormalMixtureModel(K, D));

		// the total number of parameters of observation density = K * (D + D * D).
		std::vector<double> lowerBounds, upperBounds;
		const std::size_t numParameters = K * (D + D * D);
		lowerBounds.reserve(numParameters);
		upperBounds.reserve(numParameters);
#if 0
		// mean vector & convariance matrix.
		for (std::size_t i = 0; i < numParameters; ++i)
		{
			lowerBounds.push_back(-1000.0);
			upperBounds.push_back(1000.0);
		}
#else
		// for annulus problem.

		// mean vector.
		for (std::size_t i = 0; i < K * D; ++i)
		{
			lowerBounds.push_back(-100.0);
			upperBounds.push_back(100.0);
		}
		// convariance matrix.
		for (std::size_t i = K * D; i < numParameters; ++i)
		{
			lowerBounds.push_back(-10.0);
			upperBounds.push_back(10.0);
		}
#endif
		cdmm->initializeModel(lowerBounds, upperBounds);
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	//
	{
		// read a observation sequence.
		swl::ContinuousDensityMixtureModel::dmatrix_type observations;
		std::size_t N = 0;  // length of observation sequence, N.
		{
#if __TEST_MIXTURE_MODEL == 1

#if 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_50.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_100.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test1_1500.seq");
#elif 1
			// annulus problem.
			// [ref] "Neural Networks for Pattern Recognition", C. M. Bishop, Oxford University Press, 1995, pp. 68.
			std::ifstream stream("./data/mixture_model/annulus_data_20_to_50.seq");
#else
			std::istream stream = std::cin;
#endif

#elif __TEST_MIXTURE_MODEL == 2

#if 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_50.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_100.seq");
#elif 0
			std::ifstream stream("./data/mixture_model/bi_normal_mixture_test2_1500.seq");
#elif 1
			// annulus problem.
			// [ref] "Neural Networks for Pattern Recognition", C. M. Bishop, Oxford University Press, 1995, pp. 68.
			std::ifstream stream("./data/mixture_model/annulus_data_20_to_50.seq");
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

			std::size_t D = 0;
			const bool retval = swl::ContinuousDensityMixtureModel::readSequence(stream, N, D, observations);
			if (!retval || cdmm->getObservationDim() != D)
			{
				std::ostringstream stream;
				stream << "sample sequence reading error at " << __LINE__ << " in " << __FILE__;
				throw std::runtime_error(stream.str().c_str());
				return;
			}
		}

		// EM algorithm.
		{
			// z = 1 (default) is min. entropy.
			// z = 0 is max. likelihood.
			// z = -1 is max. entropy.
			// z = -inf corresponds to very high temperature (good for initialization).
			const double z = 1.0;
			const bool doesTrimParameter = true;

			const double terminationTolerance = 0.001;
			const std::size_t maxIteration = 1000;
			std::size_t numIteration = (std::size_t)-1;
			double initLogProbability = 0.0, finalLogProbability = 0.0;
			cdmm->trainByMAPUsingEntropicPrior(N, observations, z, doesTrimParameter, terminationTolerance, maxIteration, numIteration, initLogProbability, finalLogProbability);

			// normalize pi.
			//cdmm->normalizeModelParameters();

			//
			std::cout << "------------------------------------" << std::endl;
			std::cout << "EM algorithm" << std::endl;
			std::cout << "\tnumber of iterations = " << numIteration << std::endl;
			std::cout << "\tlog prob(observations | initial model) = " << std::scientific << initLogProbability << std::endl;
			std::cout << "\tlog prob(observations | estimated model) = " << std::scientific << finalLogProbability << std::endl;
			std::cout << "\testimated model:" << std::endl;
			cdmm->writeModel(std::cout);
		}
	}
}

}  // namespace local
}  // unnamed namespace

void multivariate_normal_mixture_model()
{
	std::cout << "multivariate normal mixture model -------------------------------------" << std::endl;

	//local::model_reading_and_writing();
	//const bool outputToFile = false;
	//local::observation_sequence_generation(outputToFile);
	//local::observation_sequence_reading_and_writing();

	std::cout << "\ntrain by ML ---------------------------------------------------------" << std::endl;
	local::ml_learning_by_em();
	std::cout << "\ntrain by MAP using conjugate prior ----------------------------------" << std::endl;
	local::map_learning_by_em_using_conjugate_prior();
	std::cout << "\ntrain by MAP using entropic prior -----------------------------------" << std::endl;
	local::map_learning_by_em_using_entropic_prior();
}

#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultinomialObservations.h"
#include "RndUtilLocalApi.h"
#include <numeric>
#include <cstring>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMultinomialObservations::HmmWithMultinomialObservations(const size_t K, const size_t D)
: base_type(K, D), B_(K, D, 0.0),  // 0-based index.
  B_conj_()
{
}

HmmWithMultinomialObservations::HmmWithMultinomialObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &B)
: base_type(K, D, pi, A), B_(B),
  B_conj_()
{
}

HmmWithMultinomialObservations::HmmWithMultinomialObservations(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *B_conj)
: base_type(K, D, pi_conj, A_conj), B_(K, D, 0.0),
  B_conj_(B_conj)
{
}

HmmWithMultinomialObservations::~HmmWithMultinomialObservations()
{
}

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n;
	const double denominatorB = denominatorA + gamma(N-1, state);
	double numeratorB;
	for (size_t d = 0; d < D_; ++d)
	{
		numeratorB = 0.0;
		for (n = 0; n < N; ++n)
		{
			if (observations[n] == (unsigned int)d)
				numeratorB += gamma(n, state);
		}

		B_(state, d) = 0.001 + 0.999 * numeratorB / denominatorB;
	}
}

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n, r;
	double denominatorB = denominatorA;
	for (r = 0; r < R; ++r)
		denominatorB += gammas[r](Ns[r]-1, state);

	double numeratorB;
	for (size_t d = 0; d < D_; ++d)
	{
		numeratorB = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
			{
				if (observationSequences[r][n] == (unsigned int)d)
					numeratorB += gammas[r](n, state);
			}

		B_(state, d) = 0.001 + 0.999 * numeratorB / denominatorB;
	}
}

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n;
	const double denominatorB = denominatorA + gamma(N-1, state);
	double numeratorB;
	for (size_t d = 0; d < D_; ++d)
	{
		numeratorB = (*B_conj_)(state, d) - 1.0;
		for (n = 0; n < N; ++n)
		{
			if (observations[n] == (unsigned int)d)
				numeratorB += gamma(n, state);
		}

		B_(state, d) = 0.001 + 0.999 * numeratorB / (denominatorB + D_ * ((*B_conj_)(state, d) - 1.0));
	}
}

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n, r;
	double denominatorB = denominatorA;
	for (r = 0; r < R; ++r)
		denominatorB += gammas[r](Ns[r]-1, state);

	double numeratorB;
	for (size_t d = 0; d < D_; ++d)
	{
		numeratorB = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
			{
				if (observationSequences[r][n] == (unsigned int)d)
					numeratorB += gammas[r](n, state);
			}

		B_(state, d) = 0.001 + 0.999 * numeratorB / (denominatorB + D_ * ((*B_conj_)(state, d) - 1.0));
	}
}

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const unsigned int state, const double z, const bool doesTrimParameter, const dmatrix_type &expNumEmit, std::vector<double> &thetaEmit, std::vector<bool> &isObservationsTrimmed)
{
	const double eps = 1e-50;

#if 0
	// M-step.
	// reestimate observation(emission) distribution in each state.

	std::vector<double> omega(D_, 0.0), theta(D_, 0.0);
	size_t d, n;
	for (d = 0; d < D_; ++d)
	{
		omega[d] = 0.0;
		for (n = 0; n < N; ++n)
		{
			if (observations[n] == (unsigned int)d)
				omega[d] += gamma(n, state);
		}
	}

	double entropicMAPLogLikelihood = 0.0;
	const bool retval = computeMAPEstimateOfMultinomialUsingEntropicPrior(omega, z, theta, entropicMAPLogLikelihood, terminationTolerance, maxIteration, false);
	assert(retval);

	// trim parameters.
	if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
	{
		throw std::runtime_error("not yet implemented");
	}

	for (d = 0; d < D_; ++d)
		B_(state, d) = theta[d];
#else
	// trim transition probabilities.
	//	only trim if we are in the min. entropy setting (z = 1).
	//	if z << 0, we would trim everything.
	if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
	{
		if (!isObservationsTrimmed[state])  // not yet trimmed.
		{
			double grad;
			for (size_t d = 0; d < D_; ++d)
			{
				grad = expNumEmit(state, d) / (std::fabs(thetaEmit[d]) < eps ? 1.0 : thetaEmit[d]);
				if (thetaEmit[d] <= std::exp(-grad / z))
				{
					thetaEmit[d] = 0.0;
					isObservationsTrimmed[state] = true;
				}
			}

			const double sumTheta = std::accumulate(thetaEmit.begin(), thetaEmit.end(), 0.0);
			assert(std::fabs(sumTheta) >= eps);
			for (size_t d = 0; d < D_; ++d)
				B_(state, d) = thetaEmit[d] / sumTheta; 
		}
	}
	else
	{
		for (size_t d = 0; d < D_; ++d)
			B_(state, d) = thetaEmit[d];
	}
#endif
}

unsigned int HmmWithMultinomialObservations::doGenerateObservationsSymbol(const unsigned int state) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	const double prob = (double)std::rand() / RAND_MAX;

	double accum = 0.0;
	unsigned int observation = (unsigned int)D_;
	for (size_t d = 0; d < D_; ++d)
	{
		accum += B_(state, d);
		//accum += doEvaluateEmissionProbability(state, d);
		if (prob < accum)
		{
			observation = (unsigned int)d;
			break;
		}
	}

	return observation;

	// POSTCONDITIONS [] >>
	//	-. if observation = D_, an error occurs.
}

/*virtual*/ void HmmWithMultinomialObservations::doInitializeRandomSampleGeneration(const unsigned int seed /*= (unsigned int)-1*/) const
{
	if ((unsigned int)-1 != seed)
		std::srand(seed);
}

bool HmmWithMultinomialObservations::doReadObservationDensity(std::istream &stream)
{
	size_t i, k;
	std::string dummy;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "B:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "B:") != 0)
#endif
		return false;

	// K x D
	B_.resize(K_, D_);
	for (k = 0; k < K_; ++k)
		for (i = 0; i < D_; ++i)
			stream >> B_(k, i);

	return true;
}

bool HmmWithMultinomialObservations::doWriteObservationDensity(std::ostream &stream) const
{
	size_t i, k;

	// K x D
	stream << "B:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		for (i = 0; i < D_; ++i)
			stream << B_(k, i) << ' ';
		stream << std::endl;
	}

	return true;
}

void HmmWithMultinomialObservations::doInitializeObservationDensity(const std::vector<double> & /*lowerBoundsOfObservationDensity*/, const std::vector<double> & /*upperBoundsOfObservationDensity*/)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	size_t i;
	double sum;
	for (size_t k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (i = 0; i < D_; ++i)
		{
			B_(k, i) = (double)std::rand() / RAND_MAX;
			sum += B_(k, i);
		}
		for (i = 0; i < D_; ++i)
			B_(k, i) /= sum;
	}
}

void HmmWithMultinomialObservations::doNormalizeObservationDensityParameters()
{
	size_t i;
	double sum;
	for (size_t k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (i = 0; i < D_; ++i)
			sum += B_(k, i);
		for (i = 0; i < D_; ++i)
			B_(k, i) /= sum;
	}
}

}  // namespace swl

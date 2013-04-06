#if !defined(__SWL_RND_UTIL__SAMPLING_IMPORTANCE_RESAMPLING__H_)
#define __SWL_RND_UTIL__SAMPLING_IMPORTANCE_RESAMPLING__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <vector>


namespace swl {

//--------------------------------------------------------------------------
// sampling importance resampling (SIR): sequential importance sampling (SIS) + resampling
// particle filter

class SWL_RND_UTIL_API SamplingImportanceResampling
{
public:
	//typedef SamplingImportanceResampling base_type;
	typedef boost::numeric::ublas::vector<double> vector_type;

public:
	// transition distribution
	struct TransitionDistribution
	{
		// p(x(k) | x(k-1))
		virtual double evaluate(const size_t step, const vector_type &currX, const vector_type &prevX) const = 0;
	};
	// observation distribution
	struct ObservationDistribution
	{
		// p(y(k) | x(k))
		virtual double evaluate(const size_t step, const vector_type &x, const vector_type &y) const = 0;
	};

	// proposal distribution
	struct ProposalDistribution
	{
		// p(x(k) | x(k-1))
		virtual double evaluate(const size_t step, const vector_type &currX, const vector_type &prevX, const vector_type &y) const = 0;
		// p(x(k) | x(0:k-1), y(0:k))
		virtual double evaluate(const size_t step, const vector_type &currX, const std::vector<vector_type> &prevXs, const std::vector<vector_type> &ys) const = 0;

		// x ~ p(x(k) | x(k-1))
		virtual void sample(const size_t step, const vector_type &x, const vector_type &y, vector_type &sample) const = 0;
		// x ~ p(x(k) | x(0:k-1), y(0:k))
		virtual void sample(const size_t step, const std::vector<vector_type> &xs, const std::vector<vector_type> &ys, vector_type &sample) const = 0;
	};

public:
	SamplingImportanceResampling(const double effectiveSampleSize, TransitionDistribution &transitionDistribution, ObservationDistribution &observationDistribution, ProposalDistribution &proposalDistribution);
	~SamplingImportanceResampling();

private:
	SamplingImportanceResampling(const SamplingImportanceResampling &rhs);
	SamplingImportanceResampling & operator=(const SamplingImportanceResampling &rhs);

public:
	void sample(const size_t step, const size_t particleNum, const std::vector<vector_type> &xs, const vector_type &y, std::vector<vector_type> &newXs, std::vector<double> &weights, vector_type *estimatedX = NULL) const;

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::uniform_real<> > generator_type;

private:
	const double effectiveSampleSize_;

	TransitionDistribution &transitionDistribution_;
	ObservationDistribution &observationDistribution_;
	ProposalDistribution &proposalDistribution_;

	base_generator_type baseGenerator_;
	mutable generator_type generator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__SAMPLING_IMPORTANCE_RESAMPLING__H_

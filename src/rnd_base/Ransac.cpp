#include "swl/Config.h"
#include "swl/rnd_base/Ransac.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(max)
#undef max
#endif

namespace swl {

size_t Ransac::runRANSAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double threshold)
{
	if (totalSampleCount_ < minimalSampleSetSize_)
		return -1;

	if (isProsacSampling) sortSamples();

	size_t maxIteration = maxIterationCount;

	size_t inlierCount = 0;
	inlierFlags_.resize(totalSampleCount_, false);
	std::vector<bool> currInlierFlags(totalSampleCount_, false);

	std::vector<size_t> indices(minimalSampleSetSize_, -1);

	// TODO [check] >>
	size_t prosacSampleCount = 10;
	iteration_ = 0;
	while (maxIteration > iteration_ && inlierCount < minInlierCount)
	{
		// draw a sample
		if (isProsacSampling)
		{
			drawProsacSample(prosacSampleCount, minimalSampleSetSize_, indices);

			// this incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, indices);

		// estimate a model
		if (estimateModel(indices) && verifyModel())
		{
			// evaluate a model
			const size_t currInlierCount = lookForInliers(currInlierFlags, threshold);

			if (currInlierCount > inlierCount)
			{
				const double inlierRatio = double(currInlierCount) / totalSampleCount_;
				const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / std::log(1.0 - std::pow(inlierRatio, (double)minimalSampleSetSize_)));
				if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;

				inlierCount = currInlierCount;
				//for (size_t i = 0; i < totalSampleCount_; ++i) inlierFlags_[i] = currInlierFlags[i];
				inlierFlags_.swap(currInlierFlags);
			}
		}

		++iteration_;
	}

	// re-estimate with all inliers and loop until the number of inliers is not increased anymore
	size_t oldInlierCount = inlierCount;
	do
	{
		if (!estimateModelFromInliers()) return -1;

		oldInlierCount = inlierCount;
		inlierCount = lookForInliers(inlierFlags_, threshold);
	} while (inlierCount > oldInlierCount);

	inlierCount = lookForInliers(inlierFlags_, threshold);

	return inlierCount;
}

size_t Ransac::runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double outlierUniformProbability, const size_t maxEMIterationCount)
{
	if (totalSampleCount_ < minimalSampleSetSize_)
		return -1;

	if (isProsacSampling) sortSamples();

	size_t maxIteration = maxIterationCount;

	size_t inlierCount = 0;
	inlierFlags_.resize(totalSampleCount_, false);
	std::vector<double> inlierProbs(totalSampleCount_, 0.0);
	double minNegativeLogLikelihood = std::numeric_limits<double>::max();

	std::vector<size_t> indices(minimalSampleSetSize_, -1);

	// TODO [check] >>
	size_t prosacSampleCount = 10;
	iteration_ = 0;
	while (maxIteration > iteration_ && inlierCount < minInlierCount)
	{
		// draw a sample
		if (isProsacSampling)
		{
			drawProsacSample(prosacSampleCount, minimalSampleSetSize_, indices);

			// this incrementing strategy is naive and simple but works just fine most of the time.
			if (prosacSampleCount < totalSampleCount_)
				++prosacSampleCount;
		}
		else drawRandomSample(totalSampleCount_, minimalSampleSetSize_, indices);

		// estimate a model
		if (estimateModel(indices) && verifyModel())
		{
			// compute inliers' probabilities
			computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

			// EM algorithm
			const double tol = 1.0e-5;

			double gamma = 0.5, prevGamma;
			for (size_t i = 0; i < maxEMIterationCount; ++i)
			{
				const double outlierProb = (1.0 - gamma) * outlierUniformProbability;
				double sumInlierProb = 0.0;
				for (size_t k = 0; k < totalSampleCount_; ++k)
				{
					const double inlierProb = gamma * inlierProbs[k];
					sumInlierProb += inlierProb / (inlierProb + outlierProb);
				}

				prevGamma = gamma;
				gamma = sumInlierProb / totalSampleCount_;

				if (std::fabs(gamma - prevGamma) < tol) break;
			}

			// evaluate a model
			const double outlierProb = (1.0 - gamma) * outlierUniformProbability;
			double negativeLogLikelihood = 0.0;
			for (size_t k = 0; k < totalSampleCount_; ++k)
				negativeLogLikelihood -= std::log(gamma * inlierProbs[k] + outlierProb);  // negative log likelihood

			//
			if (negativeLogLikelihood < minNegativeLogLikelihood)
			{
				const size_t newMaxIteration = (size_t)std::floor(std::log(alarmRatio) / std::log(1.0 - std::pow(gamma, (double)minimalSampleSetSize_)));
				if (newMaxIteration < maxIteration) maxIteration = newMaxIteration;

				inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);

				minNegativeLogLikelihood = negativeLogLikelihood;
			}
		}

		++iteration_;
	}

	// re-estimate with all inliers and loop until the number of inliers is not increased anymore
	size_t oldInlierCount = 0;
	do
	{
		if (!estimateModelFromInliers()) return -1;

		// compute inliers' probabilities
		computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

		oldInlierCount = inlierCount;
		inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);
	} while (inlierCount > oldInlierCount);

	// compute inliers' probabilities
	computeInlierProbabilities(inlierProbs, inlierSquaredStandardDeviation);

	inlierCount = lookForInliers(inlierFlags_, inlierProbs, outlierUniformProbability);

	return inlierCount;
}

void Ransac::drawRandomSample(const size_t maxCount, const size_t count, std::vector<size_t> &indices) const
{
	for (size_t i = 0; i < count; )
	{
		const size_t idx = std::rand() % maxCount;
		std::vector<size_t>::iterator it = std::find(indices.begin(), indices.end(), idx);

		if (indices.end() == it)
			indices[i++] = idx;
	}
}

void Ransac::drawProsacSample(const size_t maxCount, const size_t count, std::vector<size_t> &indices) const
{
	for (size_t i = 0; i < count; )
	{
		const size_t idx = std::rand() % maxCount;
		std::vector<size_t>::iterator it = std::find(indices.begin(), indices.end(), idx);

		if (indices.end() == it)
			indices[i++] = idx;
	}

	for (std::vector<size_t>::iterator it = indices.begin(); it != indices.end(); ++it)
		*it = sortedIndices_[*it];
}

void Ransac::sortSamples()
{
	sortedIndices_.reserve(totalSampleCount_);
	for (size_t i = 0; i < totalSampleCount_; ++i)
		sortedIndices_.push_back(i);

	if (scores_ && !scores_->empty())
		std::sort(sortedIndices_.begin(), sortedIndices_.end(), CompareByScore(*scores_));
}

}  // namespace swl

#if !defined(__SWL_RND_UTIL__RANSAC__H_)
#define __SWL_RND_UTIL__RANSAC__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <vector>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_RND_UTIL_API Ransac
{
public:
	//typedef Ransac base_type;

protected:
	Ransac(const size_t sampleCount, const size_t minimalSampleSetSize)
	: totalSampleCount_(sampleCount), minimalSampleSetSize_(minimalSampleSetSize), scores_(NULL), sortedIndices_(), inlierFlags_(), iteration_(0)
	{
	}
	Ransac(const size_t sampleCount, const size_t minimalSampleSetSize, const std::vector<double> &scores)
	: totalSampleCount_(sampleCount), minimalSampleSetSize_(minimalSampleSetSize), scores_(&scores), sortedIndices_(), inlierFlags_(), iteration_(0)
	{
	}
public:
	virtual ~Ransac();

public:
	virtual size_t runRANSAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double threshold);
	virtual size_t runMLESAC(const size_t maxIterationCount, const size_t minInlierCount, const double alarmRatio, const bool isProsacSampling, const double inlierSquaredStandardDeviation, const double outlierUniformProbability, const size_t maxEMIterationCount);

	const std::vector<bool> & getInliers() const  {  return inlierFlags_;  }
	size_t getIterationCount() const  {  return iteration_;  }

protected:
	void drawRandomSample(const size_t maxCount, const size_t count, std::vector<size_t> &indices) const;
	void drawProsacSample(const size_t maxCount, const size_t count, std::vector<size_t> &indices) const;
	void sortSamples();

private:
	virtual bool estimateModel(const std::vector<size_t> &indices) = 0;
	virtual bool verifyModel() const = 0;
	virtual bool estimateModelFromInliers() = 0;

	// for RANSAC
	virtual size_t lookForInliers(std::vector<bool> &inliers, const double threshold) const = 0;
	// for MLESAC
	virtual void computeInlierProbabilities(std::vector<double> &inlierProbs, const double inlierSquaredStandardDeviation) const = 0;
	virtual size_t lookForInliers(std::vector<bool> &inliers, const std::vector<double> &inlierProbs, const double outlierUniformProbability) const = 0;

private:
	struct CompareByScore
	{
	public:
		CompareByScore(const std::vector<double> &scores) : scores_(scores) {}

		bool operator()(const int lhs, const int rhs) const
		{  return scores_[lhs] > scores_[rhs];  }

	private:
		const std::vector<double> &scores_;
	};

protected:
	const size_t totalSampleCount_;
	const size_t minimalSampleSetSize_;

	const std::vector<double> *scores_;
	std::vector<size_t> sortedIndices_;

	std::vector<bool> inlierFlags_;
	size_t iteration_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__RANSAC__H_

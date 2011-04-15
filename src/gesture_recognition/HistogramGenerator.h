#if !defined(__SWL_GESTURE_RECOGNITION__HISTOGRAM_GENERATOR__H_)
#define __SWL_GESTURE_RECOGNITION__HISTOGRAM_GENERATOR__H_ 1


#include <vector>

namespace cv {

class Mat;
typedef Mat MatND;

}


namespace swl {

//-----------------------------------------------------------------------------
//

struct HistogramUtil
{
	static void normalizeHistogram(cv::MatND &hist, const double factor);
	static void drawHistogram1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg);
	static void drawHistogram2D(const cv::MatND &hist, const int horzBinCount, const int vertBinCount, const double maxVal, const int horzBinSize, const int vertBinSize, cv::Mat &histImg);
};


//-----------------------------------------------------------------------------
//

class HistogramGeneratorBase
{
protected:
	HistogramGeneratorBase()  {}
	virtual ~HistogramGeneratorBase();

public:
	virtual void createHistograms(const size_t binNum, const double histogramNormalizationFactor) = 0;

	const std::vector<cv::MatND> & getHistograms() const  {  return histograms_;  }
	const cv::MatND & getHistogram(const size_t idx) const  {  return histograms_[idx];  }

protected:
	std::vector<cv::MatND> histograms_;
};


//-----------------------------------------------------------------------------
//

class ReferencePhaseHistogramGeneratorBase: public HistogramGeneratorBase
{
public:
	typedef HistogramGeneratorBase base_type;

public:
	ReferencePhaseHistogramGeneratorBase(const double sigma)
	: base_type(), sigma_(sigma)
	{}

private:
	ReferencePhaseHistogramGeneratorBase(const ReferencePhaseHistogramGeneratorBase &rhs);
	ReferencePhaseHistogramGeneratorBase & operator=(const ReferencePhaseHistogramGeneratorBase &rhs);

protected:
	void createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const;
	void createUniformHistogram(cv::MatND &hist) const;

protected:
	const double sigma_;
};


//-----------------------------------------------------------------------------
//

class ReferenceFullPhaseHistogramGenerator: public ReferencePhaseHistogramGeneratorBase
{
public:
	// TODO [adjust] >> design parameter
	static const size_t REF_UNIMODAL_HISTOGRAM_NUM = 36;
	static const size_t REF_UNIFORM_HISTOGRAM_NUM = 1;
	static const size_t REF_HISTOGRAM_NUM = REF_UNIMODAL_HISTOGRAM_NUM + REF_UNIFORM_HISTOGRAM_NUM;

public:
	typedef ReferencePhaseHistogramGeneratorBase base_type;

public:
	ReferenceFullPhaseHistogramGenerator(const double sigma)
	: base_type(sigma)
	{}

private:
	ReferenceFullPhaseHistogramGenerator(const ReferenceFullPhaseHistogramGenerator &rhs);
	ReferenceFullPhaseHistogramGenerator & operator=(const ReferenceFullPhaseHistogramGenerator &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);
};


//-----------------------------------------------------------------------------
//

class ReferenceHistogramGeneratorForShortTimeGesture: public ReferencePhaseHistogramGeneratorBase
{
public:
	// TODO [adjust] >> design parameter
	static const size_t REF_BIMODAL_HISTOGRAM_NUM_FOR_TWO_HAND_GESTURE = 2;
	static const size_t REF_UNIFORM_HISTOGRAM_NUM = 1;
	static const size_t REF_HISTOGRAM_NUM = REF_BIMODAL_HISTOGRAM_NUM_FOR_TWO_HAND_GESTURE + REF_UNIFORM_HISTOGRAM_NUM;

public:
	typedef ReferencePhaseHistogramGeneratorBase base_type;

public:
	ReferenceHistogramGeneratorForShortTimeGesture(const double sigma)
	: base_type(sigma)
	{}

private:
	ReferenceHistogramGeneratorForShortTimeGesture(const ReferenceHistogramGeneratorForShortTimeGesture &rhs);
	ReferenceHistogramGeneratorForShortTimeGesture & operator=(const ReferenceHistogramGeneratorForShortTimeGesture &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);
};


//-----------------------------------------------------------------------------
//

class ReferenceHistogramGeneratorForLongTimeGesture: public ReferencePhaseHistogramGeneratorBase
{
public:
	// TODO [adjust] >> design parameter
	static const size_t REF_BIMODAL_HISTOGRAM_NUM = 2;
	static const size_t REF_UNIFORM_HISTOGRAM_NUM = 0; //1;
	static const size_t REF_HISTOGRAM_NUM = REF_BIMODAL_HISTOGRAM_NUM + REF_UNIFORM_HISTOGRAM_NUM;

public:
	typedef ReferencePhaseHistogramGeneratorBase base_type;

public:
	ReferenceHistogramGeneratorForLongTimeGesture(const double sigma)
	: base_type(sigma)
	{}

private:
	ReferenceHistogramGeneratorForLongTimeGesture(const ReferenceHistogramGeneratorForLongTimeGesture &rhs);
	ReferenceHistogramGeneratorForLongTimeGesture & operator=(const ReferenceHistogramGeneratorForLongTimeGesture &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);
};


//-----------------------------------------------------------------------------
//

class ReferenceHistogramGeneratorForThirdClassGesture: public ReferencePhaseHistogramGeneratorBase
{
public:
	// TODO [adjust] >> design parameter
	static const size_t REF_UNIMODAL_HISTOGRAM_NUM = 0;
	static const size_t REF_TRIMODAL_HISTOGRAM_NUM = 2;
	static const size_t REF_UNIFORM_HISTOGRAM_NUM = 0; //1;
	static const size_t REF_HISTOGRAM_NUM = REF_UNIMODAL_HISTOGRAM_NUM + REF_TRIMODAL_HISTOGRAM_NUM + REF_UNIFORM_HISTOGRAM_NUM;

public:
	typedef ReferencePhaseHistogramGeneratorBase base_type;

public:
	ReferenceHistogramGeneratorForThirdClassGesture(const double sigma)
	: base_type(sigma)
	{}

private:
	ReferenceHistogramGeneratorForThirdClassGesture(const ReferenceHistogramGeneratorForThirdClassGesture &rhs);
	ReferenceHistogramGeneratorForThirdClassGesture & operator=(const ReferenceHistogramGeneratorForThirdClassGesture &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);
};


//-----------------------------------------------------------------------------
//

class GestureIdPatternHistogramGeneratorBase: public HistogramGeneratorBase
{
public:
	typedef HistogramGeneratorBase base_type;

public:
	GestureIdPatternHistogramGeneratorBase(const double sigma)
	: base_type(), sigma_(sigma)
	{}

private:
	GestureIdPatternHistogramGeneratorBase(const GestureIdPatternHistogramGeneratorBase &rhs);
	GestureIdPatternHistogramGeneratorBase & operator=(const GestureIdPatternHistogramGeneratorBase &rhs);

protected:
	void createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const;
	void createUniformHistogram(cv::MatND &hist) const;

protected:
	const double sigma_;
};


//-----------------------------------------------------------------------------
//

class GestureIdPatternHistogramGeneratorForShortTimeGesture: public GestureIdPatternHistogramGeneratorBase
{
public:
	typedef GestureIdPatternHistogramGeneratorBase base_type;

public:
	GestureIdPatternHistogramGeneratorForShortTimeGesture(const double sigma)
	: base_type(sigma)
	{}

private:
	GestureIdPatternHistogramGeneratorForShortTimeGesture(const GestureIdPatternHistogramGeneratorForShortTimeGesture &rhs);
	GestureIdPatternHistogramGeneratorForShortTimeGesture & operator=(const GestureIdPatternHistogramGeneratorForShortTimeGesture &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);
};

}  // namespace swl


#endif  // __SWL_GESTURE_RECOGNITION__HISTOGRAM_GENERATOR__H_

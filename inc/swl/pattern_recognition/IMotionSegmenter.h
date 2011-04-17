#if !defined(__SWL_PATTERN_RECOGNITION__MOTION_SEGMENTER_INTERFACE__H_)
#define __SWL_PATTERN_RECOGNITION__MOTION_SEGMENTER_INTERFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------
//

struct IMotionSegmenter
{
	virtual ~IMotionSegmenter()  {}

	virtual bool segment() = 0;
};


}  // namespace swl


#endif  // __SWL_PATTERN_RECOGNITION__MOTION_SEGMENTER_INTERFACE__H_

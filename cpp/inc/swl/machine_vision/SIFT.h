#if !defined(__SWL_MACHINE_VISION__SIFT__H_)
#define __SWL_MACHINE_VISION__SIFT__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#include <map>
#include <set>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_MACHINE_VISION_API SIFT
{
public:
	//typedef SIFT base_type;

public:
	SIFT();
	~SIFT();

public:
	bool extractFeature();
	bool matchFeature();

protected:
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__SIFT__H_

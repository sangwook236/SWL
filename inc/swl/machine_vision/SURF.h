#if !defined(__SWL_MACHINE_VISION__SURF__H_)
#define __SWL_MACHINE_VISION__SURF__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#include <map>
#include <set>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_MACHINE_VISION_API SURF
{
public:
	//typedef SURF base_type;

public:
	SURF();
	~SURF();

public:
	bool extractFeature();
	bool matchFeature();

protected:
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__SURF__H_

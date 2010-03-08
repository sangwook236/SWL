#if !defined(__SWL_KINEMATICS__LINK__H_)
#define __SWL_KINEMATICS__LINK__H_ 1


#include "swl/kinematics/ExportKinematics.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class Link

class SWL_KINEMATICS_API Link
{
public:
	//typedef Link base_type;

public:
	Link();
	Link(const Link &rhs);
	virtual ~Link();

	Link & operator=(const Link &rhs);
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__LINK__H_

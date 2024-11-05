#if !defined(__SWL_KINEMATICS__JOINT__H_)
#define __SWL_KINEMATICS__JOINT__H_ 1


#include "swl/kinematics/JointParam.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class Joint

class SWL_KINEMATICS_API Joint
{
public:
	//typedef Joint base_type;

public:
	Joint();
	Joint(const Joint &rhs);
	virtual ~Joint();

	Joint & operator=(const Joint &rhs);

private:
	///
	JointParam jointParam_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__JOINT__H_

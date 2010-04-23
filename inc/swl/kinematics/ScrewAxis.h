#if !defined(__SWL_KINEMATICS__SCREW_AXIS__H_)
#define __SWL_KINEMATICS__SCREW_AXIS__H_ 1


#include "swl/kinematics/JointParam.h"
#include "swl/math/Vector.h"


namespace swl {

//--------------------------------------------------------------------------------
//

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_KINEMATICS_EXPORT_TEMPLATE template class SWL_KINEMATICS_API Vector3<double>;
#endif


//--------------------------------------------------------------------------------
// class ScrewAxis

class SWL_KINEMATICS_API ScrewAxis
{
public:
	//typedef ScrewAxis base_type;

public:
	ScrewAxis(const Vector3<double> &dir, const Vector3<double> &pos, const double pitch = 0.0)
	: //base_type(),
	  dir_(dir), pos_(pos), pitch_(pitch)
	{}
	ScrewAxis(const ScrewAxis &rhs)
	: //base_type(rhs),
	  dir_(rhs.dir_), pos_(rhs.pos_), pitch_(rhs.pitch_)
	{}
	virtual ~ScrewAxis()  {}

	ScrewAxis & operator=(const ScrewAxis &rhs);

public:
	/// accessor & mutator
	void setDir(const Vector3<double> &dir)  {  dir_ = dir;  }
	const Vector3<double> & getDir() const  {  return dir_;  }

	void setPos(const Vector3<double> &pos)  {  pos_ = pos;  }
	const Vector3<double> & getPos() const  {  return pos_;  }

	void setPitch(const double pitch)  {  pitch_ = pitch;  }
	double getPitch() const  {  return pitch_;  }

protected:
	/// direction of screw axis
	Vector3<double> dir_;
	/// position of screw axis
	Vector3<double> pos_;
	/// pitch of screw axis
	double pitch_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__SCREW_AXIS__H_

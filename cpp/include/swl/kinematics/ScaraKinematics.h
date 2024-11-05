#if !defined(__SWL_KINEMATICS__SCARA_KINEMATICS__H_)
#define __SWL_KINEMATICS__SCARA_KINEMATICS__H_ 1


#include "swl/kinematics/RobotKinematics.h"


namespace swl {

//--------------------------------------------------------------------------------
// class Scara2Kinematics: for 2-R SCARA Robot (Planar Robot)

class SWL_KINEMATICS_API Scara2Kinematics: public RobotKinematics
{
public:
	typedef RobotKinematics base_type;

private:
	Scara2Kinematics(const Scara2Kinematics &rhs);
	Scara2Kinematics & operator=(const Scara2Kinematics &rhs);

public:
	Scara2Kinematics();
	virtual ~Scara2Kinematics();

public:
	///
	/*virtual*/ bool solveForward(const coords_type &jointCoords, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL);
	/*virtual*/ bool solveInverse(const coords_type &cartesianCoords, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL);
};


//--------------------------------------------------------------------------------
// class Scara3Kinematics: for 3-R SCARA Robot (Planar Robot)

class SWL_KINEMATICS_API Scara3Kinematics: public Scara2Kinematics
{
public:
	typedef Scara2Kinematics base_type;

public:
	Scara3Kinematics();
	virtual ~Scara3Kinematics();

private:
	Scara3Kinematics(const Scara3Kinematics &rhs);
	Scara3Kinematics & operator=(const Scara3Kinematics &rhs);

public:
	///
	/*virtual*/ bool solveForward(const coords_type &jointCoords, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL);
	/*virtual*/ bool solveInverse(const coords_type &cartesianCoords, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL);
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__SCARA_KINEMATICS__H_

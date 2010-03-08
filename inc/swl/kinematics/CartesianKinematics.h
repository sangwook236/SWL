#if !defined(__SWL_KINEMATICS__CARTESIAN_KINEMATICS__H_)
#define __SWL_KINEMATICS__CARTESIAN_KINEMATICS__H_ 1


#include "swl/kinematics/RobotKinematics.h"
#include "swl/math/TMatrix.h"


namespace swl {

//--------------------------------------------------------------------------------
// class Cartesian2Kinematics: for 2-P Cartesian Robot

class SWL_KINEMATICS_API Cartesian2Kinematics : public RobotKinematics
{
public:
	typedef RobotKinematics base_type;

public:
	Cartesian2Kinematics();
	virtual ~Cartesian2Kinematics();

private:
	Cartesian2Kinematics(const Cartesian2Kinematics &rhs);
	Cartesian2Kinematics & operator=(const Cartesian2Kinematics &rhs);

public:
	///
	/*virtual*/ bool solveForward(const coords_type &jointCoords, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL);
	/*virtual*/ bool solveInverse(const coords_type &cartesianCoords, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL);
//@}
};


//--------------------------------------------------------------------------------
// class Cartesian3Kinematics: for 3-P Cartesian Robot

class SWL_KINEMATICS_API Cartesian3Kinematics: public Cartesian2Kinematics
{
public:
	typedef Cartesian2Kinematics base_type;

public:
	Cartesian3Kinematics();
	virtual ~Cartesian3Kinematics();

private:
	Cartesian3Kinematics(const Cartesian3Kinematics &rhs);
	Cartesian3Kinematics & operator=(const Cartesian3Kinematics &rhs);

public:
	///
	/*virtual*/ bool solveForward(const coords_type &jointCoords, coords_type &cartesianCoords, const coords_type *refCartesianCoordsPtr = NULL);
	/*virtual*/ bool solveInverse(const coords_type &cartesianCoords, coords_type &jointCoords, const coords_type *refJointCoordsPtr = NULL);
//@}
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__CARTESIAN_KINEMATICS__H_

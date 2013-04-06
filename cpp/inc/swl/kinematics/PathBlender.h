#if !defined(__SWL_KINEMATICS__PATH_BLENDER__H_)
#define __SWL_KINEMATICS__PATH_BLENDER__H_ 1


#include "swl/kinematics/PathPlanner.h"


namespace swl {

//--------------------------------------------------------------------------------
// class PathBlender

class SWL_KINEMATICS_API PathBlender: public PathPlanner
{
public:
	typedef PathPlanner base_type;

private:
	explicit PathBlender(const PathBlender &rhs);
	PathBlender & operator=(const PathBlender &rhs);

public:
	PathBlender(KinematicsBase &kinem, const PathPlanner &aPrevPlanner, const PathPlanner &aNextPlanner);
	virtual ~PathBlender();

public:
	///
	/*virtual*/ bool plan();

	/// in joint coordinates
	/*virtual*/ bool getNextPose(coords_type &aPose, const coords_type *refPose = 0L);

	/// in joint coordinates
	/*virtual*/ void addViaPose(const coords_type & /*aViaPose*/)  {}
	/*virtual*/ void addViaPose(pose_ctr::const_iterator /*citFirstPose*/, pose_ctr::const_iterator /*citLastPose*/)  {}

protected:
	///
	const PathPlanner &prevPlanner_;
	const PathPlanner &nextPlanner_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__PATH_BLENDER__H_

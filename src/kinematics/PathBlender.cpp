#include "swl/Config.h"
#include "swl/kinematics/PathBlender.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class PathBlender

PathBlender::PathBlender(KinematicsBase &kinem, const PathPlanner &aPrevPlanner, const PathPlanner &aNextPlanner)
: base_type(kinem),
  prevPlanner_(aPrevPlanner), nextPlanner_(aNextPlanner)
{
}

PathBlender::PathBlender(const PathBlender &rhs)
: base_type(rhs),
  prevPlanner_(rhs.prevPlanner_), nextPlanner_(rhs.nextPlanner_)
{
}

PathBlender::~PathBlender()
{
}

bool PathBlender::getNextPose(PathBlender::coords_type &aPose, const PathBlender::coords_type *refPose /*= NULL*/)
{
	// TODO [add] >>
	return true;
}

bool PathBlender::plan()
{
	currStep_ = 0u;
	currPose_.assign(initPose_.begin(), initPose_.end());

	// TODO [add] >>
	return true;
}

}  // namespace swl

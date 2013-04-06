#include "swl/Config.h"
#include "swl/kinematics/CurvePathPlanner.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------------
// class CurvePathPlanner

CurvePathPlanner::CurvePathPlanner(KinematicsBase &kinem, const CurvePathPlanner::ECurve curveType)
: base_type(kinem),
  poseCtr_(), curveType_(curveType)
{
}

CurvePathPlanner::CurvePathPlanner(const CurvePathPlanner &rhs)
: base_type(rhs),
  poseCtr_(rhs.poseCtr_), curveType_(rhs.curveType_)
{
}

CurvePathPlanner::~CurvePathPlanner()
{
}

void CurvePathPlanner::reset()
{
	base_type::reset();

	poseCtr_.clear();
	//curveType_ = CURVE_PTP;
}

bool CurvePathPlanner::getNextPose(CurvePathPlanner::coords_type &aPose, const CurvePathPlanner::coords_type *refPose /*= NULL*/)
{
	// TODO [add] >>
	return true;
}

void CurvePathPlanner::addViaPose(const CurvePathPlanner::coords_type &aViaPose)
{  poseCtr_.push_back(aViaPose);  }

void CurvePathPlanner::addViaPose(CurvePathPlanner::pose_ctr::const_iterator citFirstPose, CurvePathPlanner::pose_ctr::const_iterator citLastPose)
{  poseCtr_.insert(poseCtr_.end(), citFirstPose, citLastPose);  }

bool CurvePathPlanner::plan()
{
	currStep_ = 0u;
	currPose_.assign(initPose_.begin(), initPose_.end());

	// TODO [add] >>
	return true;
}

}  // namespace swl

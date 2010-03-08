#if !defined(__SWL_KINEMATICS__PATH_PLANNER__H_)
#define __SWL_KINEMATICS__PATH_PLANNER__H_ 1


#include "swl/kinematics/ExportKinematics.h"
#include <vector>
#include <list>


namespace swl {

class KinematicsBase;

//-----------------------------------------------------------------------------------------
//

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_KINEMATICS_EXPORT_TEMPLATE template class SWL_KINEMATICS_API std::vector<double>;
#endif


//--------------------------------------------------------------------------------
// class PathPlanner

class SWL_KINEMATICS_API PathPlanner
{
public:
	//typedef PathPlanner				base_type;
	typedef std::vector<double>			coords_type;
	typedef std::list<coords_type>		pose_ctr;

protected:
	PathPlanner(KinematicsBase &kinem);
	explicit PathPlanner(const PathPlanner &rhs);
	virtual ~PathPlanner();

private:
	PathPlanner & operator=(const PathPlanner &rhs);

public:
	///
	virtual bool plan() = 0;
	virtual bool isDone() const;

	///
	virtual void reset();
	void resetStep()  {  currStep_ = maxStep_ = 0u;  }

	/// in joint coordinates
	const coords_type & getCurrPose() const;
	virtual bool getNextPose(coords_type &aPose, const coords_type *refPose = NULL) = 0;

	/// in joint coordinates
	void setInitPose(const coords_type &aInitPose);
	void setFinalPose(const coords_type &aFinalPose);

	/// in joint coordinates
	virtual void addViaPose(const coords_type & /*aViaPose*/) = 0;
	virtual void addViaPose(pose_ctr::const_iterator /*citFirstPose*/, pose_ctr::const_iterator /*citLastPose*/) = 0;

	///
	void setVelocityRatio(const double dVelocityRatio)
	{
		if (dVelocityRatio <= 0.0) velocityRatio_ = 0.001;
		else if (velocityRatio_ > 1.0) velocityRatio_ = 1.0;
		else velocityRatio_ = dVelocityRatio;
	}
	double getVelocityRatio() const  {  return velocityRatio_;  }

	void setSamplingTime(const unsigned int uiSamplingTime)
	{  samplingTime_ = uiSamplingTime ? uiSamplingTime : 1u;  }
	unsigned int getSamplingTime() const  {  return samplingTime_;  }

protected:
	///
	KinematicsBase &kinematics_;
	size_t dof_;

	/// in joint coordinates
	coords_type initPose_, finalPose_, currPose_;

	/// 0.0 < Velocity Ratio <= 1.0
	double velocityRatio_;

	/// sampling time, [msec]
	unsigned int samplingTime_;

	///
	unsigned int currStep_, maxStep_;
};

}  // namespace swl


#endif  // __SWL_KINEMATICS__PATH_PLANNER__H_

#if !defined(__SWL_GRAPHICS__SHAPE__H_)
#define __SWL_GRAPHICS__SHAPE__H_ 1


#include "swl/graphics/GraphicsObj.h"
#include "swl/graphics/GeometryPool.h"
#include "swl/graphics/Appearance.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class Shape

class SWL_GRAPHICS_API Shape: public GraphicsObj
{
public:
	typedef GraphicsObj						base_type;
	typedef GeometryPool::geometry_id_type	geometry_id_type;
	typedef GeometryPool::geometry_type		geometry_type;
	typedef Appearance						appearance_type;

public:
	Shape();
	Shape(const Shape &rhs);
	virtual ~Shape();

	Shape & operator=(const Shape &rhs);

public:
	///
	geometry_id_type & getGeometryId()  {  return geometryId_;  }
	const geometry_id_type & getGeometryId() const  {  return geometryId_;  }

	geometry_type getGeometry() const;

	//
	appearance_type & getAppearance()  {  return appearance_;  }
	const appearance_type & getAppearance() const  {  return appearance_;  }

private:
	///
	geometry_id_type geometryId_;
	appearance_type appearance_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SHAPE__H_

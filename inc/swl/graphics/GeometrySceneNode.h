#if !defined(__SWL_GRAPHICS__GEOMETRY_SCENE_NODE__H_)
#define __SWL_GRAPHICS__GEOMETRY_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/graphics/GeometryPool.h"


namespace swl {

class GeometryPool;

//--------------------------------------------------------------------------
// class GeometrySceneNode

template<typename SceneVisitor>
class GeometrySceneNode: public LeafSceneNode<SceneVisitor>
{
public:
	typedef LeafSceneNode					base_type;
	typedef GeometryPool::geometry_id_type	geometry_id_type;
	typedef GeometryPool::geometry_type		geometry_type;

public:
#if defined(UNICODE) || defined(_UNICODE)
	GeometrySceneNode(const geometry_id_type &geometryId, const std::wstring &name = std::wstring())
#else
	GeometrySceneNode(const geometry_id_type &geometryId, const std::string &name = std::string())
#endif
	: base_type(name),
	  geometryId_(geometryId) //geometryId_(GeometryPool::UNDEFINED_GEOMETRY_ID)
	{}
	GeometrySceneNode(const GeometrySceneNode &rhs)
	: base_type(rhs),
	  geometryId_(rhs.geometryId_)
	{}
	virtual ~GeometrySceneNode()
	{}

	GeometrySceneNode & operator=(const GeometrySceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		geometryId_ = rhs.geometryId_;
		return *this;
	}
 
public:
	/*final*/ /*virtual*/ void accept(const visitor_type &visitor) const
	{
		visitor.visit(*this);
	}

	geometry_id_type & getGeometryId()  {  return geometryId_;  }
	const geometry_id_type & getGeometryId() const  {  return geometryId_;  }

	geometry_type getGeometry() const
	{
		return GeometryPool::getInstance().getGeometry(geometryId_);
	}

private:
	geometry_id_type geometryId_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GEOMETRY_SCENE_NODE__H_

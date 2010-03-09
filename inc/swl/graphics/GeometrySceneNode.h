#if !defined(__SWL_GRAPHICS__GEOMETRY_SCENE_NODE__H_)
#define __SWL_GRAPHICS__GEOMETRY_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/graphics/GeometryPool.h"


namespace swl {

class GeometryPool;

//--------------------------------------------------------------------------
// class GeometrySceneNode

class SWL_GRAPHICS_API GeometrySceneNode: public LeafSceneNode
{
public:
	typedef LeafSceneNode					base_type;
	typedef GeometryPool::geometry_id_type	geometry_id_type;
	typedef GeometryPool::geometry_type		geometry_type;

public:
	GeometrySceneNode(const geometry_id_type &geometryId);
	GeometrySceneNode(const GeometrySceneNode &rhs);
	virtual ~GeometrySceneNode();

	GeometrySceneNode & operator=(const GeometrySceneNode &rhs);
 
public:
	/*final*/ /*virtual*/ void accept(const ISceneVisitor &visitor) const;

	geometry_id_type & getGeometryId()  {  return geometryId_;  }
	const geometry_id_type & getGeometryId() const  {  return geometryId_;  }

	geometry_type getGeometry() const;

private:
	geometry_id_type geometryId_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GEOMETRY_SCENE_NODE__H_

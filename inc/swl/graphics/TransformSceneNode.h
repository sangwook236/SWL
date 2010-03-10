#if !defined(__SWL_GRAPHICS__TRANSFORM_SCENE_NODE__H_)
#define __SWL_GRAPHICS__TRANSFORM_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/math/TMatrix.h"


namespace swl {

//--------------------------------------------------------------------------
// class TransformSceneNode

class SWL_GRAPHICS_API TransformSceneNode: public GroupSceneNode
{
public:
	typedef GroupSceneNode		base_type;
	typedef TMatrix3<double>	transform_type;

public:
	TransformSceneNode();
	TransformSceneNode(const TransformSceneNode &rhs);
	virtual ~TransformSceneNode();

	TransformSceneNode & operator=(const TransformSceneNode &rhs);

public:
	/*final*/ /*virtual*/ void accept(const ISceneVisitor &visitor) const;

	transform_type & getTransform()  {  return transform_;  }
	const transform_type & getTransform() const  {  return transform_;  }

private:
	transform_type transform_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__TRANSFORM_SCENE_NODE__H_

#if !defined(__SWL_GRAPHICS__TRANSFORM_SCENE_NODE__H_)
#define __SWL_GRAPHICS__TRANSFORM_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/math/TMatrix.h"


namespace swl {

//--------------------------------------------------------------------------
// class TransformSceneNode

template<typename SceneVisitor>
class TransformSceneNode: public GroupSceneNode<SceneVisitor>
{
public:
	typedef GroupSceneNode<SceneVisitor> base_type;
	typedef TMatrix3<double> transform_type;
	typedef typename base_type::visitor_type visitor_type;

public:
	TransformSceneNode()
	: base_type(),
	  transform_()
	{}
	TransformSceneNode(const TransformSceneNode &rhs)
	: base_type(rhs),
	  transform_(rhs.transform_)
	{}
	virtual ~TransformSceneNode()
	{}

	TransformSceneNode & operator=(const TransformSceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		transform_ = rhs.transform_;
		return *this;
	}

public:
	/*final*/ /*virtual*/ void accept(const visitor_type &visitor) const
	{
		visitor.visit(*this);
	}

	transform_type & getTransform()  {  return transform_;  }
	const transform_type & getTransform() const  {  return transform_;  }

private:
	transform_type transform_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__TRANSFORM_SCENE_NODE__H_

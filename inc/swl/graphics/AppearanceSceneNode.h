#if !defined(__SWL_GRAPHICS__APPEARANCE_SCENE_NODE__H_)
#define __SWL_GRAPHICS__APPEARANCE_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/graphics/Appearance.h"


namespace swl {

//--------------------------------------------------------------------------
// class AppearanceSceneNode

class SWL_GRAPHICS_API AppearanceSceneNode: public LeafSceneNode
{
public:
	typedef LeafSceneNode	base_type;
	typedef Appearance		appearance_type;

public:
	AppearanceSceneNode(const appearance_type &appearance);
	AppearanceSceneNode(const AppearanceSceneNode &rhs);
	virtual ~AppearanceSceneNode();

	AppearanceSceneNode & operator=(const AppearanceSceneNode &rhs);
 
public:
	/*final*/ /*virtual*/ void accept(const ISceneVisitor &visitor) const;

	appearance_type & getAppearance()  {  return appearance_;  }
	const appearance_type & getAppearance() const  {  return appearance_;  }

private:
	appearance_type appearance_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__APPEARANCE_SCENE_NODE__H_

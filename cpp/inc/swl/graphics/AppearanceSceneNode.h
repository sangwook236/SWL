#if !defined(__SWL_GRAPHICS__APPEARANCE_SCENE_NODE__H_)
#define __SWL_GRAPHICS__APPEARANCE_SCENE_NODE__H_ 1


#include "swl/graphics/SceneNode.h"
#include "swl/graphics/Appearance.h"


namespace swl {

//--------------------------------------------------------------------------
// class AppearanceSceneNode

template<typename SceneVisitor>
class AppearanceSceneNode: public LeafSceneNode<SceneVisitor>
{
public:
	typedef LeafSceneNode<SceneVisitor> base_type;
	typedef Appearance appearance_type;
	typedef typename base_type::visitor_type visitor_type;

public:
#if defined(_UNICODE) || defined(UNICODE)
	AppearanceSceneNode(appearance_type &appearance, const std::wstring &name = std::wstring())
#else
	AppearanceSceneNode(appearance_type &appearance, const std::string &name = std::string())
#endif
	: base_type(name),
	  appearance_(appearance)
	{}
	AppearanceSceneNode(const AppearanceSceneNode &rhs)
	: base_type(rhs),
	  appearance_(rhs.appearance_)
	{}
	virtual ~AppearanceSceneNode()
	{}

	AppearanceSceneNode & operator=(const AppearanceSceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		appearance_ = rhs.appearance_;
		return *this;
	}

public:
	/*final*/ /*virtual*/ void accept(const visitor_type &visitor) const
	{
		visitor.visit(*this);
	}

	appearance_type & getAppearance()  {  return appearance_;  }
	const appearance_type & getAppearance() const  {  return appearance_;  }

private:
	appearance_type &appearance_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__APPEARANCE_SCENE_NODE__H_

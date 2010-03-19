#if !defined(__SWL_GRAPHICS__DRAWABLE_INTERFACE__H_)
#define __SWL_GRAPHICS__DRAWABLE_INTERFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct IDrawable: mix-in style class

struct IDrawable
{
protected:
	virtual ~IDrawable()  {}

public:
	//
	virtual void draw() const = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__DRAWABLE_INTERFACE__H_

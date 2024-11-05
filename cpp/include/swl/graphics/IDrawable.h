#if !defined(__SWL_GRAPHICS__DRAWABLE_INTRFACE__H_)
#define __SWL_GRAPHICS__DRAWABLE_INTRFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct IDrawable: mix-in style class

struct IDrawable
{
public:
	//typedef IDrawable base_type;

protected:
	virtual ~IDrawable()  {}

public:
	//
	virtual void draw() const = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__DRAWABLE_INTRFACE__H_

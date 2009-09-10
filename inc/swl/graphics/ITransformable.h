#if !defined(__SWL_GRAPHICS__TRANSFORMABLE_INTERFACE__H_)
#define __SWL_GRAPHICS__TRANSFORMABLE_INTERFACE__H_ 1


namespace swl {

template<typename T> class TMatrix3;


//-----------------------------------------------------------------------------------------
// struct ITransformable: mix-in style class

struct ITransformable
{
protected:
	virtual ~ITransformable()  {}

public:
	///
	virtual bool transform(const TMatrix3<double> &mat) = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__TRANSFORMABLE_INTERFACE__H_

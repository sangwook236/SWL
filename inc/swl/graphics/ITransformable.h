#if !defined(__SWL_GRAPHICS__TRANSFORMABLE_INTERFACE__H_)
#define __SWL_GRAPHICS__TRANSFORMABLE_INTERFACE__H_ 1


namespace swl {

template<typename T> class TMatrix2;
template<typename T> class TMatrix3;


//-----------------------------------------------------------------------------------------
// struct ITransformable2: mix-in style class

struct ITransformable2
{
protected:
	virtual ~ITransformable2()  {}

public:
	///
	virtual bool move(const TMatrix2<double> &mat) = 0;
};


//-----------------------------------------------------------------------------------------
// struct ITransformable3: mix-in style class

struct ITransformable3
{
protected:
	virtual ~ITransformable3()  {}

public:
	///
	virtual bool transform(const TMatrix3<double> &mat) = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__TRANSFORMABLE_INTERFACE__H_

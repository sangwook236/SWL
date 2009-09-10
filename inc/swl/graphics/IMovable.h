#if !defined(__SWL_GRAPHICS__MOVABLE_INTERFACE__H_)
#define __SWL_GRAPHICS__MOVABLE_INTERFACE__H_ 1


namespace swl {

template<typename T> class TMatrix2;


//-----------------------------------------------------------------------------------------
// struct IMovable: mix-in style class

struct IMovable
{
protected:
	virtual ~IMovable()  {}

public:
	///
	virtual bool move(const TMatrix2<double> &mat) = 0;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__MOVABLE_INTERFACE__H_

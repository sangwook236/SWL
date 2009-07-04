#if !defined(__SWL_BASE__SIZE__H_)
#define __SWL_BASE__SIZE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct Size2

template<typename T>
struct Size2
{
public:
	typedef T value_type;

public:    
    Size2(const T& Width = T(0), const T& Height = T(0))
	: width(Width), height(Height)
	{}
    Size2(const Size2<T>& rhs)
	: width(rhs.width), height(rhs.height)
	{}
	explicit Size2(const T rhs[2])
	: width(rhs[0]), height(rhs[1])
	{}
    ~Size2()  {}

    Size2<T>& operator=(const Size2<T>& rhs)
    {
        if (this == &rhs)  return *this;
        width = rhs.width;
		height = rhs.height;
        return *this;
    }

public:
	union
	{
		struct  {  T width, height;  };
		struct  {  T dx, dy;  };
	};
};


//-----------------------------------------------------------------------------------------
// struct Size3

template<typename T>
struct Size3
{
public:
	typedef T value_type;

public:    
    Size3(const T& Width = T(0), const T& Height = T(0), const T& Depth = T(0))
	: width(Width), height(Height), depth(Depth)
	{}
    Size3(const Size3<T>& rhs)
	: width(rhs.width), height(rhs.height), depth(rhs.depth)
	{}
	explicit Size3(const T rhs[3])
	: width(rhs[0]), height(rhs[1]), depth(rhs[2])
	{}
    ~Size3()  {}

    Size3<T>& operator=(const Size3<T>& rhs)
    {
        if (this == &rhs)  return *this;
        width = rhs.width;
		height = rhs.height;
		depth = rhs.depth;
        return *this;
    }

public:
	union
	{
		struct  {  T width, height, depth;  };
		struct  {  T dx, dy, dz;  };
	};
};

}  // namespace swl


#endif  //  __SWL_BASE__SIZE__H_

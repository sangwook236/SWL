#if !defined(__SWL_COMMON__TUPLE__H_)
#define __SWL_COMMON__TUPLE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// struct Tuple2

template<typename T>
struct Tuple2
{
public:
	typedef T value_type;

public:    
    Tuple2(const T& First = T(0), const T& Second = T(0))
	: first(First), second(Second)
	{}
    Tuple2(const Tuple2<T>& rhs)
	: first(rhs.first), second(rhs.second)
	{}
	explicit Tuple2(const T rhs[2])
	: first(rhs[0]), second(rhs[1])
	{}
    ~Tuple2()  {}

    Tuple2<T>& operator=(const Tuple2<T>& rhs)
    {
        if (this == &rhs)  return *this;
        first = rhs.first;
		second = rhs.second;
        return *this;
    }

public:
	union
	{
		struct  {  T first, second;  };
		struct  {  T x, y;  };
		struct  {  T min, max;  };
		T tuple[2];
	};
};


//-----------------------------------------------------------------------------------------
// struct Tuple3

template<typename T>
struct Tuple3
{
public:
	typedef T value_type;

public:    
    Tuple3(const T& First = T(0), const T& Second = T(0), const T& Third = T(0))
	: first(First), second(Second), third(Third)
	{}
    Tuple3(const Tuple3<T>& rhs)
	: first(rhs.first), second(rhs.second), third(rhs.third)
	{}
	explicit Tuple3(const T rhs[3])
	: first(rhs[0]), second(rhs[1]), third(rhs[2])
	{}
    ~Tuple3()  {}

    Tuple3<T>& operator=(const Tuple3<T>& rhs)
    {
        if (this == &rhs)  return *this;
        first = rhs.first;
		second = rhs.second;
		third = rhs.third;
        return *this;
    }

public:
	union
	{
		struct  {  T first, second, third;  };
		struct  {  T x, y, z;  };
		T tuple[3];
	};
};


//-----------------------------------------------------------------------------------------
// struct Tuple4

template<typename T>
struct Tuple4
{
public:
	typedef T	value_type;

public:    
    Tuple4(const T& First = T(0), const T& Second = T(0), const T& Third = T(0), const T& Fourth = T(0))
	:  first(First), second(Second), third(Third), fourth(Fourth)
	{}
    Tuple4(const Tuple4<T>& rhs)
	:  first(rhs.first), second(rhs.second), third(rhs.third), fourth(rhs.fourth)
	{}
	explicit Tuple4(const T rhs[4])
	:  first(rhs[0]), second(rhs[1]), third(rhs[2]), fourth(rhs[3])
	{}
    ~Tuple4()  {}

    Tuple4<T>& operator=(const Tuple4<T>& rhs)
    {
        if (this == &rhs)  return *this;
        first = rhs.first;
		second = rhs.second;
		third = rhs.third;
		fourth = rhs.fourth;
        return *this;
    }

public:
	union
	{
		struct  {  T first, second, third, fourth;  };
		struct  {  T x, y, z, w;  };
		struct  {  T left, right, bottom, top;  };
		T tuple[4];
	};
};

}  // namespace swl


#endif  //  __SWL_COMMON__TUPLE__H_

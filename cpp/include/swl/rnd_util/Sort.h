#if !defined(__SWL_RND_UTIL__SORT__H_)
#define __SWL_RND_UTIL__SORT__H_ 1


#include <algorithm>
#include <functional>
#include <vector>
#include <iterator>


namespace swl {

//--------------------------------------------------------------------------
// Sorting Algorithm

class Sort
{
public:
	template<typename BidirectionalIterator>
	static void mergeSort(BidirectionalIterator first, BidirectionalIterator last)
	{
		const std::size_t len = std::distance(first, last);
		if (len <= 1) return;
		BidirectionalIterator middle = std::next(first, len / 2);
		mergeSort(first, middle);
		mergeSort(middle, last);
		std::inplace_merge(first, middle, last);
	}

	template<typename ForwardIterator>
	static void quickSort(ForwardIterator first, ForwardIterator last)
	{
		if (first == last) return;
		const auto pivot = *std::next(first, std::distance(first, last) / 2);
#if defined(__GNUC__)
		ForwardIterator middle1 = std::partition(first, last, [pivot](const decltype(pivot)& em) { return em < pivot; });
		ForwardIterator middle2 = std::partition(middle1, last, [pivot](const decltype(pivot)& em) { return !(pivot < em); });
#else
		ForwardIterator middle1 = std::partition(first, last, [pivot](const auto& em) { return em < pivot; });
		ForwardIterator middle2 = std::partition(middle1, last, [pivot](const auto& em) { return !(pivot < em); });
#endif
		quickSort(first, middle1);
		quickSort(middle2, last);
	}
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__SORT__H_

#include "swl/rnd_util/LevenshteinDistance.h"
#include <string>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------
//

// [ref] http://rosettacode.org/wiki/Levenshtein_distance

std::size_t computeLevenshteinDistance(const std::string &s1, const std::string &s2)
{
	const std::size_t m(s1.size());
	const std::size_t n(s2.size());

	if (0 == m) return n;
	if (0 == n) return m;

	std::vector<std::size_t> costs(n + 1);
	for (std::size_t k = 0; k <= n; ++k) costs[k] = k;

	std::size_t i = 0;
	for (std::string::const_iterator it1 = s1.begin(); it1 != s1.end(); ++it1, ++i)
	{
		costs[0] = i + 1;
		std::size_t corner = i;

		std::size_t j = 0;
		for (std::string::const_iterator it2 = s2.begin(); it2 != s2.end(); ++it2, ++j)
		{
			std::size_t upper = costs[j+1];
			if (*it1 == *it2)
			{
				costs[j+1] = corner;
			}
			else
			{
				std::size_t t(upper<corner?upper:corner);
				costs[j+1] = (costs[j] < t ? costs[j] : t) + 1;
			}

			corner = upper;
		}
	}

	std::size_t result = costs[n];

	return result;
}

}  // namespace swl

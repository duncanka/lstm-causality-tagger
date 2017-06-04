#ifndef LSTM_CAUSALITY_UTILITIES_H_
#define LSTM_CAUSALITY_UTILITIES_H_

#include <algorithm>
#include <boost/spirit/home/support/container.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <ostream>
#include <string>


template <class T, class Cmp>
size_t ReallyDeleteIf(T* container, Cmp& pred) {
  auto new_end = std::remove_if(container->begin(), container->end(), pred);
  size_t length_diff = container->end() - new_end;
  container->erase(new_end, container->end());
  return length_diff;
}


template <class Cmp, class Comparand = typename Cmp::second_argument_type>
struct ThresholdedCmp {
  const Comparand& threshold;

  ThresholdedCmp(const Comparand& threshold) : threshold(threshold) {}

  bool operator()(const Comparand& comparand) {
    return Cmp()(comparand, threshold);
  }
};


// From http://stackoverflow.com/a/9600752
class IndentingOStreambuf : public std::streambuf {
public:
  explicit IndentingOStreambuf(std::streambuf* dest, unsigned indent = 4)
      : dest(dest), atStartOfLine(true), indent(indent, ' '), owner(nullptr) {}

  explicit IndentingOStreambuf(std::ostream& dest, unsigned indent = 4)
      : dest(dest.rdbuf()), atStartOfLine(true), indent(indent, ' '),
        owner(&dest) {
    owner->rdbuf(this);
  }

  virtual ~IndentingOStreambuf() {
    if (owner != nullptr) {
      owner->rdbuf(dest);
    }
  }

protected:
    std::streambuf* dest;
    bool atStartOfLine;
    std::string indent;
    std::ostream* owner;

  virtual int overflow(int ch) {
    if (atStartOfLine && ch != '\n') {
      dest->sputn(indent.data(), indent.size());
    }
    atStartOfLine = ch == '\n';
    return dest->sputc(ch);
  }
};


#define BOOST_TRANSFORMED_RANGE(range_or_container, elt_name, elt_fn) \
  range_or_container | boost::adaptors::transformed( \
      [&](const typename decltype(range_or_container)::value_type& elt_name) { \
        return elt_fn; \
      })


// from http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    FloatsAlmostSame(T x, T y, int ulp=4) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x - y)
      < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
      // unless the result is subnormal
      || std::abs(x - y) < std::numeric_limits<T>::min()
      || (std::isnan(x) && std::isnan(y));
}


template <class Container>
bool Contains(const Container& container,
              const typename Container::value_type& element) {
  auto find_result = std::find(std::begin(container), std::end(container),
                               element);
  return find_result != std::end(container);
}


// Based on https://stackoverflow.com/a/12399290
template <typename T, typename Comparator=std::less<T>>
std::vector<size_t> SortIndices(const std::vector<T> &v) {
  // initialize original index locations
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(indices.begin(), indices.end(),
            [&v](size_t i1, size_t i2) {return Comparator()(v[i1], v[i2]);});

  return indices;
}


// From https://stackoverflow.com/a/1267878
template<class T>
void Reorder(std::vector<T> *v, const std::vector<size_t>& order) {
  for (unsigned s = 1; s < order.size(); ++s) {
    unsigned d;
    for (d = order[s]; d < s; d = order[d]);
    if (d == s)
      while (d = order[d], d != s)
        swap((*v)[s], (*v)[d]);
  }
}


// Generic output function for containers. Disabled for non-container types or
// variants of std::basic_string.
template<class Container,
         typename std::enable_if<
           boost::spirit::traits::is_container<Container>::value
             && !std::is_same<std::basic_string<typename Container::value_type>,
                              Container>::value,
           int>::type = 0>
std::ostream& operator<<(std::ostream& os, const Container& c) {
  os << '[';
  for (auto iter = c.begin(), end = c.end(); iter != end; ++iter) {
    os << *iter;
    auto next_iter = iter;
    ++next_iter;
    if (next_iter != end)
      os << ", ";
  }
  os << ']';
  return os;
}

#endif /* LSTM_CAUSALITY_UTILITIES_H_ */

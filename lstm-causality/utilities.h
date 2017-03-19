#ifndef LSTM_CAUSALITY_UTILITIES_H_
#define LSTM_CAUSALITY_UTILITIES_H_

#include <algorithm>
#include <boost/range/adaptor/transformed.hpp>
#include <ostream>


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
    AlmostSame(T x, T y, int ulp=4) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x - y)
      < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
      // unless the result is subnormal
      || std::abs(x - y) < std::numeric_limits<T>::min()
      || (std::isnan(x) && std::isnan(y));
}


#endif /* LSTM_CAUSALITY_UTILITIES_H_ */

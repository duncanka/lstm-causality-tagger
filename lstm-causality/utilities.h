#ifndef LSTM_CAUSALITY_UTILITIES_H_
#define LSTM_CAUSALITY_UTILITIES_H_

#include <algorithm>

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


#endif /* LSTM_CAUSALITY_UTILITIES_H_ */

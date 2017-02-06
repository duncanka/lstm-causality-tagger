#include <iostream>

using namespace std;

#include "BecauseData.h"

ostream& operator<<(ostream& os, const BecauseRelation& rel) {
  auto print_tokens = [&](const BecauseRelation::IndexList& indices) {
    const auto& tokens = rel.GetTokensForIndices(indices);
    for (auto i = tokens.begin(), end = tokens.end(); i != end; ++i) {
      os << i->get();
      if (i != end - 1)
      os << ' ';
    }
  };

  os << rel.GetRelationName() << "(connective=";
  print_tokens(rel.connective_indices);
  for (unsigned i = 0; i < rel.arg_names.size(); ++i) {
    os << ", " << rel.ArgNameForArgIndex(i) << '=';
    print_tokens(rel.arguments[i]);
  }
  os << ')';

  return os;
}


const vector<string> CausalityRelation::ARG_NAMES = {
    "Cause", "Effect", "Means"
};

const vector<string> CausalityRelation::TYPE_NAMES = {
    "Consequence", "Motivation", "Purpose"
};

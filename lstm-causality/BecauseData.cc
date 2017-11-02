#include <iostream>

using namespace std;

#include "BecauseData.h"


void BecauseRelation::PrintTokens(
    ostream& os, const BecauseRelation::IndexList& indices) const {
  const auto& tokens = GetTokensForIndices(indices);
  for (auto i = tokens.begin(), end = tokens.end(); i != end; ++i) {
    os << i->get();
    if (i != end - 1)
    os << ' ';
  }
};


ostream& operator<<(ostream& os, const BecauseRelation& rel) {
  os << rel.GetRelationName() << "(connective=";
  rel.PrintTokens(os, rel.connective_indices);
  for (unsigned i = 0; i < rel.arg_names.size(); ++i) {
    os << ", " << rel.ArgNameForArgIndex(i) << '=';
    rel.PrintTokens(os, rel.arguments[i]);
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

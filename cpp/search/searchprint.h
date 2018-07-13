#ifndef SEARCHPRINT_H
#define SEARCHPRINT_H

struct PrintTreeOptions {  
  PrintTreeOptions();

  PrintTreeOptions& maxDepth(int);
  PrintTreeOptions& maxChildren(int);
  PrintTreeOptions& minChildrenVisits(uint64_t);
  PrintTreeOptions& minChildVisits(uint64_t);
  PrintTreeOptions& minChildrenVisitsProp(double);
  PrintTreeOptions& minChildVisitsProp(double);

  int maxDepth_;
  int maxChildren_;
  uint64_t minChildrenVisits_;
  uint64_t minChildVisits_;
  double minChildrenVisitsProp_;
  double minChildVisitsProp_;
};

inline PrintTreeOptions::PrintTreeOptions()
  :maxDepth_(1),
   maxChildren_(100000),
   minChildrenVisits_(1),
   minChildVisits_(1),
   minChildrenVisitsProp_(0.0),
   minChildVisitsProp_(0.0)
{}

inline PrintTreeOptions& PrintTreeOptions::maxDepth(int d) { maxDepth_ = d; return *this;}
inline PrintTreeOptions& PrintTreeOptions::maxChildren(int c) { maxChildren_ = c; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minChildrenVisits(uint64_t v) { minChildrenVisits_ = v; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minChildVisits(uint64_t v) { minChildVisits_ = v; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minChildrenVisitsProp(double p) { minChildrenVisitsProp_ = p; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minChildVisitsProp(double p) { minChildVisitsProp_ = p; return *this;}

#endif

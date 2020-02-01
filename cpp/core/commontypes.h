#ifndef COMMONTYPES_H
#define COMMONTYPES_H

struct enabled_t {
  enum value { TRUE, FALSE, AUTO };
  value x;

  enabled_t() = default;
  constexpr enabled_t(value a) : x(a) { }
  explicit operator bool() = delete;
  constexpr bool operator==(enabled_t a) const { return x == a.x; }
  constexpr bool operator!=(enabled_t a) const { return x != a.x; }

  std::string toString() {
    return x == TRUE ? "true" : x == FALSE ? "false" : "auto";
  }

  static bool tryParse(const std::string& v, enabled_t& buf) {
    if(v == "1" || v == "t" || v == "true" || v == "enabled" || v == "y" || v == "yes")
      buf = TRUE;
    else if(v == "0" || v == "f" || v == "false" || v == "disabled" || v == "n" || v == "no")
      buf = FALSE;
    else if(v == "auto")
      buf = AUTO;
    else
      return false;
    return true;
  }

};

#endif //COMMONTYPES_H

#ifndef COMMONTYPES_H
#define COMMONTYPES_H

struct enabled_t {
  enum value { False, True, Auto };
  value x;

  enabled_t() = default;
  constexpr enabled_t(value a) : x(a) { }
  explicit operator bool() = delete;
  constexpr bool operator==(enabled_t a) const { return x == a.x; }
  constexpr bool operator!=(enabled_t a) const { return x != a.x; }

  std::string toString() {
    return x == True ? "true" : x == False ? "false" : "auto";
  }

  static bool tryParse(const std::string& v, enabled_t& buf) {
    if(v == "1" || v == "t" || v == "true" || v == "enabled" || v == "y" || v == "yes")
      buf = True;
    else if(v == "0" || v == "f" || v == "false" || v == "disabled" || v == "n" || v == "no")
      buf = False;
    else if(v == "auto")
      buf = Auto;
    else
      return false;
    return true;
  }

};

#endif //COMMONTYPES_H

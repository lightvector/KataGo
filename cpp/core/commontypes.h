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
};

#endif //COMMONTYPES_H

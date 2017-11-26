"""Interpret SGF property values.

This is intended for use with SGF FF[4]; see http://www.red-bean.com/sgf/

This supports all general properties and Go-specific properties, but not
properties for other games. Point, Move and Stone values are interpreted as Go
points.

"""

import codecs
from math import isinf, isnan

from . import sgf_grammar

def normalise_charset_name(s):
    """Convert an encoding name to the form implied in the SGF spec.

    In particular, normalises to 'ISO-8859-1' and 'UTF-8'.

    Raises LookupError if the encoding name isn't known to Python.

    """
    return (codecs.lookup(s).name.replace("_", "-").upper()
            .replace("ISO8859", "ISO-8859"))


def interpret_go_point(bb, size):
    """Convert a raw SGF Go Point, Move, or Stone value to coordinates.

    bb   -- bytes-like object
    size -- board size (int)

    Returns a pair (row, col), or None for a pass.

    Raises ValueError if the data is malformed or the coordinates are out of
    range.

    Only supports board sizes up to 26.

    The returned coordinates are in the GTP coordinate system, where (0, 0) is
    the lower left.

    """
    if bb == b"" or (bb == b"tt" and size <= 19):
        return None
    # May propagate ValueError
    col_b, row_b = bb
    col = col_b - 97 # 97 == ord("a")
    row = size - row_b + 96
    if not ((0 <= col < size) and (0 <= row < size)):
        raise ValueError("Invalid point for stone")
    return row, col

def serialise_go_point(move, size):
    """Serialise a Go Point, Move, or Stone value.

    move -- pair (row, col), or None for a pass

    Returns a bytes object.

    Only supports board sizes up to 26.

    The move coordinates are in the GTP coordinate system, where (0, 0) is the
    lower left.

    """
    if not 1 <= size <= 26:
        raise ValueError
    if move is None:
        # Prefer 'tt' where possible, for the sake of older code
        if size <= 19:
            return b"tt"
        else:
            return b""
    row, col = move
    if not ((0 <= col < size) and (0 <= row < size)):
        raise ValueError
    col_b = b"abcdefghijklmnopqrstuvwxy"[col]
    row_b = b"abcdefghijklmnopqrstuvwxy"[size - row - 1]
    return bytes((col_b, row_b))


class _Context:
    def __init__(self, size, encoding):
        self.size = size
        self.encoding = encoding

def interpret_none(bb, context=None):
    """Convert a raw None value to a boolean.

    That is, unconditionally returns True.

    """
    return True

def serialise_none(b, context=None):
    """Serialise a None value.

    Ignores its parameter.

    """
    return b""


def interpret_number(bb, context=None):
    """Convert a raw Number value to the integer it represents.

    This is a little more lenient than the SGF spec: it permits arbitrary
    leading and trailing whitespace, and ignores spaces anywhere inside the
    data.

    From Python 3.6 on, it also permits 'grouping' underscores between
    numerals.

    """
    return int(bb.replace(b' ', b''), 10)

def serialise_number(i, context=None):
    """Serialise a Number value.

    i -- integer

    """
    return str(i).encode("ascii")


def interpret_real(bb, context=None):
    """Convert a raw Real value to the float it represents.

    This is more lenient than the SGF spec: it accepts strings accepted as a
    float by Python's float() (eg "1e3"). It rejects infinities and NaNs.

    """
    result = float(bb)
    if isinf(result):
        raise ValueError("infinite")
    if isnan(result):
        raise ValueError("not a number")
    return result

def serialise_real(f, context=None):
    """Serialise a Real value.

    f -- real number (int or float)

    If the absolute value is too small to conveniently express as a decimal,
    returns b"0" (this currently happens if abs(f) is less than 0.0001).

    """
    f = float(f)
    try:
        i = int(f)
    except OverflowError:
        # infinity
        raise ValueError
    if f == i:
        # avoid trailing '.0'; also avoid scientific notation for large numbers
        return str(i).encode("ascii")
    s = repr(f)
    if 'e-' in s:
        return b"0"
    return s.encode("ascii")


def interpret_double(bb, context=None):
    """Convert a raw Double value to an integer.

    Returns 1 or 2 (unknown values are treated as 1).

    """
    if bb.strip() == b"2":
        return 2
    else:
        return 1

def serialise_double(i, context=None):
    """Serialise a Double value.

    i -- integer (1 or 2)

    (unknown values are treated as 1)

    """
    if i == 2:
        return b"2"
    return b"1"


def interpret_colour(bb, context=None):
    """Convert a raw Color value to an sgfmill colour.

    Returns 'b' or 'w'.

    """
    colour = bb.lower()
    if colour not in (b'b', b'w'):
        raise ValueError
    return colour.decode("ascii")

def serialise_colour(colour, context=None):
    """Serialise a Colour value.

    colour -- 'b' or 'w'

    """
    if colour not in ('b', 'w'):
        raise ValueError
    return colour.upper().encode("ascii")



def interpret_simpletext(bb, context):
    """Convert a raw SimpleText value to a string.

    See sgf_grammar.simpletext_value() for details.

    bb -- raw value

    Returns a string.

    """
    return sgf_grammar.simpletext_value(bb).decode(context.encoding)

def serialise_simpletext(s, context):
    """Serialise a SimpleText value.

    See sgf_grammar.escape_text() for details.

    s -- string

    """
    if not hasattr(s, 'encode'):
        raise TypeError("expected string, given %s" % type(s).__name__)
    return sgf_grammar.escape_text(s.encode(context.encoding))


def interpret_text(bb, context):
    """Convert a raw Text value to a string.

    See sgf_grammar.text_value() for details.

    bb -- raw value

    Returns a string.

    """
    return sgf_grammar.text_value(bb).decode(context.encoding)

def serialise_text(s, context):
    """Serialise a Text value.

    See sgf_grammar.escape_text() for details.

    s -- string

    """
    return serialise_simpletext(s, context)


def interpret_point(bb, context):
    """Convert a raw SGF Point or Stone value to coordinates.

    See interpret_go_point() above for details.

    Returns a pair (row, col).

    """
    result = interpret_go_point(bb, context.size)
    if result is None:
        raise ValueError
    return result

def serialise_point(point, context):
    """Serialise a Point or Stone value.

    point -- pair (row, col)

    See serialise_go_point() above for details.

    """
    if point is None:
        raise ValueError
    return serialise_go_point(point, context.size)


def interpret_move(bb, context):
    """Convert a raw SGF Move value to coordinates.

    See interpret_go_point() above for details.

    Returns a pair (row, col), or None for a pass.

    """
    return interpret_go_point(bb, context.size)

def serialise_move(move, context):
    """Serialise a Move value.

    move -- pair (row, col), or None for a pass

    See serialise_go_point() above for details.

    """
    return serialise_go_point(move, context.size)


def interpret_point_list(values, context):
    """Convert a raw SGF list of Points to a set of coordinates.

    values -- list of bytes-like objects

    Returns a set of pairs (row, col).

    If 'values' is empty, returns an empty set.

    This interprets compressed point lists.

    Doesn't complain if there is overlap, or if a single point is specified as
    a 1x1 rectangle.

    Raises ValueError if the data is otherwise malformed.

    """
    result = set()
    for bb in values:
        # No need to use parse_compose(), as \: would always be an error.
        p1, is_rectangle, p2 = bb.partition(b":")
        if is_rectangle:
            top, left = interpret_point(p1, context)
            bottom, right = interpret_point(p2, context)
            if not (bottom <= top and left <= right):
                raise ValueError
            for row in range(bottom, top+1):
                for col in range(left, right+1):
                    result.add((row, col))
        else:
            pt = interpret_point(p1, context)
            result.add(pt)
    return result

def serialise_point_list(points, context):
    """Serialise a list of Points, Moves, or Stones.

    points -- iterable of pairs (row, col)

    Returns a list of bytes objects.

    If 'points' is empty, returns an empty list.

    Doesn't produce a compressed point list.

    """
    result = [serialise_point(point, context) for point in points]
    result.sort()
    return result


def interpret_AP(s, context):
    """Interpret an AP (application) property value.

    Returns a pair of strings (name, version number)

    Permits the version number to be missing (which is forbidden by the SGF
    spec), in which case the second returned value is an empty string.

    """
    application, version = sgf_grammar.parse_compose(s)
    if version is None:
        version = b""
    return (interpret_simpletext(application, context),
            interpret_simpletext(version, context))

def serialise_AP(value, context):
    """Serialise an AP (application) property value.

    value -- pair (application, version)
      application -- string
      version     -- string

    Note this takes a single parameter (which is a pair).

    """
    application, version = value
    return sgf_grammar.compose(serialise_simpletext(application, context),
                               serialise_simpletext(version, context))


def interpret_ARLN_list(values, context):
    """Interpret an AR (arrow) or LN (line) property value.

    Returns a list of pairs (point, point), where point is a pair (row, col)

    """
    result = []
    for s in values:
        p1, p2 = sgf_grammar.parse_compose(s)
        result.append((interpret_point(p1, context),
                       interpret_point(p2, context)))
    return result

def serialise_ARLN_list(values, context):
    """Serialise an AR (arrow) or LN (line) property value.

    values -- list of pairs (point, point), where point is a pair (row, col)

    """
    return [serialise_point(p1, context) + b":" + serialise_point(p2, context)
            for p1, p2 in values]


def interpret_FG(bb, context):
    """Interpret an FG (figure) property value.

    Returns a pair (flags, string), or None.

    flags is an integer; see http://www.red-bean.com/sgf/properties.html#FG

    """
    if bb == b"":
        return None
    flags, name = sgf_grammar.parse_compose(bb)
    return int(flags), interpret_simpletext(name, context)

def serialise_FG(value, context):
    """Serialise an FG (figure) property value.

    value -- pair (flags, name), or None
      flags -- int
      name  -- string

    Use serialise_FG(None) to produce an empty value.

    """
    if value is None:
        return b""
    flags, name = value
    return serialise_number(flags) + b":" + serialise_simpletext(name, context)


def interpret_LB_list(values, context):
    """Interpret an LB (label) property value.

    Returns a list of pairs ((row, col), string).

    """
    result = []
    for bb in values:
        point, label = sgf_grammar.parse_compose(bb)
        result.append((interpret_point(point, context),
                       interpret_simpletext(label, context)))
    return result

def serialise_LB_list(values, context):
    """Serialise an LB (label) property value.

    values -- list of pairs ((row, col), string)

    """
    return [serialise_point(point, context) + b":" +
            serialise_simpletext(text, context)
            for point, text in values]


class Property_type:
    """Description of a property type."""
    def __init__(self, interpreter, serialiser, uses_list,
                 allows_empty_list=False):
        self.interpreter = interpreter
        self.serialiser = serialiser
        self.uses_list = bool(uses_list)
        self.allows_empty_list = bool(allows_empty_list)

def _make_property_type(type_name, allows_empty_list=False):
    return Property_type(
        globals()["interpret_" + type_name],
        globals()["serialise_" + type_name],
        uses_list=(type_name.endswith("_list")),
        allows_empty_list=allows_empty_list)

_property_types_by_name = {
    'none' :        _make_property_type('none'),
    'number' :      _make_property_type('number'),
    'real' :        _make_property_type('real'),
    'double' :      _make_property_type('double'),
    'colour' :      _make_property_type('colour'),
    'simpletext' :  _make_property_type('simpletext'),
    'text' :        _make_property_type('text'),
    'point' :       _make_property_type('point'),
    'move' :        _make_property_type('move'),
    'point_list' :  _make_property_type('point_list'),
    'point_elist' : _make_property_type('point_list', allows_empty_list=True),
    'stone_list' :  _make_property_type('point_list'),
    'AP' :          _make_property_type('AP'),
    'ARLN_list' :   _make_property_type('ARLN_list'),
    'FG' :          _make_property_type('FG'),
    'LB_list' :     _make_property_type('LB_list'),
}

P = _property_types_by_name

_property_types_by_ident = {
  'AB' : P['stone_list'],                 # setup         Add Black
  'AE' : P['point_list'],                 # setup         Add Empty
  'AN' : P['simpletext'],                 # game-info     Annotation
  'AP' : P['AP'],                         # root          Application
  'AR' : P['ARLN_list'],                  # -             Arrow
  'AW' : P['stone_list'],                 # setup         Add White
  'B'  : P['move'],                       # move          Black
  'BL' : P['real'],                       # move          Black time left
  'BM' : P['double'],                     # move          Bad move
  'BR' : P['simpletext'],                 # game-info     Black rank
  'BT' : P['simpletext'],                 # game-info     Black team
  'C'  : P['text'],                       # -             Comment
  'CA' : P['simpletext'],                 # root          Charset
  'CP' : P['simpletext'],                 # game-info     Copyright
  'CR' : P['point_list'],                 # -             Circle
  'DD' : P['point_elist'],                # - [inherit]   Dim points
  'DM' : P['double'],                     # -             Even position
  'DO' : P['none'],                       # move          Doubtful
  'DT' : P['simpletext'],                 # game-info     Date
  'EV' : P['simpletext'],                 # game-info     Event
  'FF' : P['number'],                     # root          Fileformat
  'FG' : P['FG'],                         # -             Figure
  'GB' : P['double'],                     # -             Good for Black
  'GC' : P['text'],                       # game-info     Game comment
  'GM' : P['number'],                     # root          Game
  'GN' : P['simpletext'],                 # game-info     Game name
  'GW' : P['double'],                     # -             Good for White
  'HA' : P['number'],                     # game-info     Handicap
  'HO' : P['double'],                     # -             Hotspot
  'IT' : P['none'],                       # move          Interesting
  'KM' : P['real'],                       # game-info     Komi
  'KO' : P['none'],                       # move          Ko
  'LB' : P['LB_list'],                    # -             Label
  'LN' : P['ARLN_list'],                  # -             Line
  'MA' : P['point_list'],                 # -             Mark
  'MN' : P['number'],                     # move          set move number
  'N'  : P['simpletext'],                 # -             Nodename
  'OB' : P['number'],                     # move          OtStones Black
  'ON' : P['simpletext'],                 # game-info     Opening
  'OT' : P['simpletext'],                 # game-info     Overtime
  'OW' : P['number'],                     # move          OtStones White
  'PB' : P['simpletext'],                 # game-info     Player Black
  'PC' : P['simpletext'],                 # game-info     Place
  'PL' : P['colour'],                     # setup         Player to play
  'PM' : P['number'],                     # - [inherit]   Print move mode
  'PW' : P['simpletext'],                 # game-info     Player White
  'RE' : P['simpletext'],                 # game-info     Result
  'RO' : P['simpletext'],                 # game-info     Round
  'RU' : P['simpletext'],                 # game-info     Rules
  'SL' : P['point_list'],                 # -             Selected
  'SO' : P['simpletext'],                 # game-info     Source
  'SQ' : P['point_list'],                 # -             Square
  'ST' : P['number'],                     # root          Style
  'SZ' : P['number'],                     # root          Size
  'TB' : P['point_elist'],                # -             Territory Black
  'TE' : P['double'],                     # move          Tesuji
  'TM' : P['real'],                       # game-info     Timelimit
  'TR' : P['point_list'],                 # -             Triangle
  'TW' : P['point_elist'],                # -             Territory White
  'UC' : P['double'],                     # -             Unclear pos
  'US' : P['simpletext'],                 # game-info     User
  'V'  : P['real'],                       # -             Value
  'VW' : P['point_elist'],                # - [inherit]   View
  'W'  : P['move'],                       # move          White
  'WL' : P['real'],                       # move          White time left
  'WR' : P['simpletext'],                 # game-info     White rank
  'WT' : P['simpletext'],                 # game-info     White team
}
_text_property_type = P['text']

del P


class Presenter(_Context):
    """Convert property values between Python and SGF-string representations.

    Instantiate with:
      size     -- board size (int)
      encoding -- encoding for the SGF strings

    Public attributes (treat as read-only):
      size     -- int
      encoding -- string (normalised form)

    See the _property_types_by_ident table above for a list of properties
    initially known, and their types.

    Initially, treats unknown (private) properties as if they had type Text.

    """

    def __init__(self, size, encoding):
        try:
            encoding.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("encoding names must be ascii")
        try:
            encoding = normalise_charset_name(encoding)
        except LookupError:
            raise ValueError("unknown encoding: %s" % encoding)
        super().__init__(size, encoding)
        self.property_types_by_ident = _property_types_by_ident.copy()
        self.default_property_type = _text_property_type

    def get_property_type(self, identifier):
        """Return the Property_type for the specified PropIdent.

        Rasies KeyError if the property is unknown.

        """
        return self.property_types_by_ident[identifier]

    def register_property(self, identifier, property_type):
        """Specify the Property_type for a PropIdent."""
        self.property_types_by_ident[identifier] = property_type

    def deregister_property(self, identifier):
        """Forget the type for the specified PropIdent."""
        del self.property_types_by_ident[identifier]

    def set_private_property_type(self, property_type):
        """Specify the Property_type to use for unknown properties.

        Pass property_type = None to make unknown properties raise an error.

        """
        self.default_property_type = property_type

    def _get_effective_property_type(self, identifier):
        try:
            return self.property_types_by_ident[identifier]
        except KeyError:
            result = self.default_property_type
            if result is None:
                raise ValueError("unknown property")
            return result

    def interpret_as_type(self, property_type, raw_values):
        """Variant of interpret() for explicitly specified type.

        property_type -- Property_type

        """
        if not raw_values:
            raise ValueError("no raw values")
        if property_type.uses_list:
            if raw_values == [b""]:
                raw = []
            else:
                raw = raw_values
        else:
            if len(raw_values) > 1:
                raise ValueError("multiple values")
            raw = raw_values[0]
        return property_type.interpreter(raw, self)

    def interpret(self, identifier, raw_values):
        """Return a Python representation of a property value.

        identifier -- PropIdent
        raw_values -- nonempty list of 8-bit strings in the presenter's encoding

        See the interpret_... functions above for details of how values are
        represented as Python types.

        Raises ValueError if it cannot interpret the value.

        Note that in some cases the interpret_... functions accept values which
        are not strictly permitted by the specification.

        elist handling: if the property's value type is a list type and
        'raw_values' is a list containing a single empty string, passes an
        empty list to the interpret_... function (that is, this function treats
        all lists like elists).

        Doesn't enforce range restrictions on values with type Number.

        """
        return self.interpret_as_type(
            self._get_effective_property_type(identifier), raw_values)

    def serialise_as_type(self, property_type, value):
        """Variant of serialise() for explicitly specified type.

        property_type -- Property_type

        """
        serialised = property_type.serialiser(value, self)
        if property_type.uses_list:
            if serialised == []:
                if property_type.allows_empty_list:
                    return [b""]
                else:
                    raise ValueError("empty list")
            return serialised
        else:
            return [serialised]

    def serialise(self, identifier, value):
        """Serialise a Python representation of a property value.

        identifier -- PropIdent
        value      -- corresponding Python value

        Returns a nonempty list of 8-bit strings in the presenter's encoding,
        suitable for use as raw PropValues.

        See the serialise_... functions above for details of the acceptable
        values for each type.

        elist handling: if the property's value type is an elist type and the
        serialise_... function returns an empty list, this returns a list
        containing a single empty string.

        Raises ValueError if it cannot serialise the value.

        In general, the serialise_... functions try not to produce an invalid
        result, but do not try to prevent garbage input happening to produce a
        valid result.

        """
        return self.serialise_as_type(
            self._get_effective_property_type(identifier), value)

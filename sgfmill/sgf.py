"""Represent SGF games.

This is intended for use with SGF FF[4]; see http://www.red-bean.com/sgf/

"""

import datetime

from . import sgf_grammar
from . import sgf_properties


class Node:
    """An SGF node.

    Instantiate with a raw property map (see sgf_grammar) and an
    sgf_properties.Presenter.

    A Node doesn't belong to a particular game (cf Tree_node below), but it
    knows its board size (in order to interpret move values) and the encoding
    to use for the raw property strings.

    Changing the SZ property isn't allowed.

    """
    def __init__(self, property_map, presenter):
        # Map identifier (PropIdent) -> nonempty list of raw values
        self._property_map = property_map
        self._presenter = presenter

    def get_size(self):
        """Return the board size used to interpret property values."""
        return self._presenter.size

    def get_encoding(self):
        """Return the encoding used for raw property values.

        Returns a string (a valid Python codec name, eg "UTF-8").

        """
        return self._presenter.encoding

    def get_presenter(self):
        """Return the node's sgf_properties.Presenter."""
        return self._presenter

    def has_property(self, identifier):
        """Check whether the node has the specified property."""
        return identifier in self._property_map

    def properties(self):
        """Find the properties defined for the node.

        Returns a list of property identifiers (strings), in unspecified order.

        """
        return list(self._property_map.keys())

    def get_raw_list(self, identifier):
        """Return the raw values of the specified property.

        Returns a nonempty list of bytes objects, in the raw property encoding.

        The objects contain the exact bytes that go between the square brackets
        (without interpreting escapes or performing any whitespace conversion).

        Raises KeyError if there was no property with the given identifier.

        (If the property is an empty elist, this returns a list containing a
        single empty bytes object.)

        """
        return self._property_map[identifier]

    def get_raw(self, identifier):
        """Return a single raw value of the specified property.

        Returns a bytes object, in the raw property encoding.

        The object contains the exact bytes that go between the square brackets
        (without interpreting escapes or performing any whitespace conversion).

        Raises KeyError if there was no property with the given identifier.

        If the property has multiple values, this returns the first (if the
        value is an empty elist, this returns an empty bytes object).

        """
        return self._property_map[identifier][0]

    def get_raw_property_map(self):
        """Return the raw values of all properties as a dict.

        Returns a dict mapping property identifiers to lists of raw values
        (see get_raw_list()).

        Returns the same dict each time it's called.

        Treat the returned dict as read-only.

        """
        return self._property_map


    def _set_raw_list(self, identifier, values):
        if (identifier == "SZ" and
            values != [sgf_properties.serialise_number(self._presenter.size)]):
            raise ValueError("changing size is not permitted")
        self._property_map[identifier] = values

    def unset(self, identifier):
        """Remove the specified property.

        Raises KeyError if the property isn't currently present.

        """
        if identifier == "SZ" and self._presenter.size != 19:
            raise ValueError("changing size is not permitted")
        del self._property_map[identifier]


    def set_raw_list(self, identifier, values):
        """Set the raw values of the specified property.

        identifier -- string passing is_valid_property_identifier()
        values     -- nonempty iterable of bytes objects in the raw property
                      encoding

        The values specify the exact bytes to appear between the square
        brackets in the SGF file; you must perform any necessary escaping
        first.

        (To specify an empty elist, pass a list containing a single empty
        bytes object.)

        """
        if not sgf_grammar.is_valid_property_identifier(identifier):
            raise ValueError("ill-formed property identifier")
        values = list(values)
        if not values:
            raise ValueError("empty property list")
        for value in values:
            if not sgf_grammar.is_valid_property_value(value):
                raise ValueError("ill-formed raw property value")
        self._set_raw_list(identifier, values)

    def set_raw(self, identifier, value):
        """Set the specified property to a single raw value.

        identifier -- string passing is_valid_property_identifier()
        value      -- bytes object in the raw property encoding

        The value specifies the exact bytes to appear between the square
        brackets in the SGF file; you must perform any necessary escaping
        first.

        """
        if not sgf_grammar.is_valid_property_identifier(identifier):
            raise ValueError("ill-formed property identifier")
        if not sgf_grammar.is_valid_property_value(value):
            raise ValueError("ill-formed raw property value")
        self._set_raw_list(identifier, [value])


    def get(self, identifier):
        """Return the interpreted value of the specified property.

        Returns the value as a suitable Python representation.

        Raises KeyError if the node does not have a property with the given
        identifier.

        Raises ValueError if it cannot interpret the value.

        See sgf_properties.Presenter.interpret() for details.

        """
        return self._presenter.interpret(
            identifier, self._property_map[identifier])

    def set(self, identifier, value):
        """Set the value of the specified property.

        identifier -- string passing is_valid_property_identifier()
        value      -- new property value (in its Python representation)

        For properties with value type 'none', use value True.

        Raises ValueError if it cannot represent the value.

        See sgf_properties.Presenter.serialise() for details.

        """
        self._set_raw_list(
            identifier, self._presenter.serialise(identifier, value))

    def get_raw_move(self):
        """Return the raw value of the move from a node.

        Returns a pair (colour, raw value)

        colour is 'b' or 'w'.

        Returns None, None if the node contains no B or W property.

        """
        values = self._property_map.get("B")
        if values is not None:
            colour = "b"
        else:
            values = self._property_map.get("W")
            if values is not None:
                colour = "w"
            else:
                return None, None
        return colour, values[0]

    def get_move(self):
        """Retrieve the move from a node.

        Returns a pair (colour, move)

        colour is 'b' or 'w'.

        move is (row, col), or None for a pass.

        Returns None, None if the node contains no B or W property.

        """
        colour, raw = self.get_raw_move()
        if colour is None:
            return None, None
        return (colour,
                sgf_properties.interpret_go_point(raw, self._presenter.size))

    def get_setup_stones(self):
        """Retrieve Add Black / Add White / Add Empty properties from a node.

        Returns a tuple (black_points, white_points, empty_points)

        Each value is a set of pairs (row, col).

        """
        try:
            bp = self.get("AB")
        except KeyError:
            bp = set()
        try:
            wp = self.get("AW")
        except KeyError:
            wp = set()
        try:
            ep = self.get("AE")
        except KeyError:
            ep = set()
        return bp, wp, ep

    def has_setup_stones(self):
        """Check whether the node has any AB/AW/AE properties."""
        d = self._property_map
        return ("AB" in d or "AW" in d or "AE" in d)

    def set_move(self, colour, move):
        """Set the B or W property.

        colour -- 'b' or 'w'.
        move -- (row, col), or None for a pass.

        Replaces any existing B or W property in the node.

        """
        if colour not in ('b', 'w'):
            raise ValueError
        if 'B' in self._property_map:
            del self._property_map['B']
        if 'W' in self._property_map:
            del self._property_map['W']
        self.set(colour.upper(), move)

    def set_setup_stones(self, black, white, empty=None):
        """Set Add Black / Add White / Add Empty properties.

        black, white, empty -- list or set of pairs (row, col)

        Removes any existing AB/AW/AE properties from the node.

        """
        if 'AB' in self._property_map:
            del self._property_map['AB']
        if 'AW' in self._property_map:
            del self._property_map['AW']
        if 'AE' in self._property_map:
            del self._property_map['AE']
        if black:
            self.set('AB', black)
        if white:
            self.set('AW', white)
        if empty:
            self.set('AE', empty)

    def add_comment_text(self, text):
        """Add or extend the node's comment.

        text -- string

        If the node doesn't have a C property, adds one with the specified
        text.

        Otherwise, adds the specified text to the existing C property value
        (with two newlines in front).

        """
        if self.has_property('C'):
            self.set('C', self.get('C') + "\n\n" + text)
        else:
            self.set('C', text)

    def __str__(self):
        """String description of the node, for debugging."""
        def fmt(bb):
            return bb.decode(self.get_encoding(), "replace")
        def format_property(ident, values):
            return ident + "".join("[%s]" % fmt(bb) for bb in values)
        return "\n".join(
            format_property(ident, values)
            for (ident, values) in sorted(self._property_map.items())) \
            + "\n"


class Tree_node(Node):
    """A node embedded in an SGF game.

    A Tree_node is a Node that also knows its position within an Sgf_game.

    Do not instantiate directly; retrieve from an Sgf_game or another Tree_node.

    A Tree_node is a list-like container of its children: it can be indexed,
    sliced, and iterated over like a list, and supports index().

    A Tree_node with no children is treated as having truth value false.

    Public attributes (treat as read-only):
      owner  -- the node's Sgf_game
      parent -- the nodes's parent Tree_node (None for the root node)

    """
    def __init__(self, parent, properties):
        self.owner = parent.owner
        self.parent = parent
        self._children = []
        Node.__init__(self, properties, parent._presenter)

    def _add_child(self, node):
        self._children.append(node)

    def __len__(self):
        return len(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def index(self, child):
        return self._children.index(child)

    def new_child(self, index=None):
        """Create a new Tree_node and add it as this node's last child.

        If 'index' is specified, the new node is inserted in the child list at
        the specified index instead (behaves like list.insert).

        Returns the new node.

        """
        child = Tree_node(self, {})
        if index is None:
            self._children.append(child)
        else:
            self._children.insert(index, child)
        return child

    def delete(self):
        """Remove this node from its parent."""
        if self.parent is None:
            raise ValueError("can't remove the root node")
        self.parent._children.remove(self)

    def reparent(self, new_parent, index=None):
        """Move this node to a new place in the tree.

        new_parent -- Tree_node from the same game.

        Raises ValueError if the new parent is this node or one of its
        descendants.

        If 'index' is specified, the node is inserted in the new parent's child
        list at the specified index (behaves like list.insert); otherwise it's
        placed at the end.

        """
        if new_parent.owner != self.owner:
            raise ValueError("new parent doesn't belong to the same game")
        n = new_parent
        while True:
            if n == self:
                raise ValueError("would create a loop")
            n = n.parent
            if n is None:
                break
        # self.parent is not None because moving the root would create a loop.
        self.parent._children.remove(self)
        self.parent = new_parent
        if index is None:
            new_parent._children.append(self)
        else:
            new_parent._children.insert(index, self)

    def find(self, identifier):
        """Find the nearest ancestor-or-self containing the specified property.

        Returns a Tree_node, or None if there is no such node.

        """
        node = self
        while node is not None:
            if node.has_property(identifier):
                return node
            node = node.parent
        return None

    def find_property(self, identifier):
        """Return the value of a property, defined at this node or an ancestor.

        This is intended for use with properties of type 'game-info', and with
        properties with the 'inherit' attribute.

        This returns the interpreted value, in the same way as get().

        It searches up the tree, in the same way as find().

        Raises KeyError if no node defining the property is found.

        """
        node = self.find(identifier)
        if node is None:
            raise KeyError
        return node.get(identifier)

class _Root_tree_node(Tree_node):
    """Variant of Tree_node used for a game root."""
    def __init__(self, property_map, owner):
        self.owner = owner
        self.parent = None
        self._children = []
        Node.__init__(self, property_map, owner.presenter)

class _Unexpanded_root_tree_node(_Root_tree_node):
    """Variant of _Root_tree_node used with 'loaded' Sgf_games."""
    def __init__(self, owner, coarse_tree):
        _Root_tree_node.__init__(self, coarse_tree.sequence[0], owner)
        self._coarse_tree = coarse_tree

    def _expand(self):
        sgf_grammar.make_tree(
            self._coarse_tree, self, Tree_node, Tree_node._add_child)
        delattr(self, '_coarse_tree')
        self.__class__ = _Root_tree_node

    def __len__(self):
        self._expand()
        return self.__len__()

    def __getitem__(self, key):
        self._expand()
        return self.__getitem__(key)

    def index(self, child):
        self._expand()
        return self.index(child)

    def new_child(self, index=None):
        self._expand()
        return self.new_child(index)

    def _main_sequence_iter(self):
        presenter = self._presenter
        for properties in sgf_grammar.main_sequence_iter(self._coarse_tree):
            yield Node(properties, presenter)


class Sgf_game:
    """An SGF game tree.

    The complete game tree is represented using Tree_nodes. The various methods
    which return Tree_nodes will always return the same object for the same
    node.

    Instantiate with
      size     -- int (board size), in range 1 to 26
      encoding -- the raw property encoding (default "UTF-8")

    'encoding' must be a valid Python codec name.

    The following root node properties are set for newly-created games:
      FF[4]
      GM[1]
      SZ[size]
      CA[encoding]

    Changing FF and GM is permitted (but this library will carry on using the
    FF[4] and GM[1] rules). Changing SZ is not permitted (unless the change
    leaves the effective value unchanged). Changing CA is permitted; this
    controls the encoding used by serialise().

    """
    def __new__(cls, size, encoding="UTF-8", *args, **kwargs):
        # To complete initialisation after this, you need to set 'root'.
        if not 1 <= size <= 26:
            raise ValueError("size out of range: %s" % size)
        game = super(Sgf_game, cls).__new__(cls)
        game.size = size
        game.presenter = sgf_properties.Presenter(size, encoding)
        return game

    def __init__(self, *args, **kwargs):
        self.root = _Root_tree_node({}, self)
        self.root.set_raw('FF', b"4")
        self.root.set_raw('GM', b"1")
        self.root.set_raw('SZ', sgf_properties.serialise_number(self.size))
        # Read the encoding back so we get the normalised form
        self.root.set_raw('CA', self.presenter.encoding.encode())

    @classmethod
    def from_coarse_game_tree(cls, coarse_game, override_encoding=None):
        """Alternative constructor: create an Sgf_game from the parser output.

        coarse_game       -- Coarse_game_tree
        override_encoding -- encoding name, eg "UTF-8" (optional)

        The nodes' property maps (as returned by get_raw_property_map()) will
        be the same dictionary objects as the ones from the Coarse_game_tree.

        The board size and raw property encoding are taken from the SZ and CA
        properties in the root node (defaulting to 19 and "ISO-8859-1",
        respectively).

        If override_encoding is specified, the source data is assumed to be in
        the specified encoding (no matter what the CA property says), and the
        CA property is set to match.

        """
        def _get_raw(identifier):
            return coarse_game.sequence[0][identifier][0]
        try:
            size_bb = _get_raw('SZ')
        except KeyError:
            size = 19
        else:
            try:
                size = int(size_bb)
            except ValueError:
                raise ValueError("bad SZ property: %s" % size_bb)
        if override_encoding is None:
            try:
                encoding = _get_raw('CA').decode("ascii", "replace")
            except KeyError:
                encoding = "ISO-8859-1"
        else:
            encoding = override_encoding
        game = cls.__new__(cls, size, encoding)
        game.root = _Unexpanded_root_tree_node(game, coarse_game)
        if override_encoding is not None:
            game.root.set_raw("CA", game.presenter.encoding.encode())
        return game

    @classmethod
    def from_bytes(cls, bb, override_encoding=None):
        """Alternative constructor: read a single Sgf_game from a bytestring.

        bb                -- bytes-like object
        override_encoding -- encoding name, eg "UTF-8" (optional)

        Raises ValueError if it can't parse the data. See parse_sgf_game()
        for details.

        Assumes the data is in the encoding described by the CA property in the
        root node (defaulting to "ISO-8859-1"), and uses that as the raw
        property encoding.

        But if override_encoding is specified, assumes the data is in that
        encoding (no matter what the CA property says), and sets the CA
        property and raw property encoding to match.

        The board size is taken from the SZ property in the root node
        (defaulting to 19).

        """
        coarse_game = sgf_grammar.parse_sgf_game(bb)
        return cls.from_coarse_game_tree(coarse_game, override_encoding)

    @classmethod
    def from_string(cls, s):
        """Alternative constructor: read a single Sgf_game from a string.

        s -- string

        The game's raw property encoding and CA property will be UTF-8
        (replacing any CA property present in the string).

        Raises ValueError if it can't parse the data. See parse_sgf_game()
        for details.

        The board size is taken from the SZ property in the root node
        (defaulting to 19).

        """
        if not hasattr(s, 'encode'):
            raise TypeError("expected string, given %s" % type(s).__name__)
        return cls.from_bytes(s.encode("utf-8"), override_encoding="utf-8")

    def serialise(self, wrap=79):
        """Serialise the SGF data as bytes.

        wrap -- int (default 79), or None

        Returns a bytes object, in the encoding specified by the CA property in
        the root node (defaulting to "ISO-8859-1").

        If the raw property encoding and the target encoding match (which is
        the usual case), the raw property values are included unchanged in the
        output (even if they are improperly encoded.)

        Otherwise, if any raw property value is improperly encoded,
        UnicodeDecodeError is raised, and if any property value can't be
        represented in the target encoding, UnicodeEncodeError is raised.

        If the target encoding doesn't identify a Python codec, ValueError is
        raised. Behaviour is unspecified if the target encoding isn't
        ASCII-compatible (eg, UTF-16).

        If 'wrap' is not None, makes some effort to keep output lines no longer
        than 'wrap'.

        """
        try:
            encoding = self.get_charset()
        except ValueError:
            raise ValueError("unsupported charset: %s" %
                             self.root.get_raw("CA"))
        coarse_tree = sgf_grammar.make_coarse_game_tree(
            self.root, lambda node:node, Node.get_raw_property_map)
        serialised = sgf_grammar.serialise_game_tree(coarse_tree, wrap)
        if encoding == self.root.get_encoding():
            return serialised
        else:
            return serialised.decode(self.root.get_encoding()).encode(encoding)


    def get_property_presenter(self):
        """Return the property presenter.

        Returns an sgf_properties.Presenter.

        This can be used to customise how property values are interpreted and
        serialised.

        """
        return self.presenter

    def get_root(self):
        """Return the root node (as a Tree_node)."""
        return self.root

    def get_last_node(self):
        """Return the last node in the 'leftmost' variation (as a Tree_node)."""
        node = self.root
        while node:
            node = node[0]
        return node

    def get_main_sequence(self):
        """Return the 'leftmost' variation.

        Returns a list of Tree_nodes, from the root to a leaf.

        """
        node = self.root
        result = [node]
        while node:
            node = node[0]
            result.append(node)
        return result

    def get_main_sequence_below(self, node):
        """Return the 'leftmost' variation below the specified node.

        node -- Tree_node

        Returns a list of Tree_nodes, from the first child of 'node' to a leaf.

        """
        if node.owner is not self:
            raise ValueError("node doesn't belong to this game")
        result = []
        while node:
            node = node[0]
            result.append(node)
        return result

    def get_sequence_above(self, node):
        """Return the partial variation leading to the specified node.

        node -- Tree_node

        Returns a list of Tree_nodes, from the root to the parent of 'node'.

        """
        if node.owner is not self:
            raise ValueError("node doesn't belong to this game")
        result = []
        while node.parent is not None:
            node = node.parent
            result.append(node)
        result.reverse()
        return result

    def main_sequence_iter(self):
        """Provide the 'leftmost' variation as an iterator.

        Returns an iterator providing Node instances, from the root to a leaf.

        The Node instances may or may not be Tree_nodes.

        It's OK to use these Node instances to modify properties: even if they
        are not the same objects as returned by the main tree navigation
        methods, they share the underlying property maps.

        If you know the game has no variations, or you're only interested in
        the 'leftmost' variation, you can use this function to retrieve the
        nodes without building the entire game tree.

        """
        if isinstance(self.root, _Unexpanded_root_tree_node):
            return self.root._main_sequence_iter()
        return iter(self.get_main_sequence())

    def extend_main_sequence(self):
        """Create a new Tree_node and add to the 'leftmost' variation.

        Returns the new node.

        """
        return self.get_last_node().new_child()

    def get_size(self):
        """Return the board size as an integer."""
        return self.size

    def get_charset(self):
        """Return the effective value of the CA root property.

        This applies the default, and returns the normalised form.

        Raises ValueError if the CA property doesn't identify a Python codec.

        """
        try:
            s = self.root.get("CA")
        except KeyError:
            return "ISO-8859-1"
        try:
            return sgf_properties.normalise_charset_name(s)
        except LookupError:
            raise ValueError("no codec available for CA %s" % s)

    def get_komi(self):
        """Return the komi as a float.

        Returns 0.0 if the KM property isn't present in the root node.

        Raises ValueError if the KM property is malformed.

        """
        try:
            return self.root.get("KM")
        except KeyError:
            return 0.0

    def get_handicap(self):
        """Return the number of handicap stones as a small integer.

        Returns None if the HA property isn't present, or has (illegal) value
        zero.

        Raises ValueError if the HA property is otherwise malformed.

        """
        try:
            handicap = self.root.get("HA")
        except KeyError:
            return None
        if handicap == 0:
            handicap = None
        elif handicap == 1:
            raise ValueError
        return handicap

    def get_player_name(self, colour):
        """Return the name of the specified player.

        Returns None if there is no corresponding 'PB' or 'PW' property.

        """
        try:
            return self.root.get({'b' : 'PB', 'w' : 'PW'}[colour])
        except KeyError:
            return None

    def get_winner(self):
        """Return the colour of the winning player.

        Returns None if there is no RE property, or if neither player won.

        """
        try:
            colour = self.root.get("RE")[0].lower()
        except LookupError:
            return None
        if colour not in ("b", "w"):
            return None
        return colour

    def set_date(self, date=None):
        """Set the DT property to a single date.

        date -- datetime.date (defaults to today)

        (SGF allows dates to be rather more complicated than this, so there's
         no corresponding get_date() method.)

        """
        if date is None:
            date = datetime.date.today()
        self.root.set('DT', date.strftime("%Y-%m-%d"))


import random
import numpy as np

#Implements legal moves without superko
class Board:
  EMPTY = 0
  BLACK = 1
  WHITE = 2
  WALL = 3

  ZOBRIST_STONE = [[],[],[],[]]
  ZOBRIST_PLA = []

  ZOBRIST_RAND = random.Random()
  ZOBRIST_RAND.seed(123987456)

  for i in range((19+1)*(19+2)+1):
    ZOBRIST_STONE[BLACK].append(ZOBRIST_RAND.getrandbits(64))
    ZOBRIST_STONE[WHITE].append(ZOBRIST_RAND.getrandbits(64))
  for i in range(4):
    ZOBRIST_PLA.append(ZOBRIST_RAND.getrandbits(64))

  def __init__(self,size,copy_other=None):
    if size < 2 or size > 39:
      raise ValueError("Invalid board size: " + str(size))
    self.size = size
    self.arrsize = (size+1)*(size+2)+1
    self.dy = size+1
    self.adj = [-self.dy,-1,1,self.dy]
    self.diag = [-self.dy-1,-self.dy+1,self.dy-1,self.dy+1]

    if copy_other is not None:
      self.pla = copy_other.pla
      self.board = np.copy(copy_other.board)
      self.group_head = np.copy(copy_other.group_head)
      self.group_stone_count = np.copy(copy_other.group_stone_count)
      self.group_liberty_count = np.copy(copy_other.group_liberty_count)
      self.group_next = np.copy(copy_other.group_next)
      self.group_prev = np.copy(copy_other.group_prev)
      self.zobrist = copy_other.zobrist
      self.simple_ko_point = copy_other.simple_ko_point
    else:
      self.pla = Board.BLACK
      self.board = np.zeros(shape=(self.arrsize), dtype=np.int8)
      self.group_head = np.zeros(shape=(self.arrsize), dtype=np.int16)
      self.group_stone_count = np.zeros(shape=(self.arrsize), dtype=np.int16)
      self.group_liberty_count = np.zeros(shape=(self.arrsize), dtype=np.int16)
      self.group_next = np.zeros(shape=(self.arrsize), dtype=np.int16)
      self.group_prev = np.zeros(shape=(self.arrsize), dtype=np.int16)
      self.zobrist = 0
      self.simple_ko_point = None

      for i in range(-1,size+1):
        self.board[self.loc(i,-1)] = Board.WALL
        self.board[self.loc(i,size)] = Board.WALL
        self.board[self.loc(-1,i)] = Board.WALL
        self.board[self.loc(size,i)] = Board.WALL

      #More-easily catch errors
      self.group_head[0] = -1
      self.group_next[0] = -1
      self.group_prev[0] = -1

  def copy():
    return Board(self,self.size,copy_other=self)

  @staticmethod
  def get_opp(pla):
    return 3-pla
  @staticmethod
  def loc_static(x,y,size):
    return (x+1) + (size+1)*(y+1)

  def loc(self,x,y):
    return (x+1) + self.dy*(y+1)
  def loc_x(self,loc):
    return (loc % self.dy)-1
  def loc_y(self,loc):
    return (loc // self.dy)-1


  def pos_zobrist(self):
    return self.zobrist
  def sit_zobrist(self):
    return self.zobrist ^ Board.ZOBRIST_PLA[self.pla]

  def num_liberties(self,loc):
    if self.board[loc] == Board.EMPTY or self.board[loc] == Board.WALL:
      return 0
    return self.group_liberty_count[self.group_head[loc]]

  def is_simple_eye(self,pla,loc):
    adj0 = loc + self.adj[0]
    adj1 = loc + self.adj[1]
    adj2 = loc + self.adj[2]
    adj3 = loc + self.adj[3]

    if (self.board[adj0] != pla and self.board[adj0] != Board.WALL) or \
       (self.board[adj1] != pla and self.board[adj1] != Board.WALL) or \
       (self.board[adj2] != pla and self.board[adj2] != Board.WALL) or \
       (self.board[adj3] != pla and self.board[adj3] != Board.WALL):
      return False

    opp = Board.get_opp(pla)
    opp_corners = 0
    diag0 = loc + self.diag[0]
    diag1 = loc + self.diag[1]
    diag2 = loc + self.diag[2]
    diag3 = loc + self.diag[3]
    if self.board[diag0] == opp:
      opp_corners += 1
    if self.board[diag1] == opp:
      opp_corners += 1
    if self.board[diag2] == opp:
      opp_corners += 1
    if self.board[diag3] == opp:
      opp_corners += 1

    if opp_corners >= 2:
      return False
    if opp_corners <= 0:
      return True

    against_wall = (
      self.board[adj0] == Board.WALL or \
      self.board[adj1] == Board.WALL or \
      self.board[adj2] == Board.WALL or \
      self.board[adj3] == Board.WALL
    )

    if against_wall:
      return False
    return True


  def would_be_legal(self,pla,loc):
    if pla != Board.BLACK and pla != Board.WHITE:
      return False
    if not self.is_on_board(loc):
      return False
    if self.board[loc] != Board.EMPTY:
      return False
    if self.would_be_suicide(pla,loc):
      return False
    if loc == self.simple_ko_point:
      return False
    return True

  def would_be_suicide(self,pla,loc):
    adj0 = loc + self.adj[0]
    adj1 = loc + self.adj[1]
    adj2 = loc + self.adj[2]
    adj3 = loc + self.adj[3]

    opp = Board.get_opp(pla)

    #If empty or capture, then not suicide
    if self.board[adj0] == Board.EMPTY or (self.board[adj0] == opp and self.group_liberty_count[self.group_head[adj0]] == 1) or \
       self.board[adj1] == Board.EMPTY or (self.board[adj1] == opp and self.group_liberty_count[self.group_head[adj1]] == 1) or \
       self.board[adj2] == Board.EMPTY or (self.board[adj2] == opp and self.group_liberty_count[self.group_head[adj2]] == 1) or \
       self.board[adj3] == Board.EMPTY or (self.board[adj3] == opp and self.group_liberty_count[self.group_head[adj3]] == 1):
      return False
    #If connects to own stone with enough liberties, then not suicide
    if self.board[adj0] == pla and self.group_liberty_count[self.group_head[adj0]] > 1 or \
       self.board[adj1] == pla and self.group_liberty_count[self.group_head[adj1]] > 1 or \
       self.board[adj2] == pla and self.group_liberty_count[self.group_head[adj2]] > 1 or \
       self.board[adj3] == pla and self.group_liberty_count[self.group_head[adj3]] > 1:
      return False
    return True

  #Returns the number of liberties a new stone placed here would have, or maxLibs if it would be >= maxLibs.
  def get_liberties_after_play(self,pla,loc,maxLibs):
    opp = Board.get_opp(pla)
    libs = []
    capturedGroupHeads = []

    #First, count immediate liberties and groups that would be captured
    for i in range(4):
      adj = loc + self.adj[i]
      if self.board[adj] == Board.EMPTY:
        libs.append(adj)
        if len(libs) >= maxLibs:
          return maxLibs

      elif self.board[adj] == opp and self.num_liberties(adj) == 1:
        libs.append(adj)
        if len(libs) >= maxLibs:
          return maxLibs

        head = self.group_head[adj]
        if head not in capturedGroupHeads:
          capturedGroupHeads.append(head)

    def wouldBeEmpty(possibleLib):
      if self.board[possibleLib] == Board.EMPTY:
        return true
      elif self.board[possibleLib] == opp:
        return (self.group_head[possibleLib] in capturedGroupHeads)
      return False

    #Next, walk through all stones of all surrounding groups we would connect with and count liberties, avoiding overlap.
    connectingGroupHeads = []
    for i in range(4):
      adj = loc + self.adj[i]
      if self.board[adj] == pla:
        head = self.group_head[adj]
        if head not in connectingGroupHeads:
          connectingGroupHeads.append(head)

          cur = adj
          while True:
            for k in range(4):
              possibleLib = cur + self.adj[k]
              if possibleLib != loc and wouldBeEmpty(possibleLib) and possibleLib not in libs:
                libs.append(possibleLib)
                if len(libs) >= maxLibs:
                  return maxLibs

            cur = self.group_next[cur]
            if cur == adj:
              break

    return len(libs)


  def to_string(self):
    def get_piece(x,y):
      loc = self.loc(x,y)
      if self.board[loc] == Board.BLACK:
        return 'X '
      elif self.board[loc] == Board.WHITE:
        return 'O '
      elif (x == 3 or x == self.size/2 or x == self.size-1-3) and (y == 3 or y == self.size/2 or y == self.size-1-3):
        return '* '
      else:
        return '. '

    return "\n".join("".join(get_piece(x,y) for x in range(self.size)) for y in range(self.size))

  def to_liberty_string(self):
    def get_piece(x,y):
      loc = self.loc(x,y)
      if self.board[loc] == Board.BLACK or self.board[loc] == Board.WHITE:
        libs = self.group_liberty_count[self.group_head[loc]]
        if libs <= 9:
          return str(libs) + ' '
        else:
          return '@ '
      elif (x == 3 or x == self.size/2 or x == self.size-1-3) and (y == 3 or y == self.size/2 or y == self.size-1-3):
        return '* '
      else:
        return '. '

    return "\n".join("".join(get_piece(x,y) for x in range(self.size)) for y in range(self.size))

  def set_pla(self,pla):
    self.pla = pla
  def is_on_board(self,loc):
    return loc >= 0 and loc < self.arrsize and self.board[loc] != Board.WALL

  #Set a given location with error checking. Suicide setting allowed.
  def set_stone(self,pla,loc):
    if pla != Board.EMPTY and pla != Board.BLACK and pla != Board.WHITE:
      raise ValueError("Invalid pla for board.set")
    if not self.is_on_board(loc):
      raise ValueError("Invalid loc for board.set")

    if self.board[loc] == pla:
      pass
    elif self.board[loc] == Board.EMPTY:
      self.add_unsafe(pla,loc)
    elif pla == Board.EMPTY:
      self.remove_single_stone_unsafe(loc)
    else:
      self.remove_single_stone_unsafe(loc)
      self.add_unsafe(pla,loc)

    #Clear any ko restrictions
    self.simple_ko_point = None


  #Play a stone at the given location, with non-superko legality checking and updating the pla and simple ko point
  def play(self,pla,loc):
    if pla != Board.BLACK and pla != Board.WHITE:
      raise ValueError("Invalid pla for board.play")
    if not self.is_on_board(loc):
      raise ValueError("Invalid loc for board.set")
    if self.board[loc] != Board.EMPTY:
      raise ValueError("Location is nonempty")
    if self.would_be_suicide(pla,loc):
      raise ValueError("Move would be illegal suicide")
    if loc == self.simple_ko_point:
      raise ValueError("Move would be illegal simple ko recapture")

    self.add_unsafe(pla,loc)
    self.pla = Board.get_opp(pla)

  def do_pass(self):
    self.pla = Board.get_opp(self.pla)
    self.simple_ko_point = None

  #Add a stone, assumes that the location is empty without checking
  def add_unsafe(self,pla,loc):
    opp = Board.get_opp(pla)

    #Put the stone down
    self.board[loc] = pla
    self.zobrist ^= Board.ZOBRIST_STONE[pla][loc]

    #Initialize the group for that stone
    self.group_head[loc] = loc
    self.group_stone_count[loc] = 1
    liberties = 0
    for dloc in self.adj:
      if self.board[loc+dloc] == Board.EMPTY:
        liberties += 1
    self.group_liberty_count[loc] = liberties
    self.group_next[loc] = loc
    self.group_prev[loc] = loc

    #Fill surrounding liberties of all adjacent groups
    #Carefully avoid doublecounting
    adj0 = loc + self.adj[0]
    adj1 = loc + self.adj[1]
    adj2 = loc + self.adj[2]
    adj3 = loc + self.adj[3]
    if self.board[adj0] == Board.BLACK or self.board[adj0] == Board.WHITE:
      self.group_liberty_count[self.group_head[adj0]] -= 1
    if self.board[adj1] == Board.BLACK or self.board[adj1] == Board.WHITE:
      if self.group_head[adj1] != self.group_head[adj0]:
        self.group_liberty_count[self.group_head[adj1]] -= 1
    if self.board[adj2] == Board.BLACK or self.board[adj2] == Board.WHITE:
      if self.group_head[adj2] != self.group_head[adj0] and \
         self.group_head[adj2] != self.group_head[adj1]:
        self.group_liberty_count[self.group_head[adj2]] -= 1
    if self.board[adj3] == Board.BLACK or self.board[adj3] == Board.WHITE:
      if self.group_head[adj3] != self.group_head[adj0] and \
         self.group_head[adj3] != self.group_head[adj1] and \
         self.group_head[adj3] != self.group_head[adj2]:
        self.group_liberty_count[self.group_head[adj3]] -= 1

    #Merge groups
    if self.board[adj0] == pla:
      self.merge_unsafe(loc,adj0)
    if self.board[adj1] == pla:
      self.merge_unsafe(loc,adj1)
    if self.board[adj2] == pla:
      self.merge_unsafe(loc,adj2)
    if self.board[adj3] == pla:
      self.merge_unsafe(loc,adj3)

    #Resolve captures
    opp_stones_captured = 0
    caploc = 0
    if self.board[adj0] == opp and self.group_liberty_count[self.group_head[adj0]] == 0:
      opp_stones_captured += self.group_stone_count[self.group_head[adj0]]
      caploc = adj0
      self.remove_unsafe(adj0)
    if self.board[adj1] == opp and self.group_liberty_count[self.group_head[adj1]] == 0:
      opp_stones_captured += self.group_stone_count[self.group_head[adj1]]
      caploc = adj1
      self.remove_unsafe(adj1)
    if self.board[adj2] == opp and self.group_liberty_count[self.group_head[adj2]] == 0:
      opp_stones_captured += self.group_stone_count[self.group_head[adj2]]
      caploc = adj2
      self.remove_unsafe(adj2)
    if self.board[adj3] == opp and self.group_liberty_count[self.group_head[adj3]] == 0:
      opp_stones_captured += self.group_stone_count[self.group_head[adj3]]
      caploc = adj3
      self.remove_unsafe(adj3)

    if self.group_liberty_count[self.group_head[loc]] == 0:
      self.remove_unsafe(loc)

    #Update ko point for legality checking
    if opp_stones_captured == 1 and \
       self.group_stone_count[self.group_head[loc]] == 1 and \
       self.group_liberty_count[self.group_head[loc]] == 1:
      self.simple_ko_point = caploc
    else:
      self.simple_ko_point = None


  def is_head_adjacent(self,head,loc):
    return (
      self.group_head[loc+self.adj[0]] == head or \
      self.group_head[loc+self.adj[1]] == head or \
      self.group_head[loc+self.adj[2]] == head or \
      self.group_head[loc+self.adj[3]] == head
    )

  #Helper, merge two groups assuming they're owned by the same player and adjacent
  def merge_unsafe(self,loc0,loc1):
    if self.group_stone_count[self.group_head[loc0]] >= self.group_stone_count[self.group_head[loc1]]:
      parent = loc0
      child = loc1
    else:
      child = loc0
      parent = loc1

    phead = self.group_head[parent]
    chead = self.group_head[child]
    if phead == chead:
      return

    #Walk the child group assigning the new head and simultaneously counting liberties
    new_stone_count = self.group_stone_count[phead] + self.group_stone_count[chead]
    new_liberties = self.group_liberty_count[phead]
    loc = child
    while True:
      adj0 = loc + self.adj[0]
      adj1 = loc + self.adj[1]
      adj2 = loc + self.adj[2]
      adj3 = loc + self.adj[3]

      #Any adjacent empty space is a new liberty as long as it isn't adjacent to the parent head
      if self.board[adj0] == Board.EMPTY and not self.is_head_adjacent(phead,adj0):
        new_liberties += 1
      if self.board[adj1] == Board.EMPTY and not self.is_head_adjacent(phead,adj1):
        new_liberties += 1
      if self.board[adj2] == Board.EMPTY and not self.is_head_adjacent(phead,adj2):
        new_liberties += 1
      if self.board[adj3] == Board.EMPTY and not self.is_head_adjacent(phead,adj3):
        new_liberties += 1

      #Now assign the new parent head to take over the child (this also
      #prevents double-counting liberties)
      self.group_head[loc] = phead

      #Advance around the linked list
      loc = self.group_next[loc]
      if loc == child:
        break

    #Zero out the old head
    self.group_stone_count[chead] = 0
    self.group_liberty_count[chead] = 0

    #Update the new head
    self.group_stone_count[phead] = new_stone_count
    self.group_liberty_count[phead] = new_liberties

    #Combine the linked lists
    plast = self.group_prev[phead]
    clast = self.group_prev[chead]
    self.group_next[clast] = phead
    self.group_next[plast] = chead
    self.group_prev[chead] = plast
    self.group_prev[phead] = clast

  #Remove all stones in a group
  def remove_unsafe(self,group):
    head = self.group_head[group]
    pla = self.board[group]
    opp = Board.get_opp(pla)

    #Walk all the stones in the group and delete them
    loc = group
    while True:
      #Add a liberty to all surrounding opposing groups, taking care to avoid double counting
      adj0 = loc + self.adj[0]
      adj1 = loc + self.adj[1]
      adj2 = loc + self.adj[2]
      adj3 = loc + self.adj[3]
      if self.board[adj0] == opp:
        self.group_liberty_count[self.group_head[adj0]] += 1
      if self.board[adj1] == opp:
        if self.group_head[adj1] != self.group_head[adj0]:
          self.group_liberty_count[self.group_head[adj1]] += 1
      if self.board[adj2] == opp:
        if self.group_head[adj2] != self.group_head[adj0] and \
           self.group_head[adj2] != self.group_head[adj1]:
          self.group_liberty_count[self.group_head[adj2]] += 1
      if self.board[adj3] == opp:
        if self.group_head[adj3] != self.group_head[adj0] and \
           self.group_head[adj3] != self.group_head[adj1] and \
           self.group_head[adj3] != self.group_head[adj2]:
          self.group_liberty_count[self.group_head[adj3]] += 1

      next_loc = self.group_next[loc]

      #Zero out all the stuff
      self.board[loc] = Board.EMPTY
      self.zobrist ^= Board.ZOBRIST_STONE[opp][loc]
      self.group_head[loc] = 0
      self.group_next[loc] = 0
      self.group_prev[loc] = 0

      #Advance around the linked list
      loc = next_loc
      if loc == group:
        break

    #Zero out the head
    self.group_stone_count[head] = 0
    self.group_liberty_count[head] = 0

  #Remove a single stone
  def remove_single_stone_unsafe(self,rloc):
    pla = self.board[rloc]

    #Record all the stones in the group
    stones = []
    loc = rloc
    while True:
      stones.append(loc)
      loc = self.group_next[loc]
      if loc == rloc:
        break

    #Remove them all
    self.remove_unsafe(rloc)

    #Then add them back one by one
    for loc in stones:
      if loc != rloc:
        self.add_unsafe(pla,loc)


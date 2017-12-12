/*
 * fastboard.cpp
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modificationss).
 */

#include <cassert>
#include <iostream>
#include <cstring>
#include <vector>
#include "core/rand.h"
#include "fastboard.h"

//STATIC VARS-----------------------------------------------------------------------------
bool FastBoard::IS_ZOBRIST_INITALIZED = false;
Hash FastBoard::ZOBRIST_SIZE_X_HASH[MAX_SIZE+1];
Hash FastBoard::ZOBRIST_SIZE_Y_HASH[MAX_SIZE+1];
Hash FastBoard::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
Hash FastBoard::ZOBRIST_PLAYER_HASH[4];

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

FastBoard::FastBoard()
{
  init(19,19);
}

FastBoard::FastBoard(int size)
{
  init(size,size);
}

FastBoard::FastBoard(int x, int y)
{
  init(x,y);
}


FastBoard::FastBoard(const FastBoard& other)
{
  x_size = other.x_size;
  y_size = other.y_size;

  int arr_size = (x_size+1)*(y_size+2)+1;

  memcpy(colors, other.colors, sizeof(Color)*arr_size);
  memcpy(chain_data, other.chain_data, sizeof(ChainData)*arr_size);
  memcpy(chain_head, other.chain_head, sizeof(Loc)*arr_size);
  memcpy(next_in_chain, other.next_in_chain, sizeof(Loc)*arr_size);

  ko_loc = other.ko_loc;

  pos_hash = other.pos_hash;

  memcpy(adj_offsets, other.adj_offsets, sizeof(short)*8);
}

void FastBoard::init(int xS, int yS)
{
  assert(IS_ZOBRIST_INITALIZED);
  assert(xS <= MAX_SIZE && yS <= MAX_SIZE);

  x_size = xS;
  y_size = yS;

  for(int y = 0; y < y_size+2; y++)
  {
    for(int x = 0; x < x_size+1; x++)
    {
      Loc loc = x + y*(x_size+1);
      if(x == 0 || y == 0 || y == y_size+1)
        colors[loc] = C_WALL;
      else
      {
        colors[loc] = C_EMPTY;
        empty_list.add(loc);
      }
    }
  }
  colors[MAX_ARR_SIZE-1] = C_WALL;

  ko_loc = NULL_LOC;

  pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];

  Location::getAdjacentOffsets(adj_offsets,x_size);
}

void FastBoard::initHash()
{
  if(IS_ZOBRIST_INITALIZED)
    return;
  Rand rand("FastBoard::initHash()");

  for(int i = 0; i<MAX_ARR_SIZE; i++)
  {
    for(Color j = 0; j<4; j++)
    {
      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_BOARD_HASH[i][j] = 0;
      else
        ZOBRIST_BOARD_HASH[i][j] = rand.nextUInt64();
    }
  }
  for(int i = 0; i<4; i++)
    ZOBRIST_PLAYER_HASH[i] = rand.nextUInt64();
  for(int i = 0; i<MAX_SIZE+1; i++)
    ZOBRIST_SIZE_X_HASH[i] = rand.nextUInt64();
  for(int i = 0; i<MAX_SIZE+1; i++)
    ZOBRIST_SIZE_Y_HASH[i] = rand.nextUInt64();

  IS_ZOBRIST_INITALIZED = true;
}


//Gets the number of liberties of the chain at loc. Assertion: location must be black or white.
int FastBoard::getNumLiberties(Loc loc) const
{
  assert(colors[loc] == C_BLACK || colors[loc] == C_WHITE);
  return chain_data[chain_head[loc]].num_liberties;
}

//Check if moving here is illegal due to self-capture
bool FastBoard::isSuicide(Loc loc, Player player) const
{
  Player enemy = getEnemy(player);
  for(int i = 0; i < 4; i++)
  {
    Loc adj = loc + adj_offsets[i];

    if(colors[adj] == C_EMPTY)
      return false;
    else if(colors[adj] == player)
    {
      if(getNumLiberties(adj) > 1)
        return false;
    }
    else if(colors[adj] == enemy)
    {
      if(getNumLiberties(adj) == 1)
        return false;
    }
  }

  return true;
}

//Check if moving here is illegal due to simple ko
bool FastBoard::isKoBanned(Loc loc) const
{
  return loc == ko_loc;
}

//Check if moving here is illegal.
bool FastBoard::isLegal(Loc loc, Player player) const
{
  return loc == PASS_LOC || (loc >= 0 && loc < MAX_ARR_SIZE && (colors[loc] == C_EMPTY) && !isKoBanned(loc) && !isSuicide(loc, player));
}

//Check if this location contains a simple eye for the specified player.
bool FastBoard::isSimpleEye(Loc loc, Player player) const
{
  if(colors[loc] != C_EMPTY)
    return false;

  bool against_wall = false;

  //Check that surounding points are owned
  for(int i = 0; i < 4; i++)
  {
    Loc adj = loc + adj_offsets[i];
    if(colors[adj] == C_WALL)
      against_wall = true;
    else if(colors[adj] != player)
      return false;
  }

  //Check that opponent does not own too many diagonal points
  Player enemy = getEnemy(player);
  int num_enemy_corners = 0;
  for(int i = 4; i < 8; i++)
  {
    Loc corner = loc + adj_offsets[i];
    if(colors[corner] == enemy)
      num_enemy_corners++;
  }

  if(num_enemy_corners >= 2 || (against_wall && num_enemy_corners >= 1))
    return false;

  return true;
}

bool FastBoard::setStone(Loc loc, Color color)
{
  if(loc < 0 || loc >= MAX_ARR_SIZE || colors[loc] == C_WALL)
    return false;

  if(colors[loc] == color)
  {}
  else if(colors[loc] == C_EMPTY)
    playMoveAssumeLegal(loc,color);
  else if(color == C_EMPTY)
    removeSingleStone(loc);
  else {
    removeSingleStone(loc);
    if(!isSuicide(loc,color))
      playMoveAssumeLegal(loc,color);
  }

  ko_loc = NULL_LOC;
  return true;
}


//Attempts to play the specified move. Returns true if successful, returns false if the move was illegal.
bool FastBoard::playMove(Loc loc, Player player)
{
  if(isLegal(loc,player))
  {
    playMoveAssumeLegal(loc,player);
    return true;
  }
  return false;
}

//Plays the specified move, assuming it is legal, and returns a MoveRecord for the move
FastBoard::MoveRecord FastBoard::playMoveRecorded(Loc loc, Player player)
{
  MoveRecord record;
  record.loc = loc;
  record.pla = player;
  record.ko_loc = ko_loc;
  record.capDirs = 0;

  Player enemy = getEnemy(player);
  for(int i = 0; i < 4; i++)
  {
    int adj = loc + adj_offsets[i];
    if(colors[adj] == enemy && getNumLiberties(adj) == 1)
      record.capDirs |= (((uint8_t)1) << i);
  }
  playMoveAssumeLegal(loc, player);
  return record;
}

//Undo the move given by record. Moves MUST be undone in the order they were made.
//Undos will NOT typically restore the precise representation in the board to the way it was. The heads of chains
//might change, the order of the circular lists might change, etc.
void FastBoard::undo(FastBoard::MoveRecord record)
{
  ko_loc = record.ko_loc;

  Loc loc = record.loc;
  if(loc == PASS_LOC)
    return;

  //Re-fill stones in all captured directions
  for(int i = 0; i<4; i++)
  {
    int adj = loc + adj_offsets[i];
    if(record.capDirs & (1 << i))
    {
      if(colors[adj] == C_EMPTY)
        addChain(adj, getEnemy(record.pla));
    }
  }

  //Delete the stone played here.
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][colors[loc]];
  colors[loc] = C_EMPTY;
  empty_list.add(loc);

  //Uneat enemy liberties
  changeSurroundingLiberties(loc, getEnemy(record.pla),+1);

  //If this was not a single stone, we need to recompute the chain from scratch
  if(chain_data[chain_head[loc]].num_locs > 1)
  {
    //Run through the whole chain and make their heads point to nothing
    Loc cur = loc;
    do
    {
      chain_head[cur] = NULL_LOC;
      cur = next_in_chain[cur];
    } while (cur != loc);

    //Rebuild each chain adjacent now
    for(int i = 0; i<4; i++)
    {
      int adj = loc + adj_offsets[i];
      if(colors[adj] == record.pla && chain_head[adj] == NULL_LOC)
        rebuildChain(adj, record.pla);
    }
  }
}

//Plays the specified move, assuming it is legal.
void FastBoard::playMoveAssumeLegal(Loc loc, Player player)
{
  //Pass?
  if(loc == PASS_LOC)
  {
    ko_loc = NULL_LOC;
    return;
  }

  Player enemy = getEnemy(player);

  //Add the new stone as an independent group
  colors[loc] = player;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][player];
  chain_data[loc].owner = player;
  chain_data[loc].num_locs = 1;
  chain_data[loc].num_liberties = countImmediateLiberties(loc);
  chain_head[loc] = loc;
  next_in_chain[loc] = loc;
  empty_list.remove(loc);

  //Merge with surrounding friendly chains and capture any necessary enemy chains
  int num_captured = 0; //Number of stones captured
  Loc possible_ko_loc = NULL_LOC;  //What location a ko ban might become possible in
  int num_enemies_seen = 0;  //How many enemy chains we have seen so far
  Loc enemy_heads_seen[4];   //Heads of the enemy chains seen so far

  for(int i = 0; i < 4; i++)
  {
    int adj = loc + adj_offsets[i];

    //Friendly chain!
    if(colors[adj] == player)
    {
      //Already merged?
      if(chain_head[adj] == chain_head[loc])
        continue;

      //Otherwise, eat one liberty and merge them
      chain_data[chain_head[adj]].num_liberties--;
      mergeChains(adj,loc);
    }

    //Enemy chain!
    else if(colors[adj] == enemy)
    {
      Loc enemy_head = chain_head[adj];

      //Have we seen it already?
      bool seen = false;
      for(int j = 0; j<num_enemies_seen; j++)
        if(enemy_heads_seen[j] == enemy_head)
        {seen = true; break;}

      if(seen)
        continue;

      //Not already seen! Eat one liberty from it and mark it as seen
      chain_data[enemy_head].num_liberties--;
      enemy_heads_seen[num_enemies_seen++] = enemy_head;

      //Kill it?
      if(getNumLiberties(adj) == 0)
      {
        num_captured += removeChain(adj);
        possible_ko_loc = adj;
      }
    }
  }

  //We have a ko if 1 stone was captured and the capturing move is one isolated stone
  if(num_captured == 1 && chain_data[chain_head[loc]].num_locs == 1)
    ko_loc = possible_ko_loc;
  else
    ko_loc = NULL_LOC;

}

//Counts the number of liberties immediately next to loc
int FastBoard::countImmediateLiberties(Loc loc)
{
  int num_libs = 0;
  for(int i = 0; i < 4; i++)
    if(colors[loc + adj_offsets[i]] == C_EMPTY)
      num_libs++;

  return num_libs;
}

//Loc is a liberty of head's chain if loc is empty and adjacent to a stone of head.
//Assumes loc is empty
bool FastBoard::isLibertyOf(Loc loc, Loc head)
{
  for(int i = 0; i<4; i++)
  {
    Loc adj = loc + adj_offsets[i];
    if(colors[adj] == colors[head] && chain_head[adj] == head)
      return true;
  }
  return false;
}

void FastBoard::mergeChains(Loc loc1, Loc loc2)
{
  //Find heads
  Loc head1 = chain_head[loc1];
  Loc head2 = chain_head[loc2];

  assert(head1 != head2);
  assert(chain_data[head1].owner == chain_data[head2].owner);

  //Make sure head2 is the smaller chain.
  if(chain_data[head1].num_locs < chain_data[head2].num_locs)
  {
    Loc temp = head1;
    head1 = head2;
    head2 = temp;
  }

  //Iterate through each stone of head2's chain to make it a member of head1's chain.
  //Count new liberties for head1 as we go.
  chain_data[head1].num_locs += chain_data[head2].num_locs;
  int numNewLiberties = 0;
  Loc loc = head2;
  while(true)
  {
    //Any adjacent liberty is a new liberty for head1 if it is not adjacent to a stone of head1
    for(int i = 0; i<4; i++)
    {
      Loc adj = loc + adj_offsets[i];
      if(colors[adj] == C_EMPTY && !isLibertyOf(adj,head1))
        numNewLiberties++;
    }

    //Now, add this stone to head1.
    chain_head[loc] = head1;

    //If are not back around, we are done.
    if(next_in_chain[loc] != head2)
      loc = next_in_chain[loc];
    else
      break;
  }

  //Add in the liberties
  chain_data[head1].num_liberties += numNewLiberties;

  //We link up (head1 -> next1 -> ... -> last1 -> head1) and (head2 -> next2 -> ... -> last2 -> head2)
  //as: head1 -> head2 -> next2 -> ... -> last2 -> next1 -> ... -> last1 -> head1
  //loc is now last_2
  next_in_chain[loc] = next_in_chain[head1];
  next_in_chain[head1] = head2;
}

//Returns number of stones captured
int FastBoard::removeChain(Loc loc)
{
  int num_stones_removed = 0; //Num stones removed
  Player enemy = getEnemy(colors[loc]);

  //Walk around the chain...
  Loc cur = loc;
  do
  {
    //Empty out this location
    pos_hash ^= ZOBRIST_BOARD_HASH[cur][colors[cur]];
    colors[cur] = C_EMPTY;
    num_stones_removed++;
    empty_list.add(cur);

    //For each distinct enemy chain around, add a liberty to it.
    changeSurroundingLiberties(cur,enemy,+1);

    cur = next_in_chain[cur];

  } while (cur != loc);

  return num_stones_removed;
}

//Remove a single stone, even a stone part of a larger group.
void FastBoard::removeSingleStone(Loc loc)
{
  Player player = colors[loc];

  //Save the entire chain's stone locations
  int num_locs = chain_data[chain_head[loc]].num_locs;
  int locs[num_locs];
  int idx = 0;
  Loc cur = loc;
  do
  {
    locs[idx++] = cur;
    cur = next_in_chain[cur];
  } while (cur != loc);
  assert(idx == num_locs);

  //Delete the entire chain
  removeChain(loc);

  //Then add all the other stones back one by one.
  for(int i = 0; i<num_locs; i++) {
    if(locs[i] != loc)
      playMoveAssumeLegal(locs[i],player);
  }
}

//Add a chain of the given player to the given region of empty space, floodfilling it.
//Assumes that this region does not border any chains of the desired color already
void FastBoard::addChain(Loc loc, Player pla)
{
  chain_data[loc].num_liberties = 0;
  chain_data[loc].num_locs = 0;
  chain_data[loc].owner = pla;

  //Add a chain with links front -> ... -> loc -> loc with all head pointers towards loc
  Loc front = addChainHelper(loc, loc, loc, pla);

  //Now, we make loc point to front, and that completes the circle!
  next_in_chain[loc] = front;
}

//Floodfill a chain of the given color into this region of empty spaces
//Make the specified loc the head for all the chains and updates the chainData of head with the number of stones.
//Does NOT connect the stones into a circular list. Rather, it produces an linear linked list with the tail pointing
//to tailTarget, and returns the head of the list. The tail is guaranteed to be loc.
Loc FastBoard::addChainHelper(Loc head, Loc tailTarget, Loc loc, Player player)
{
  //Add stone here
  colors[loc] = player;
  pos_hash ^= ZOBRIST_BOARD_HASH[loc][player];
  chain_head[loc] = head;
  chain_data[head].num_locs++;
  next_in_chain[loc] = tailTarget;
  empty_list.remove(loc);

  //Eat enemy liberties
  changeSurroundingLiberties(loc,getEnemy(player),-1);

  //Recursively add stones around us.
  Loc nextTailTarget = loc;
  for(int i = 0; i<4; i++)
  {
    Loc adj = loc + adj_offsets[i];
    if(colors[adj] == C_EMPTY)
      nextTailTarget = addChainHelper(head,nextTailTarget,adj,player);
  }
  return nextTailTarget;
}

//Floods through a chain of the specified player already on the board
//rebuilding its links and counting its liberties as we go.
//Requires that all their heads point towards
//some invalid location, such as NULL_LOC or a location not of color.
//The head of the chain will be loc.
void FastBoard::rebuildChain(Loc loc, Player pla)
{
  chain_data[loc].num_liberties = 0;
  chain_data[loc].num_locs = 0;
  chain_data[loc].owner = pla;

  //Rebuild chain with links front -> ... -> loc -> loc with all head pointers towards loc
  Loc front = rebuildChainHelper(loc, loc, loc, pla);

  //Now, we make loc point to front, and that completes the circle!
  next_in_chain[loc] = front;
}

//Does same thing as addChain, but floods through a chain of the specified color already on the board
//rebuilding its links and also counts its liberties as we go. Requires that all their heads point towards
//some invalid location, such as NULL_LOC or a location not of color.
Loc FastBoard::rebuildChainHelper(Loc head, Loc tailTarget, Loc loc, Player player)
{
  //Count new liberties
  for(int i = 0; i<4; i++)
  {
    Loc adj = loc + adj_offsets[i];
    if(colors[adj] == C_EMPTY && !isLibertyOf(adj,head))
      chain_data[head].num_liberties++;
  }

  //Add stone here to the chain by setting its head
  chain_head[loc] = head;
  chain_data[head].num_locs++;
  next_in_chain[loc] = tailTarget;

  //Recursively add stones around us.
  Loc nextTailTarget = loc;
  for(int i = 0; i<4; i++)
  {
    Loc adj = loc + adj_offsets[i];
    if(colors[adj] == player && chain_head[adj] != head)
      nextTailTarget = rebuildChainHelper(head,nextTailTarget,adj,player);
  }
  return nextTailTarget;
}

//Apply the specified delta to the liberties of all adjacent groups of the specified color
void FastBoard::changeSurroundingLiberties(Loc loc, Player player, int delta)
{
  int num_seen = 0;  //How many enemy chains we have seen so far
  Loc heads_seen[4];   //Heads of the enemy chains seen so far
  for(int i = 0; i < 4; i++)
  {
    int adj = loc + adj_offsets[i];
    if(colors[adj] == player)
    {
      Loc head = chain_head[adj];

      //Have we seen it already?
      bool seen = false;
      for(int j = 0; j<num_seen; j++)
        if(heads_seen[j] == head)
        {seen = true; break;}

      if(seen)
        continue;

      //Not already seen! Eat one liberty from it and mark it as seen
      chain_data[head].num_liberties += delta;
      heads_seen[num_seen++] = head;
    }
  }
}

ostream& operator<<(ostream& out, const FastBoard& board)
{
  for(int y = 0; y < board.y_size; y++)
  {
    for(int x = 0; x < board.x_size; x++)
    {
      Loc loc = Location::getLoc(x,y,board.x_size);
      //char s = getCharOfColor(board.colors[loc]);
      char s = board.colors[loc] == C_EMPTY ? '.' : '0' + board.chain_data[board.chain_head[loc]].num_liberties;

      out << s << ' ';
    }
    out << " ";
    for(int x = 0; x < board.x_size; x++)
    {
      Loc loc = Location::getLoc(x,y,board.x_size);
      char s = getCharOfColor(board.colors[loc]);
      out << s << ' ';
    }

    out << "\n";
  }
  out << "\n";
  return out;
}


FastBoard::PointList::PointList()
{
  std::memset(list_, 0, sizeof(list_));
  std::memset(indices_, -1, sizeof(indices_));
  size_ = 0;
}

FastBoard::PointList::PointList(const FastBoard::PointList& other)
{
  std::memcpy(list_, other.list_, sizeof(list_));
  std::memcpy(indices_, other.indices_, sizeof(indices_));
  size_ = other.size_;
}

void FastBoard::PointList::add(Loc loc)
{
  //assert (size_ < MAX_PLAY_SIZE);
  list_[size_] = loc;
  indices_[loc] = size_;
  size_++;
}

void FastBoard::PointList::remove(Loc loc)
{
  int index = indices_[loc];
  int end_loc = list_[size_-1];
  list_[index] = end_loc;
  indices_[end_loc] = index;
  size_--;
}

int FastBoard::PointList::size()
{
  return size_;
}

Loc& FastBoard::PointList::operator[](int n)
{
  assert (n < size_);
  return list_[n];
}

Loc Location::getLoc(int x, int y, int x_size)
{
  return (x+1) + (y+1)*(x_size+1);
}
int Location::getX(Loc loc, int x_size)
{
  return (loc % (x_size+1)) - 1;
}
int Location::getY(Loc loc, int x_size)
{
  return (loc / (x_size+1)) - 1;
}
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size)
{
  adj_offsets[0] = -(x_size+1);
  adj_offsets[1] = -1;
  adj_offsets[2] = 1;
  adj_offsets[3] = (x_size+1);
  adj_offsets[4] = -(x_size+1)-1;
  adj_offsets[5] = -(x_size+1)+1;
  adj_offsets[6] = (x_size+1)-1;
  adj_offsets[7] = (x_size+1)+1;
}

string Location::toString(Loc loc, int x_size)
{
  char buf[128];
  sprintf(buf,"(%d,%d)",getX(loc,x_size),getY(loc,x_size));
  return string(buf);
}

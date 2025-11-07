#include "../neuralnet/nninputs.h"

using namespace std;

void NNInputs::fillRowVDots(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN_X);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN_Y);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill_n(rowBin, NUM_FEATURES_SPATIAL_V_DOTS * nnXLen * nnYLen,false);
  std::fill_n(rowGlobal, NUM_FEATURES_SPATIAL_V_DOTS, 0.0f);

  const Player pla = nextPlayer;
  const Player opp = getOpp(pla);
  const int xSize = board.x_size;
  const int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NUM_FEATURES_SPATIAL_V_DOTS;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  vector<Color> captures;
  vector<Color> bases;
  board.calculateOneMoveCaptureAndBasePositionsForDots(hist.rules.multiStoneSuicideLegal, captures, bases);

  Color grounding[Board::MAX_ARR_SIZE];
  board.calculateGroundingWhiteScore(grounding);

  auto boardString = board.toString();
  (void)boardString;

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      const int pos = NNPos::xyToPos(x,y,nnXLen);
      const Loc loc = Location::getLoc(x,y,xSize);

      setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_ON_BOARD, 1.0f, posStride, featureStride);

      const State state = board.getState(loc);
      const Color activeColor = getActiveColor(state);

      if(activeColor == pla)
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_PLAYER, 1.0f, posStride, featureStride);
      else if(activeColor == opp)
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_PLAYER_OPP, 1.0f, posStride, featureStride);

      const Color captureColor = captures[loc];
      if ((pla & captureColor) != 0) {
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_PLAYER_CAPTURES, 1.0f, posStride, featureStride);
      }
      if ((opp & captureColor) != 0) {
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_PLAYER_OPP_CAPTURES, 1.0f, posStride, featureStride);
      }

      const Color baseColor = bases[loc];
      if ((pla & baseColor) != 0) {
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_PLAYER_SURROUNDINGS, 1.0f, posStride, featureStride);
      }
      if ((opp & baseColor) != 0) {
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_PLAYER_OPP_SURROUNDINGS, 1.0f, posStride, featureStride);
      }

      const Color placedColor = getPlacedDotColor(state);
      if (placedColor != C_EMPTY && placedColor != activeColor) {
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_DEAD_DOTS, 1.0f, posStride, featureStride);
      }

      if (grounding[loc] != C_EMPTY) {
        setRowBin(rowBin,pos,DOTS_FEATURE_SPATIAL_GROUNDED, 1.0f, posStride, featureStride);
      }
    }
  }

  int maxTurnsOfHistoryToInclude = 5;
  if (hist.isGameFinished) {

  }
}
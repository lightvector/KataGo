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
  std::fill_n(rowGlobal, NUM_FEATURES_GLOBAL_V_DOTS, 0.0f);

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

  const Rules& rules = hist.rules;

  vector<Color> captures;
  vector<Color> bases;
  board.calculateOneMoveCaptureAndBasePositionsForDots(captures, bases);
  int deadDotsCount = 0;

  auto setSpatial = [&](const int pos, const DotsSpatialFeature spatialFeature) {
    setRowBin(rowBin, pos, static_cast<int>(spatialFeature), 1.0f, posStride, featureStride);
  };

  auto setGlobal = [&](const DotsGlobalFeature globalFeature, const float value = 1.0f) {
    rowGlobal[static_cast<int>(globalFeature)] = value;
  };

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      const int pos = NNPos::xyToPos(x,y,nnXLen);
      const Loc loc = Location::getLoc(x,y,xSize);

      setSpatial(pos, DotsSpatialFeature::OnBoard);

      const State state = board.getState(loc);
      const Color activeColor = getActiveColor(state);
      const Color placedColor = getPlacedDotColor(state);

      if (activeColor == pla)
        setSpatial(pos, DotsSpatialFeature::PlayerActive);
      else if (activeColor == opp)
        setSpatial(pos, DotsSpatialFeature::PlayerOppActive);
      else
        assert(C_EMPTY == activeColor);

      if (placedColor == pla)
        setSpatial(pos, DotsSpatialFeature::PlayerPlaced);
      else if (placedColor == opp)
        setSpatial(pos, DotsSpatialFeature::PlayerOppPlaced);
      else
        assert(C_EMPTY == placedColor);

      if (activeColor != C_EMPTY && placedColor != C_EMPTY && placedColor != activeColor) {
        // Needed for more correct score calculation, but probably it's redundant considering placed dots
        setSpatial(pos, DotsSpatialFeature::DeadDots);
        deadDotsCount++;
      }

      if (isGrounded(state)) {
        setSpatial(pos, DotsSpatialFeature::Grounded);
      }

      const Color captureColor = captures[loc];
      if ((pla & captureColor) != 0) {
        setSpatial(pos, DotsSpatialFeature::PlayerCaptures);
      }
      if ((opp & captureColor) != 0) {
        setSpatial(pos, DotsSpatialFeature::PlayerOppCaptures);
      }

      const Color baseColor = bases[loc];
      if ((pla & baseColor) != 0) {
        setSpatial(pos, DotsSpatialFeature::PlayerSurroundings);
      }
      if ((opp & baseColor) != 0) {
        setSpatial(pos, DotsSpatialFeature::PlayerOppSurroundings);
      }

      // TODO: Set up history and ladder features
    }
  }

  assert(deadDotsCount == board.numBlackCaptures + board.numWhiteCaptures);

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  const float bArea = static_cast<float>(xSize * ySize);
  //Bound komi just in case
  if(selfKomi > bArea+NNPos::KOMI_CLIP_RADIUS)
    selfKomi = bArea+NNPos::KOMI_CLIP_RADIUS;
  if(selfKomi < -bArea-NNPos::KOMI_CLIP_RADIUS)
    selfKomi = -bArea-NNPos::KOMI_CLIP_RADIUS;
  setGlobal(DotsGlobalFeature::Komi, selfKomi / NNPos::KOMI_CLIP_RADIUS);

  if (rules.multiStoneSuicideLegal) {
    setGlobal(DotsGlobalFeature::Suicide);
  }

  if (rules.dotsCaptureEmptyBases) {
    setGlobal(DotsGlobalFeature::CaptureEmpty);
  }

  if (const int startPos = rules.startPos; startPos >= Rules::START_POS_CROSS) {
    setGlobal(DotsGlobalFeature::StartPosCross);
    if (startPos >= Rules::START_POS_CROSS_2) {
      setGlobal(DotsGlobalFeature::StartPosCross2);
      if (startPos >= Rules::START_POS_CROSS_4) {
        setGlobal(DotsGlobalFeature::StartPosCross4);
      }
    }
  }

  if (rules.startPosIsRandom) {
    setGlobal(DotsGlobalFeature::StartPosIsRandom);
  }

  if (hist.winOrEffectiveDrawByGrounding(board, pla)) {
    // Train to better understand grounding
    setGlobal(DotsGlobalFeature::WinByGrounding);
  }

  setGlobal(DotsGlobalFeature::FieldSizeKomiParity, 0.0f); // TODO: implement later
}
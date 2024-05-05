# See the equivalents in trainingwrite.h and trainingwrite.cpp in the cpp directory.
# Translated from the c++ implementation.

from dataclasses import dataclass
from typing import ClassVar
import datetime
import math
import numpy as np
import random
from board import Board
import modelconfigs

@dataclass
class SGFMetadata:
    inverseBRank: int = 0
    inverseWRank: int = 0
    bIsUnranked: bool = False
    wIsUnranked: bool = False
    bRankIsUnknown: bool = False
    wRankIsUnknown: bool = False
    bIsHuman: bool = False
    wIsHuman: bool = False

    gameIsUnrated: bool = False
    gameRatednessIsUnknown: bool = False

    tcIsUnknown: bool = True
    tcIsNone: bool = False
    tcIsAbsolute: bool = False
    tcIsSimple: bool = False
    tcIsByoYomi: bool = False
    tcIsCanadian: bool = False
    tcIsFischer: bool = False

    mainTimeSeconds: float = 0.0
    periodTimeSeconds: float = 0.0
    byoYomiPeriods: int = 0
    canadianMoves: int = 0

    gameDate: datetime.date = datetime.date(1970, 1, 1)

    source: int = 0

    SOURCE_OGS: ClassVar[int] = 1
    SOURCE_KGS: ClassVar[int] = 2
    SOURCE_FOX: ClassVar[int] = 3
    SOURCE_TYGEM: ClassVar[int] = 4
    SOURCE_GOGOD: ClassVar[int] = 5
    SOURCE_GO4GO: ClassVar[int] = 6

    METADATA_INPUT_NUM_CHANNELS: ClassVar[int] = 192

    @classmethod
    def of_dict(cls, data: dict):
        data["gameDate"] = datetime.date.fromisoformat(data["gameDate"])
        return cls(**data)

    def to_dict(self):
        data = self.__dict__.copy()
        data["gameDate"] = data["gameDate"].isoformat()
        return data

    @classmethod
    def get_katago_selfplay_metadata(cls, rand: random.Random) -> "SGFMetadata":
        return SGFMetadata(
            inverseBRank = 0,
            inverseWRank = 0,
            bIsUnranked = False,
            wIsUnranked = False,
            bRankIsUnknown = False,
            wRankIsUnknown = False,
            bIsHuman = False,
            wIsHuman = False,

            gameIsUnrated = False,
            gameRatednessIsUnknown = False,

            tcIsUnknown = False,
            tcIsNone = False,
            tcIsAbsolute = False,
            tcIsSimple = False,
            tcIsByoYomi = True,
            tcIsCanadian = False,
            tcIsFischer = False,

            mainTimeSeconds = rand.choice([300,600,900,1200]),
            periodTimeSeconds = rand.choice([10.0,15.0,30.0]),
            byoYomiPeriods = 5,
            canadianMoves = 0,

            gameDate = datetime.date(2022, 1, 1) + datetime.timedelta(days=rand.randint(0,722)),
            source = 0,
        )


    def get_metadata_row(self, nextPlayer, boardArea) -> np.ndarray:
        if isinstance(nextPlayer,str):
            if nextPlayer.lower() == "w":
                nextPlayer = Board.WHITE
            elif nextPlayer.lower() == "b":
                nextPlayer = Board.BLACK

        # This class implements "version 1" of sgf metadata encoding, make sure channel dims match up.
        meta_encoder_version = 1
        assert modelconfigs.get_num_meta_encoder_input_features(meta_encoder_version) == self.METADATA_INPUT_NUM_CHANNELS

        rowMetadata = np.zeros(self.METADATA_INPUT_NUM_CHANNELS, dtype=np.float32)

        plaIsHuman = self.wIsHuman if nextPlayer == Board.WHITE else self.bIsHuman
        oppIsHuman = self.bIsHuman if nextPlayer == Board.WHITE else self.wIsHuman
        rowMetadata[0] = 1.0 if plaIsHuman else 0.0
        rowMetadata[1] = 1.0 if oppIsHuman else 0.0

        plaIsUnranked = self.wIsUnranked if nextPlayer == Board.WHITE else self.bIsUnranked
        oppIsUnranked = self.bIsUnranked if nextPlayer == Board.WHITE else self.wIsUnranked
        rowMetadata[2] = 1.0 if plaIsUnranked else 0.0
        rowMetadata[3] = 1.0 if oppIsUnranked else 0.0

        plaRankIsUnknown = self.wRankIsUnknown if nextPlayer == Board.WHITE else self.bRankIsUnknown
        oppRankIsUnknown = self.bRankIsUnknown if nextPlayer == Board.WHITE else self.wRankIsUnknown
        rowMetadata[4] = 1.0 if plaRankIsUnknown else 0.0
        rowMetadata[5] = 1.0 if oppRankIsUnknown else 0.0

        RANK_START_IDX = 6
        invPlaRank = self.inverseWRank if nextPlayer == Board.WHITE else self.inverseBRank
        invOppRank = self.inverseBRank if nextPlayer == Board.WHITE else self.inverseWRank
        RANK_LEN_PER_PLA = 34
        if not plaIsUnranked:
            for i in range(min(invPlaRank, RANK_LEN_PER_PLA)):
                rowMetadata[RANK_START_IDX + i] = 1.0
        if not oppIsUnranked:
            for i in range(min(invOppRank, RANK_LEN_PER_PLA)):
                rowMetadata[RANK_START_IDX + RANK_LEN_PER_PLA + i] = 1.0

        assert RANK_START_IDX + 2 * RANK_LEN_PER_PLA == 74
        rowMetadata[74] = 0.5 if self.gameRatednessIsUnknown else 1.0 if self.gameIsUnrated else 0.0

        rowMetadata[75] = 1.0 if self.tcIsUnknown else 0.0
        rowMetadata[76] = 1.0 if self.tcIsNone else 0.0
        rowMetadata[77] = 1.0 if self.tcIsAbsolute else 0.0
        rowMetadata[78] = 1.0 if self.tcIsSimple else 0.0
        rowMetadata[79] = 1.0 if self.tcIsByoYomi else 0.0
        rowMetadata[80] = 1.0 if self.tcIsCanadian else 0.0
        rowMetadata[81] = 1.0 if self.tcIsFischer else 0.0
        assert (
            rowMetadata[75]
            + rowMetadata[76]
            + rowMetadata[77]
            + rowMetadata[78]
            + rowMetadata[79]
            + rowMetadata[80]
            + rowMetadata[81]
            == 1.0
        )

        mainTimeSecondsCapped = min(max(self.mainTimeSeconds, 0.0), 3.0 * 86400)
        periodTimeSecondsCapped = min(max(self.periodTimeSeconds, 0.0), 1.0 * 86400)
        rowMetadata[82] = 0.4 * (math.log(mainTimeSecondsCapped + 60.0) - 6.5)
        rowMetadata[83] = 0.3 * (math.log(periodTimeSecondsCapped + 1.0) - 3.0)
        byoYomiPeriodsCapped = min(max(self.byoYomiPeriods, 0), 50)
        canadianMovesCapped = min(max(self.canadianMoves, 0), 50)
        rowMetadata[84] = 0.5 * (math.log(byoYomiPeriodsCapped + 2.0) - 1.5)
        rowMetadata[85] = 0.25 * (math.log(canadianMovesCapped + 2.0) - 1.5)

        rowMetadata[86] = 0.5 * math.log(boardArea / 361.0)

        daysDifference = (self.gameDate - datetime.date(1970, 1, 1)).days
        DATE_START_IDX = 87
        DATE_LEN = 32
        period = 7.0
        factor = pow(80000.0, 1.0 / (DATE_LEN - 1))
        twopi = 6.283185307179586476925
        for i in range(DATE_LEN):
            numRevolutions = daysDifference / period
            rowMetadata[DATE_START_IDX + i * 2 + 0] = math.cos(numRevolutions * twopi)
            rowMetadata[DATE_START_IDX + i * 2 + 1] = math.sin(numRevolutions * twopi)
            period *= factor

        assert DATE_START_IDX + 2 * DATE_LEN == 151

        assert 0 <= self.source < 16
        rowMetadata[151 + self.source] = 1.0

        assert 151 + 16 < self.METADATA_INPUT_NUM_CHANNELS

        return rowMetadata

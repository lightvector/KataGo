
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import math

import numpy as np

from board import Board
from features import Features
from sgfmetadata import SGFMetadata

if TYPE_CHECKING:
    from model_pytorch import Model

class GameState:
    RULES_TT = {
        "koRule": "KO_POSITIONAL",
        "scoringRule": "SCORING_AREA",
        "taxRule": "TAX_NONE",
        "multiStoneSuicideLegal": True,
        "hasButton": False,
        "encorePhase": 0,
        "passWouldEndPhase": False,
        "whiteKomi": 7.5,
        "asymPowersOfTwo": 0.0,
    }
    RULES_CHINESE = {
        "koRule": "KO_SIMPLE",
        "scoringRule": "SCORING_AREA",
        "taxRule": "TAX_NONE",
        "multiStoneSuicideLegal": False,
        "hasButton": False,
        "encorePhase": 0,
        "passWouldEndPhase": False,
        "whiteKomi": 7.5,
        "asymPowersOfTwo": 0.0,
    }
    RULES_JAPANESE = {
        "koRule": "KO_SIMPLE",
        "scoringRule": "SCORING_TERRITORY",
        "taxRule": "TAX_SEKI",
        "multiStoneSuicideLegal": False,
        "hasButton": False,
        "encorePhase": 0,
        "passWouldEndPhase": False,
        "whiteKomi": 6.5,
        "asymPowersOfTwo": 0.0,
    }


    def __init__(self, board_size: int, rules: Dict[str,Any]):
        self.board_size = board_size
        self.board = Board(size=board_size)
        self.moves = []
        self.boards = [self.board.copy()]
        self.rules = rules.copy()
        self.redo_stack = []

    def play(self, pla, loc):
        self.board.play(pla,loc)
        self.moves.append((pla,loc))
        self.boards.append(self.board.copy())
        if len(self.redo_stack) > 0:
            move, _ = self.redo_stack[-1]
            if move == (pla,loc):
                self.redo_stack.pop()
            else:
                self.redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self.moves) > 0

    def undo(self):
        assert self.can_undo()
        move = self.moves.pop()
        board = self.boards.pop()
        self.board = self.boards[-1].copy()
        self.redo_stack.append((move,board))

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def redo(self):
        assert self.can_redo()
        move, board = self.redo_stack.pop()
        self.moves.append(move)
        self.boards.append(board)
        self.board = self.boards[-1].copy()

    def get_input_features(self, features: Features):
        bin_input_data = np.zeros(shape=[1]+features.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[1]+features.global_input_shape, dtype=np.float32)
        pos_len = features.pos_len
        pla = self.board.pla
        opp = Board.get_opp(pla)
        move_idx = len(self.moves)
        # fill_row_features assumes N(HW)C order but we actually use NCHW order in the model, so work with it and revert
        bin_input_data = np.transpose(bin_input_data,axes=(0,2,3,1))
        bin_input_data = bin_input_data.reshape([1,pos_len*pos_len,-1])
        features.fill_row_features(self.board,pla,opp,self.boards,self.moves,move_idx,self.rules,bin_input_data,global_input_data,idx=0)
        bin_input_data = bin_input_data.reshape([1,pos_len,pos_len,-1])
        bin_input_data = np.transpose(bin_input_data,axes=(0,3,1,2))
        return bin_input_data, global_input_data

    def get_model_outputs(self, model: "Model", sgfmeta: Optional[SGFMetadata] = None, extra_output_names: List[str] = []):
        import torch
        from model_pytorch import Model, ExtraOutputs
        with torch.no_grad():
            model.eval()
            features = Features(model.config, model.pos_len)

            bin_input_data, global_input_data = self.get_input_features(features)
            # Currently we don't actually do any symmetries
            # symmetry = 0
            # model_outputs = model(apply_symmetry(batch["binaryInputNCHW"],symmetry),batch["globalInputNC"])

            input_meta = None
            if sgfmeta is not None:
                input_meta = torch.tensor(sgfmeta.get_metadata_row(self.board.pla), dtype=torch.float32, device=model.device)
                input_meta = input_meta.reshape([1,-1])

            extra_outputs = ExtraOutputs(extra_output_names)

            model_outputs = model(
                torch.tensor(bin_input_data, dtype=torch.float32, device=model.device),
                torch.tensor(global_input_data, dtype=torch.float32, device=model.device),
                input_meta = input_meta,
                extra_outputs=extra_outputs,
            )

            available_extra_outputs = extra_outputs.available

            outputs = model.postprocess_output(model_outputs)
            (
                policy_logits,      # N, num_policy_outputs, move
                value_logits,       # N, {win,loss,noresult}
                td_value_logits,    # N, {long, mid, short} {win,loss,noresult}
                pred_td_score,      # N, {long, mid, short}
                ownership_pretanh,  # N, 1, y, x
                pred_scoring,       # N, 1, y, x
                futurepos_pretanh,  # N, 2, y, x
                seki_logits,        # N, 4, y, x
                pred_scoremean,     # N
                pred_scorestdev,    # N
                pred_lead,          # N
                pred_variance_time, # N
                pred_shortterm_value_error, # N
                pred_shortterm_score_error, # N
                scorebelief_logits, # N, 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)
            ) = (x[0] for x in outputs[0]) # N = 0

            policy0 = torch.nn.functional.softmax(policy_logits[0,:],dim=0).cpu().numpy()
            policy1 = torch.nn.functional.softmax(policy_logits[1,:],dim=0).cpu().numpy()
            value = torch.nn.functional.softmax(value_logits,dim=0).cpu().numpy()
            td_value = torch.nn.functional.softmax(td_value_logits[0,:],dim=0).cpu().numpy()
            td_value2 = torch.nn.functional.softmax(td_value_logits[1,:],dim=0).cpu().numpy()
            td_value3 = torch.nn.functional.softmax(td_value_logits[2,:],dim=0).cpu().numpy()
            scoremean = pred_scoremean.cpu().item()
            td_score = pred_td_score.cpu().numpy()
            scorestdev = pred_scorestdev.cpu().item()
            lead = pred_lead.cpu().item()
            vtime = pred_variance_time.cpu().item()
            estv = math.sqrt(pred_shortterm_value_error.cpu().item())
            ests = math.sqrt(pred_shortterm_score_error.cpu().item())
            ownership = torch.tanh(ownership_pretanh).cpu().numpy()
            scoring = pred_scoring.cpu().numpy()
            futurepos = torch.tanh(futurepos_pretanh).cpu().numpy()
            seki_probs = torch.nn.functional.softmax(seki_logits[0:3,:,:],dim=0).cpu().numpy()
            seki = seki_probs[1] - seki_probs[2]
            seki2 = torch.sigmoid(seki_logits[3,:,:]).cpu().numpy()
            scorebelief = torch.nn.functional.softmax(scorebelief_logits,dim=0).cpu().numpy()

        board = self.board

        moves_and_probs0 = []
        for i in range(len(policy0)):
            move = features.tensor_pos_to_loc(i,board)
            if i == len(policy0)-1:
                moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
            elif board.would_be_legal(board.pla,move):
                moves_and_probs0.append((move,policy0[i]))

        moves_and_probs1 = []
        for i in range(len(policy1)):
            move = features.tensor_pos_to_loc(i,board)
            if i == len(policy1)-1:
                moves_and_probs1.append((Board.PASS_LOC,policy1[i]))
            elif board.would_be_legal(board.pla,move):
                moves_and_probs1.append((move,policy1[i]))

        ownership_flat = ownership.reshape([features.pos_len * features.pos_len])
        ownership_by_loc = []
        board = self.board
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                if board.pla == Board.WHITE:
                    ownership_by_loc.append((loc,ownership_flat[pos]))
                else:
                    ownership_by_loc.append((loc,-ownership_flat[pos]))

        scoring_flat = scoring.reshape([features.pos_len * features.pos_len])
        scoring_by_loc = []
        board = self.board
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                if board.pla == Board.WHITE:
                    scoring_by_loc.append((loc,scoring_flat[pos]))
                else:
                    scoring_by_loc.append((loc,-scoring_flat[pos]))

        futurepos0_flat = futurepos[0,:,:].reshape([features.pos_len * features.pos_len])
        futurepos0_by_loc = []
        board = self.board
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                if board.pla == Board.WHITE:
                    futurepos0_by_loc.append((loc,futurepos0_flat[pos]))
                else:
                    futurepos0_by_loc.append((loc,-futurepos0_flat[pos]))

        futurepos1_flat = futurepos[1,:,:].reshape([features.pos_len * features.pos_len])
        futurepos1_by_loc = []
        board = self.board
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                if board.pla == Board.WHITE:
                    futurepos1_by_loc.append((loc,futurepos1_flat[pos]))
                else:
                    futurepos1_by_loc.append((loc,-futurepos1_flat[pos]))

        seki_flat = seki.reshape([features.pos_len * features.pos_len])
        seki_by_loc = []
        board = self.board
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                if board.pla == Board.WHITE:
                    seki_by_loc.append((loc,seki_flat[pos]))
                else:
                    seki_by_loc.append((loc,-seki_flat[pos]))

        seki_flat2 = seki2.reshape([features.pos_len * features.pos_len])
        seki_by_loc2 = []
        board = self.board
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x,y)
                pos = features.loc_to_tensor_pos(loc,board)
                seki_by_loc2.append((loc,seki_flat2[pos]))

        moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)
        # Generate a random number biased small and then find the appropriate move to make
        # Interpolate from moving uniformly to choosing from the triangular distribution
        alpha = 1
        beta = 1 + math.sqrt(max(0,len(self.moves)-20))
        r = np.random.beta(alpha,beta)
        probsum = 0.0
        i = 0
        genmove_result = Board.PASS_LOC
        while True:
            (move,prob) = moves_and_probs[i]
            probsum += prob
            if i >= len(moves_and_probs)-1 or probsum > r:
                genmove_result = move
                break
            i += 1

        # Transpose attention so that both it and reverse attention are in n c (hw) format.
        for name in list(extra_outputs.returned.keys()):
            if name.endswith(".attention"):
                extra_outputs.returned[name] = torch.transpose(extra_outputs.returned[name],1,2)

        return {
            "policy0": policy0,
            "policy1": policy1,
            "moves_and_probs0": moves_and_probs0,
            "moves_and_probs1": moves_and_probs1,
            "value": value,
            "td_value": td_value,
            "td_value2": td_value2,
            "td_value3": td_value3,
            "scoremean": scoremean,
            "td_score": td_score,
            "scorestdev": scorestdev,
            "lead": lead,
            "vtime": vtime,
            "estv": estv,
            "ests": ests,
            "ownership": ownership,
            "ownership_by_loc": ownership_by_loc,
            "scoring": scoring,
            "scoring_by_loc": scoring_by_loc,
            "futurepos": futurepos,
            "futurepos0_by_loc": futurepos0_by_loc,
            "futurepos1_by_loc": futurepos1_by_loc,
            "seki": seki,
            "seki_by_loc": seki_by_loc,
            "seki2": seki2,
            "seki_by_loc2": seki_by_loc2,
            "scorebelief": scorebelief,
            "genmove_result": genmove_result,
            **{ name:activation[0].numpy() for name, activation in extra_outputs.returned.items() },
            "available_extra_outputs": available_extra_outputs,
        }


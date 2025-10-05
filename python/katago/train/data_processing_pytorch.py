import logging
import os
from enum import Enum, auto

import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

from ..train import modelconfigs

def read_npz_training_data(npz_files, batch_size: int, world_size: int, rank: int, pos_len_x: int, pos_len_y: int, device,
                           randomize_symmetries: bool, include_meta: bool, model_config: modelconfigs.ModelConfig):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    (h_base,h_builder) = build_history_matrices(model_config, device)

    include_qvalues = model_config["version"] >= 16

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"].astype(np.float32)
            valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
            if include_meta:
                metadataInputNC = npz["metadataInputNC"].astype(np.float32)
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = npz["qValueTargetsNCMove"].astype(np.float32)
            else:
                qValueTargetsNCMove = None
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len_x * pos_len_y + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:, :pos_len_x * pos_len_y]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len_x, pos_len_y
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    if not npz_files:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_npz_file, npz_files[0])

        for next_file in (npz_files[1:] + [None]):
            (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = future.result()

            num_samples = binaryInputNCHW.shape[0]
            # Just discard stuff that doesn't divide evenly
            num_whole_steps = num_samples // (batch_size * world_size)

            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            if next_file is not None:
                logging.info(f"Preloading {next_file} while processing this file")
                future = executor.submit(load_npz_file, next_file)

            for n in range(num_whole_steps):
                start = (n * world_size + rank) * batch_size
                end = start + batch_size

                batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
                batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                if include_qvalues:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

                (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                    model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                )

                if randomize_symmetries:
                    symm = int(rand.integers(0,8))
                    batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                    batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len_x, pos_len_y)
                    batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
                    if include_qvalues:
                        batch_qValueTargetsNCMove = apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len_x, pos_len_y)

                batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCHW = batch_binaryInputNCHW,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                    scoreDistrN = batch_scoreDistrN,
                    valueTargetsNCHW = batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len_x, pos_len_y):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len_x, pos_len_y))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len_x * pos_len_y),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)

class GoSpatialFeature(Enum):
    ON_BOARD = 0
    PLA_STONE = 1
    OPP_STONE = 2
    LIBERTIES_1 = 3
    LIBERTIES_2 = 4
    LIBERTIES_3 = 5
    SUPER_KO_BANNED = 6
    KO_RECAP_BLOCKED = 7
    KO_EXTRA = 8
    PREV_1_LOC = 9
    PREV_2_LOC = 10
    PREV_3_LOC = 11
    PREV_4_LOC = 12
    PREV_5_LOC = 13
    LADDER_CAPTURED = 14
    LADDER_CAPTURED_PREVIOUS_1 = 15
    LADDER_CAPTURED_PREVIOUS_2 = 16
    LADDER_WORKING_MOVES = 17
    AREA_PLA = 18
    AREA_OPP = 19
    SECOND_ENCORE_PLA = 20
    SECOND_ENCORE_OPP = 21

class GoGlobalFeature(Enum):
    PREV_1_LOC_PASS = 0
    PREV_2_LOC_PASS = 1
    PREV_3_LOC_PASS = 2
    PREV_4_LOC_PASS = 3
    PREV_5_LOC_PASS = 4
    KOMI = 5
    KO_RULE_NOT_SIMPLE = 6
    KO_RULE_EXTRA = 7
    SUICIDE = 8
    SCORING_TERRITORY = 9
    TAX_SEKI = 10
    TAX_ALL = 11
    ENCORE_PHASE_1 = 12
    ENCORE_PHASE_2 = 13
    PASS_WOULD_END_PHASE = 14
    PLAYOUT_DOUBLING_ADVANTAGE_FLAG = 15
    PLAYOUT_DOUBLING_ADVANTAGE_VALUE = 16
    HAS_BUTTON = 17
    BOARD_SIZE_KOMI_PARITY = 18

class DotsSpatialFeature(Enum):
    ON_BOARD = 0
    PLA_ACTIVE = auto()
    OPP_ACTIVE = auto()
    PLA_PLACED = auto()
    OPP_PLACED = auto()
    DEAD = auto()
    GROUNDED = auto()
    PLA_CAPTURES = auto()
    OPP_CAPTURES = auto()
    PLA_SURROUNDINGS = auto()
    OPP_SURROUNDINGS = auto()
    PREV_1_LOC = auto()
    PREV_2_LOC = auto()
    PREV_3_LOC = auto()
    PREV_4_LOC = auto()
    PREV_5_LOC = auto()
    LADDER_CAPTURED = auto()
    LADDER_CAPTURED_PREVIOUS_1 = auto()
    LADDER_CAPTURED_PREVIOUS_2 = auto()
    LADDER_WORKING_MOVES = auto()

def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)

    is_go_game = not modelconfigs.is_dots_game(model_config)
    
    prev_1_loc = GoSpatialFeature.PREV_1_LOC.value if is_go_game else DotsSpatialFeature.PREV_1_LOC.value
    prev_2_loc = GoSpatialFeature.PREV_2_LOC.value if is_go_game else DotsSpatialFeature.PREV_2_LOC.value
    prev_3_loc = GoSpatialFeature.PREV_3_LOC.value if is_go_game else DotsSpatialFeature.PREV_3_LOC.value
    prev_4_loc = GoSpatialFeature.PREV_4_LOC.value if is_go_game else DotsSpatialFeature.PREV_4_LOC.value
    prev_5_loc = GoSpatialFeature.PREV_5_LOC.value if is_go_game else DotsSpatialFeature.PREV_5_LOC.value

    ladder_captured = GoSpatialFeature.LADDER_CAPTURED.value if is_go_game else DotsSpatialFeature.LADDER_CAPTURED.value
    ladder_captured_previous_1 = GoSpatialFeature.LADDER_CAPTURED_PREVIOUS_1.value if is_go_game else DotsSpatialFeature.LADDER_CAPTURED_PREVIOUS_1.value
    ladder_captured_previous_2 = GoSpatialFeature.LADDER_CAPTURED_PREVIOUS_2.value if is_go_game else DotsSpatialFeature.LADDER_CAPTURED_PREVIOUS_2.value
    ladder_working_moves = GoSpatialFeature.LADDER_WORKING_MOVES.value if is_go_game else DotsSpatialFeature.LADDER_WORKING_MOVES.value

    data = [1.0 for _ in range(num_bin_features)]

    data[prev_1_loc] = 0.0
    data[prev_2_loc] = 0.0
    data[prev_3_loc] = 0.0
    data[prev_4_loc] = 0.0
    data[prev_5_loc] = 0.0

    data[ladder_captured] = 1.0
    data[ladder_captured_previous_1] = 0.0
    data[ladder_captured_previous_2] = 0.0

    h_base = torch.diag(torch.tensor(data, device=device, requires_grad=False))

    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[ladder_captured, ladder_captured_previous_1] = 1.0
    h_base[ladder_captured, ladder_captured_previous_2] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[prev_1_loc, prev_1_loc] = 1.0  # Enable 9 -> 9
    h0[ladder_captured, ladder_captured_previous_1] = -1.0  # Stop copying 14 -> 15
    h0[ladder_captured, ladder_captured_previous_2] = -1.0  # Stop copying 14 -> 16
    h0[ladder_captured_previous_1, ladder_captured_previous_1] = 1.0  # Enable 15 -> 15
    h0[ladder_captured_previous_1, ladder_captured_previous_2] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[prev_2_loc, prev_2_loc] = 1.0  # Enable 10 -> 10
    h1[ladder_captured_previous_1, ladder_captured_previous_2] = -1.0  # Stop copying 15 -> 16
    h1[ladder_captured_previous_2, ladder_captured_previous_2] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[prev_3_loc, prev_3_loc] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[prev_4_loc, prev_4_loc] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[prev_5_loc, prev_5_loc] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC

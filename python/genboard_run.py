#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import logging
import json
import math
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from katago.game.board import Board, IllegalMoveError
from genboard_common import Model

if __name__ == '__main__':

    description = """
    Generate completions of Go positions
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-model', help='Model file to load', required=True)
    parser.add_argument('-board', help='Board pattern using {.,*},X,O,? for empty, black, white, unknown', required=True)
    parser.add_argument('-turn', help='Approx turn number to tell the net to generate for, [0,300]', required=True, type=float)
    parser.add_argument('-turnstdev', help='Approx turn number randomness [0,100]', required=True, type=float)
    parser.add_argument('-source', help='Tell the net to mimic positions from source, {-1,0,1}', required=True, type=int)
    parser.add_argument('-verbose', help='Print various info and debug messages instead of only the board', required=False, action='store_true')
    parser.add_argument('-n', help='How many batches to generate, default 1', required=False, type=int, default=1)
    parser.add_argument('-batchsize', help='How many positions to generate, default 1', required=False, type=int, default=1)
    args = vars(parser.parse_args())

    modelfile = args["model"]
    boardstr = args["board"]
    turn = args["turn"]
    turnstdev = args["turnstdev"]
    source = args["source"]
    verbose = args["verbose"]
    numbatches = args["n"]
    batchsize = args["batchsize"]

    if turn < 0 or turn > 300:
        raise Exception("Turn must be in [0,300]")
    if turnstdev < 0 or turnstdev > 100:
        raise Exception("Turn must be in [0,100]")
    if source != -1 and source != 0 and source != 1:
        raise Exception("Source must be in {-1,0,1}")
    if numbatches < 0:
        raise Exception("Num batches must be nonnegative")
    if batchsize < 1:
        raise Exception("Batchsize must be positive")

    cpudevice = torch.device("cpu")
    if torch.cuda.is_available():
        if verbose:
            print("CUDA is available, using it",flush=True)
        gpudevice = torch.device("cuda:0")
    else:
        gpudevice = cpudevice
    model = Model.load_from_file(modelfile).to(gpudevice)

    size = 19
    boardbase = [["." for x in range(size)] for y in range(size)]
    boardbase[3][3] = ","
    boardbase[9][3] = ","
    boardbase[15][3] = ","
    boardbase[3][9] = ","
    boardbase[9][9] = ","
    boardbase[15][9] = ","
    boardbase[3][15] = ","
    boardbase[9][15] = ","
    boardbase[15][15] = ","

    num_channels = 8
    inputsbase = torch.zeros((1,num_channels,size,size))

    inference_point_channel = 0
    black_channel = 2
    white_channel = 3
    unknown_channel = 4

    # Channel 1: On-board
    inputsbase[:,1,:,:].fill_(1.0)

    def fail_if_idx_too_large(idx):
        if idx >= size * size:
            raise Exception("Provided board is larger than 19x19")

    idx = 0
    for c in boardstr:
        y = idx // 19
        x = idx % 19
        if c == "." or c == "*" or c == ",":
            fail_if_idx_too_large(idx)
        elif c == "X" or c == "x" or c == "B" or c == "b":
            fail_if_idx_too_large(idx)
            boardbase[y][x] = "X"
            inputsbase[0,black_channel,y,x] = 1.0
        elif c == "O" or c == "o" or c == "W" or c == "w":
            fail_if_idx_too_large(idx)
            boardbase[y][x] = "O"
            inputsbase[0,white_channel,y,x] = 1.0
        elif c == "?":
            fail_if_idx_too_large(idx)
            inputsbase[0,unknown_channel,y,x] = 1.0
        else:
            # Ignore this char, counteract the += 1 at the end
            idx -= 1
        idx += 1

    # Channel 5: Turn number / 100
    inputsbase[:,5,:,:].fill_(turn / 100.0)
    # Channel 6: Noise stdev in turn number / 50
    inputsbase[:,6,:,:].fill_(turnstdev / 50.0)
    # Channel 7: Source
    inputsbase[:,7,:,:].fill_(float(source))

    rand = random.Random(os.urandom(32) + hashlib.md5(boardstr.encode()).hexdigest().encode())

    with torch.no_grad():

        for i in range(numbatches):

            flipx = rand.random() < 0.5
            flipy = rand.random() < 0.5
            swapxy = rand.random() < 0.5

            flipx2 = rand.random() < 0.5
            flipy2 = rand.random() < 0.5
            swapxy2 = rand.random() < 0.5

            def query_model(inputs):
                inputstransformed = inputs.detach().clone()
                if flipx:
                    if flipy:
                        inputstransformed = torch.flip(inputstransformed,[2,3])
                    else:
                        inputstransformed = torch.flip(inputstransformed,[2])
                else:
                    if flipx:
                        inputstransformed = torch.flip(inputstransformed,[3])
                    else:
                        pass
                if swapxy:
                    inputstransformed = torch.transpose(inputstransformed,2,3)

                preds, auxpreds = model(inputstransformed.to(gpudevice))
                preds = F.softmax(preds,dim=1)
                assert(len(preds.size()) == 2)
                assert(preds.size()[0] == batchsize)
                assert(preds.size()[1] == 3)
                choices = []
                for b in range(batchsize):
                    weights = [preds[b,0],preds[b,1],preds[b,2]]
                    choice = rand.choices([0,1,2],weights=weights)[0]
                    choices.append(choice)
                return choices

            inputs = inputsbase.expand([batchsize,-1,-1,-1]).detach().clone()
            boards = [ deepcopy(boardbase) for b in range(batchsize) ]

            for y in range(size):
                for x in range(size):
                    sx = x
                    sy = y
                    if flipx2:
                        sx = size - sx - 1
                    if flipy2:
                        sy = size - sy - 1
                    if swapxy2:
                        tmp = sx
                        sx = sy
                        sy = tmp

                    if inputs[0,unknown_channel,sy,sx] == 1.0:
                        for b in range(batchsize):
                            inputs[b,unknown_channel,sy,sx] = 0.0
                            inputs[b,inference_point_channel,sy,sx] = 1.0
                        choices = query_model(inputs)
                        for b in range(batchsize):
                            inputs[b,inference_point_channel,sy,sx] = 0.0

                            choice = choices[b]
                            if choice == 0:
                                pass
                            elif choice == 1:
                                inputs[b,black_channel,sy,sx] = 1.0
                                boards[b][sy][sx] = "X"
                            elif choice == 2:
                                inputs[b,white_channel,sy,sx] = 1.0
                                boards[b][sy][sx] = "O"

            for b in range(batchsize):
                s = "\n".join([" ".join(row) for row in boards[b]])
                s += "\n"
                print(s)
            sys.stdout.flush()

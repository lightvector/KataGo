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

import data
from board import Board, IllegalMoveError
from genboard_common import Model

class ShuffledDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset, shuffle_buffer_size):
    super().__init__()
    self.dataset = dataset
    self.shuffle_buffer_size = shuffle_buffer_size

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      rand = random.Random(os.urandom(32))
    else:
      rand = random.Random(os.urandom(32)+ "#ShuffledDataset#".encode() + str(worker_info.id).encode())

    shuffle_buffer = []
    try:
      it = iter(self.dataset)
      while len(shuffle_buffer) < self.shuffle_buffer_size:
        item = next(it)
        if isinstance(item, Exception):
          yield item
        else:
          shuffle_buffer.append(item)
    except StopIteration:
      self.shuffle_buffer_size = len(shuffle_buffer)

    print("Initial shuffle buffer filled", flush=True)
    rand.shuffle(shuffle_buffer)
    try:
      while True:
        try:
          item = next(it)
          if isinstance(item, Exception):
            yield item
          else:
            idx = rand.randint(0, self.shuffle_buffer_size-1)
            old_item = shuffle_buffer[idx]
            shuffle_buffer[idx] = item
            yield old_item
        except StopIteration:
          break
      while len(shuffle_buffer) > 0:
        yield shuffle_buffer.pop()
    except GeneratorExit:
      pass

def rand_triangular(rand,maxvalue):
  r = (maxvalue+1) * (1.0 - math.sqrt(rand.random()))
  r = int(math.floor(r))
  if r <= 0:
    return 0
  if r >= maxvalue:
    return maxvalue
  return r

def random_subinterval(rand,size):
  # Anchor rectangles near the edge more often
  if rand.random() < 0.5:
    x0 = rand_triangular(rand,size)-1
    x1 = rand_triangular(rand,size)-1
  else:
    x0 = rand.randint(0,size-1)
    x1 = rand.randint(0,size-1)

  if rand.random() < 0.5:
    x0 = size - x0 - 1
    x1 = size - x1 - 1

  if x0 > x1:
    return (x1,x0)
  return (x0,x1)


class SgfDataset(torch.utils.data.IterableDataset):
  def __init__(self, files, max_turn, break_prob_per_turn, sample_prob, endless):
    self.files = files
    self.max_turn = max_turn
    self.break_prob_per_turn = break_prob_per_turn
    self.sample_prob = sample_prob
    self.endless = endless

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      rand = random.Random(os.urandom(32))
    else:
      rand = random.Random(os.urandom(32)+ "#SgfDataset#".encode() + str(worker_info.id).encode())

    files = self.files
    cpudevice = torch.device("cpu")

    try:
      while True:
        rand.shuffle(files)
        file_count = 0
        print("Iterator beginning reading of files %d / %d" % (file_count, len(files)), flush=True)
        for filename in files:
          (metadata,setup,moves,rules) = data.load_sgf_moves_exn(filename)
          # Only even 19x19 games!
          if metadata.size != 19 or len(setup) != 0 or (metadata.handicap is not None and metadata.handicap != 0):
            continue
          board = Board(size=metadata.size)
          turn_number = 0
          for (pla,loc) in moves:

            if rand.random() < self.sample_prob:
              inputs = torch.zeros((8,metadata.size,metadata.size),dtype=torch.float32,device=cpudevice)
              result = torch.zeros((3,),dtype=torch.float32,device=cpudevice)
              aux = torch.zeros((3,metadata.size,metadata.size),dtype=torch.float32,device=cpudevice)

              (alwaysknownxmin,alwaysknownxmax) = random_subinterval(rand,metadata.size)
              (alwaysknownymin,alwaysknownymax) = random_subinterval(rand,metadata.size)

              if alwaysknownxmin <= 0 and alwaysknownxmax >= metadata.size-1 and alwaysknownymin <= 0 and alwaysknownymax >= metadata.size-1:
                pass
              else:
                # Channel 1: On-board
                inputs[1,:,:].fill_(1.0)

                num_always_known_poses = 0
                if alwaysknownxmax < 0 or alwaysknownxmin >= metadata.size or alwaysknownymax < 0 or alwaysknownymin >= metadata.size:
                  num_always_known_poses = 0
                else:
                  num_always_known_poses = (
                    ( min(alwaysknownxmax, metadata.size-1) - max(alwaysknownxmin, 0) + 1) *
                    ( min(alwaysknownymax, metadata.size-1) - max(alwaysknownymin, 0) + 1)
                  )
                num_not_always_known_poses = metadata.size * metadata.size - num_always_known_poses
                inferenceidx = rand.randint(0,num_not_always_known_poses-1)

                flipx = rand.random() < 0.5
                flipy = rand.random() < 0.5
                swapxy = rand.random() < 0.5

                idx = 0
                for y in range(metadata.size):
                  for x in range(metadata.size):
                    pos = y * metadata.size + x
                    always_known = (x >= alwaysknownxmin and x <= alwaysknownxmax and y >= alwaysknownymin and y <= alwaysknownymax)

                    sx = x
                    sy = y
                    if flipx:
                      sx = metadata.size - sx - 1
                    if flipy:
                      sy = metadata.size - sy - 1
                    if swapxy:
                      tmp = sx
                      sx = sy
                      sy = tmp
                    stone = board.board[board.loc(sx,sy)]

                    # Channel 4: Unknown
                    if idx > inferenceidx and not always_known:
                      inputs[4,y,x] = 1.0
                    # Channel 0: Next inference point
                    elif idx == inferenceidx and not always_known:
                      inputs[0,y,x] = 1.0
                      result
                      if stone == Board.BLACK:
                        result[1] = 1.0
                      elif stone == Board.WHITE:
                        result[2] = 1.0
                      else:
                        result[0] = 1.0
                    else:
                      # Channel 2: Black
                      if stone == Board.BLACK:
                        inputs[2,y,x] = 1.0
                      # Channel 3: White
                      elif stone == Board.WHITE:
                        inputs[3,y,x] = 1.0

                    if stone == Board.BLACK:
                      aux[1,y,x] = 1.0
                    elif stone == Board.WHITE:
                      aux[2,y,x] = 1.0
                    else:
                      aux[0,y,x] = 1.0

                    if not always_known:
                      idx += 1

                assert(idx == num_not_always_known_poses)

                if rand.random() < 0.3:
                  turn_noise_stdev = 0.0
                  reported_turn = turn_number
                else:
                  turn_noise_stdev = (rand.random() ** 2.0) * 100
                  reported_turn = turn_number + rand.normalvariate(0.0,turn_noise_stdev)

                # Channel 5: Turn number / 100
                inputs[5,:,:].fill_(reported_turn / 100.0)
                # Channel 6: Noise stdev in turn number / 50
                inputs[6,:,:].fill_(turn_noise_stdev / 50.0)
                # Channel 7: Source
                is_kgs = ("/kgs" in filename) or ("\\KGS" in filename) or ("/KGS" in filename) or ("\\KGS" in filename)
                is_fox = ("/fox" in filename) or ("\\fox" in filename) or ("/FOX" in filename) or ("\\FOX" in filename)
                if is_kgs:
                  inputs[7,:,:].fill_(1.0)
                elif is_fox:
                  inputs[7,:,:].fill_(-1.0)

                if rand.random() < 0.5:
                  if rand.random() < 0.5:
                    inputs = torch.flip(inputs,[1,2])
                    aux = torch.flip(aux,[1,2])
                  else:
                    inputs = torch.flip(inputs,[1])
                    aux = torch.flip(aux,[1])
                else:
                  if rand.random() < 0.5:
                    inputs = torch.flip(inputs,[2])
                    aux = torch.flip(aux,[2])
                  else:
                    pass

                if rand.random() < 0.5:
                  inputs = torch.transpose(inputs,1,2)
                  aux = torch.transpose(aux,1,2)

                yield (inputs,result,aux)

            try:
              board.play(pla,loc)
            except IllegalMoveError as e:
              # On illegal move in the SGF, don't attempt to recover, just move on to new game
              print("Illegal move, skipping file " + filename + ":" + str(e), flush=True)
              break
            turn_number += 1
            if turn_number > self.max_turn:
              break
            if rand.random() < self.break_prob_per_turn:
              break

          file_count += 1
          if file_count % 200 == 0:
            print("Read through file %d / %d" % (file_count, len(files)), flush=True)

        if not self.endless:
          break

    except GeneratorExit:
      pass
    except Exception as e:
      print("EXCEPTION IN GENERATOR: " + str(e))
      traceback.print_exc()
      print("---",flush=True)
      yield e


def save_json(data,filename):
  with open(filename,"w") as f:
    json.dump(data,f)
    f.flush()
    os.fsync(f.fileno())

def load_json(filename):
  with open(filename) as f:
    data = json.load(f)
  return data


if __name__ == '__main__':

  description = """
  Train net to predict Go positions one stone at a time
  """

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
  parser.add_argument('-datadirs', help='Directory with sgfs', required=True)
  parser.add_argument('-testprop', help='Proportion of data for test', type=float, required=True)
  parser.add_argument('-lr-scale', help='LR multiplier', type=float, required=False)
  parser.add_argument('-grad-clip-scale', help='Gradient clip multiplier', type=float, required=False)
  parser.add_argument('-num-data-workers', help='Number of processes for data loading', type=int, required=False)
  args = vars(parser.parse_args())

  traindir = args["traindir"]
  datadirs = args["datadirs"]
  testprop = args["testprop"]
  lr_scale = args["lr_scale"]
  grad_clip_scale = args["grad_clip_scale"]
  num_data_workers = args["num_data_workers"]
  logfilemode = "a"

  if lr_scale is None:
    lr_scale = 1.0
  if grad_clip_scale is None:
    grad_clip_scale = 1.0

  if num_data_workers is None:
    num_data_workers = 0

  if not os.path.exists(traindir):
    os.mkdir(traindir)

  bareformatter = logging.Formatter("%(asctime)s %(message)s")
  fh = logging.FileHandler(os.path.join(traindir,"train.log"), mode=logfilemode)
  fh.setFormatter(bareformatter)
  stdouthandler = logging.StreamHandler(sys.stdout)
  stdouthandler.setFormatter(bareformatter)
  trainlogger = logging.getLogger("trainlogger")
  trainlogger.setLevel(logging.INFO)
  trainlogger.addHandler(fh)
  trainlogger.addHandler(stdouthandler)
  trainlogger.propagate=False
  np.set_printoptions(linewidth=150)
  def trainlog(s):
    trainlogger.info(s)
    sys.stdout.flush()

  shuffle_buffer_size = 100000

  trainfiles = []
  testfiles = []
  for datadir in datadirs.split(","):
    for parent, subdirs, files in os.walk(datadir):
      for name in files:
        if name.endswith(".sgf"):
          r = float.fromhex("0."+hashlib.md5(os.path.join(parent,name).encode()).hexdigest()[:16])
          if r < testprop:
            testfiles.append(os.path.join(parent,name))
          else:
            trainfiles.append(os.path.join(parent,name))

  trainlog("Found %d training sgfs" % len(trainfiles))
  trainlog("Found %d testing sgfs" % len(testfiles))

  max_turn = 300
  break_prob_per_turn = 0.01

  traindataset = ShuffledDataset(SgfDataset(trainfiles,max_turn,break_prob_per_turn,sample_prob=0.5,endless=True),shuffle_buffer_size)
  testdataset = SgfDataset(testfiles,max_turn,break_prob_per_turn,sample_prob=0.2,endless=True)

  batch_size = 128
  trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers, drop_last=True)
  testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers, drop_last=True)

  trainlog("Made data loaders")

  samples_per_epoch = 400000
  samples_per_test = 25600
  batches_per_epoch = samples_per_epoch // batch_size
  batches_per_test = samples_per_test // batch_size

  def lossfunc(inputs, results, preds, aux, auxpreds):
    assert(preds.size()[1] == 3)
    assert(auxpreds.size()[1] == 3)
    main_loss = -torch.sum(results * F.log_softmax(preds,dim=1))
    aux_loss = -torch.sum(aux * F.log_softmax(auxpreds,dim=1) * inputs[:,4:5,:,:] / torch.sum(inputs[:,1:2,:,:], dim=[2,3], keepdim=True)) * 0.3
    return main_loss, aux_loss

  cpudevice = torch.device("cpu")
  if torch.cuda.is_available():
    trainlog("CUDA is available, using it")
    gpudevice = torch.device("cuda:0")
  else:
    gpudevice = cpudevice

  modelpath = os.path.join(traindir,"model.data")
  optimpath = os.path.join(traindir,"optim.data")
  traindatapath = os.path.join(traindir,"traindata.json")
  if os.path.exists(modelpath):
    trainlog("Loading preexisting model!")
    model = Model.load_from_file(modelpath).to(gpudevice)
    optimizer = optim.SGD(model.parameters(), lr=0.00001*lr_scale, momentum=0.9)
    optimizer.load_state_dict(torch.load(optimpath))
    traindata = load_json(traindatapath)
  else:
    num_channels = 96
    num_blocks = 8
    model = Model(num_channels=num_channels, num_blocks=num_blocks).to(gpudevice)
    optimizer = optim.SGD(model.parameters(), lr=0.00001*lr_scale, momentum=0.9)
    traindata = {"samples_so_far":0, "batches_so_far":0}

    trainlog("Saving!")
    model.save_to_file(modelpath)
    torch.save(optimizer.state_dict(), optimpath)
    save_json(traindata,traindatapath)

  grad_clip_max = 400 * grad_clip_scale
  #Loosen gradient clipping as we shift to smaller learning rates
  grad_clip_max = grad_clip_max / math.sqrt(lr_scale)

  running_batch_count = 0
  running_main_loss = 0.0
  running_aux_loss = 0.0
  running_gnorm = 0.0
  running_ewms_exgnorm = 0.0
  print_every_batches = 100
  trainiter = iter(trainloader)
  testiter = iter(testloader)
  while True:
    for i in range(batches_per_epoch):
      inputs, results, auxs = next(trainiter)
      inputs = inputs.to(gpudevice)
      results = results.to(gpudevice)
      auxs = auxs.to(gpudevice)

      optimizer.zero_grad()

      preds, auxpreds = model(inputs)
      main_loss,aux_loss = lossfunc(inputs, results, preds, auxs, auxpreds)
      loss = main_loss + aux_loss
      loss.backward()
      gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max)
      optimizer.step()

      traindata["samples_so_far"] += batch_size
      traindata["batches_so_far"] += 1

      running_batch_count += 1
      running_main_loss += main_loss.item()
      running_aux_loss += aux_loss.item()
      running_gnorm += gnorm
      running_ewms_exgnorm += max(0.0, gnorm - grad_clip_max)
      if running_batch_count >= print_every_batches:
        trainlog("TRAIN samples: %d,  batches: %d,  main loss: %.5f,  aux loss: %.5f,  gnorm: %.2f,  ewms_exgnorm: %.3g" % (
          traindata["samples_so_far"],
          traindata["batches_so_far"],
          running_main_loss / (running_batch_count * batch_size),
          running_aux_loss / (running_batch_count * batch_size),
          running_gnorm / (running_batch_count),
          running_ewms_exgnorm / (running_batch_count),
        ))
        running_batch_count = 0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        running_gnorm = 0.0
        running_ewms_exgnorm *= 0.5

    trainlog("Saving!")
    model.save_to_file(modelpath)
    torch.save(optimizer.state_dict(), optimpath)
    save_json(traindata,traindatapath)

    trainlog("Testing!")
    test_samples = 0
    test_main_loss = 0.0
    test_aux_loss = 0.0
    with torch.no_grad():
      for i in range(batches_per_test):
        inputs, results, auxs = next(testiter)
        inputs = inputs.to(gpudevice)
        results = results.to(gpudevice)
        auxs = auxs.to(gpudevice)

        preds, auxpreds = model(inputs)
        main_loss, aux_loss = lossfunc(inputs, results, preds, auxs, auxpreds)
        test_samples += batch_size
        test_main_loss += main_loss.item()
        test_aux_loss += aux_loss.item()
    trainlog("TEST samples %d,  main loss: %.5f,  aux loss %.5f" % (test_samples, test_main_loss / test_samples, test_aux_loss / test_samples))


trainlog('Finished Training')

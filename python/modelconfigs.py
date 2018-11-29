#!/usr/bin/python3
import sys
import os
import argparse
import random
import time
import logging
import json
import datetime
import math

import tensorflow as tf
import numpy as np

from modelv3 import ModelV3

b6c96 = {
  "trunk_num_channels":96,
  "mid_num_channels":96,
  "regular_num_channels":64,
  "dilated_num_channels":32,
  "gpool_num_channels":32,
  "block_kind": [
    ["rconv1","regular"],
    ["rconv2","regular"],
    ["rconv3","gpool"],
    ["rconv4","regular"],
    ["rconv5","gpool"],
    ["rconv6","regular"]
  ],
  "p1_num_channels":48,
  "g1_num_channels":32,
  "v1_num_channels":32,
  "sbv2_num_channels":48,
  "bbv2_num_channels":32,
  "v2_size":32
}

b10c128 = {
  "trunk_num_channels":128,
  "mid_num_channels":128,
  "regular_num_channels":96,
  "dilated_num_channels":32,
  "gpool_num_channels":32,
  "block_kind": [
    ["rconv1","regular"],
    ["rconv2","regular"],
    ["rconv3","regular"],
    ["rconv4","regular"],
    ["rconv5","gpool"],
    ["rconv6","regular"],
    ["rconv7","regular"],
    ["rconv8","gpool"],
    ["rconv9","regular"],
    ["rconv10","regular"]
  ],
  "p1_num_channels":48,
  "g1_num_channels":32,
  "v1_num_channels":32,
  "sbv2_num_channels":48,
  "bbv2_num_channels":32,
  "v2_size":32
}

b14c192 = {
  "trunk_num_channels":192,
  "mid_num_channels":192,
  "regular_num_channels":128,
  "dilated_num_channels":64,
  "gpool_num_channels":64,
  "block_kind": [
    ["rconv1","regular"],
    ["rconv2","regular"],
    ["rconv3","regular"],
    ["rconv4","regular"],
    ["rconv5","regular"],
    ["rconv6","regular"],
    ["rconv7","gpool"],
    ["rconv8","regular"],
    ["rconv9","regular"],
    ["rconv10","regular"],
    ["rconv11","gpool"],
    ["rconv12","regular"],
    ["rconv13","regular"],
    ["rconv14","regular"]
  ],
  "p1_num_channels":48,
  "g1_num_channels":32,
  "v1_num_channels":32,
  "sbv2_num_channels":48,
  "bbv2_num_channels":32,
  "v2_size":48
}

config_of_name = {
  "b6c96": b6c96,
  "b10c128": b10c128,
  "b14c192": b14c192
}

b6_to_b10_map = {
  "rconv1": ("regular","rconv1"),
  "rconv2": ("regular","rconv2"),
  "rconv3": ("regular","new"),
  "rconv4": ("regular","new"),
  "rconv5": ("gpool","rconv3"),
  "rconv6": ("regular","rconv4"),
  "rconv7": ("regular","new"),
  "rconv8": ("gpool","rconv5"),
  "rconv9": ("regular","rconv6"),
  "rconv10": ("regular","new"),
}


def truncated_normal(shape,num_inputs):
  num_weights = np.prod(shape)
  weights = np.zeros(num_weights)
  for i in range(num_weights):
    r = np.random.standard_normal()
    while r < -2.0 or r > 2.0:
      r = np.random.standard_normal()
    weights[i] = r
  weights = np.reshape(weights,shape)
  return weights * math.sqrt(2.0 / num_inputs)

def make_mapping_scaling(name,oldcfg,newcfg):
  oldc = oldcfg[name]
  newc = newcfg[name]
  assert(newc >= oldc)
  cmapping = np.zeros(newc,dtype=np.int32)
  for i in range(newc):
    if i < oldc:
      cmapping[i] = i
    else:
      cmapping[i] = np.random.randint(oldc)

  cscaling = np.ones(newc,dtype=np.float32)
  for i in range(newc):
    target = cmapping[i]
    cscaling[i] = 1.0 / (cmapping == target).sum()
  return (cmapping,cscaling)

def upgrade_mat_in_channels(oldweights,newweights,oldname,newname,cmapping,cscaling):
  old = newweights[newname] if newname in newweights else oldweights[oldname]
  assert(len(old.shape)==2)
  assert(len(cmapping) >= old.shape[0])
  assert(old.dtype == np.float32)
  #ic,oc
  new = np.zeros([len(cmapping),old.shape[1]],dtype=old.dtype)
  for ic in range(len(cmapping)):
    for oc in range(old.shape[1]):
      new[ic,oc] = cscaling[ic] * old[cmapping[ic],oc]
  newweights[newname] = new

def upgrade_triplemat_in_channels(oldweights,newweights,oldname,newname,cmapping,cscaling):
  old = newweights[newname] if newname in newweights else oldweights[oldname]
  assert(len(old.shape)==2)
  assert(old.shape[0] % 3 == 0)
  oldsectionlen = old.shape[0] // 3
  assert(len(cmapping) >= oldsectionlen)
  assert(old.dtype == np.float32)
  #ic,oc
  new = np.zeros([len(cmapping)*3,old.shape[1]],dtype=old.dtype)
  for section in range(3):
    for dic in range(len(cmapping)):
      newic = dic + section*len(cmapping)
      for oc in range(old.shape[1]):
        new[newic,oc] = cscaling[dic] * old[cmapping[dic]+section*oldsectionlen,oc]
  newweights[newname] = new

def upgrade_mat_out_channels(oldweights,newweights,oldname,newname,cmapping):
  old = newweights[newname] if newname in newweights else oldweights[oldname]
  assert(len(old.shape)==2)
  assert(len(cmapping) >= old.shape[1])
  assert(old.dtype == np.float32)
  #ic,oc
  new = np.zeros([old.shape[0],len(cmapping)],dtype=old.dtype)
  for ic in range(old.shape[0]):
    for oc in range(len(cmapping)):
      new[ic,oc] = old[ic,cmapping[oc]]
  newweights[newname] = new

def noise_mat(newweights,newname,magnitude):
  new = newweights[newname]
  newweights[newname] = new + truncated_normal(new.shape,new.shape[0])*magnitude

def upgrade_conv_in_channels(oldweights,newweights,oldname,newname,cmapping,cscaling):
  old = newweights[newname] if newname in newweights else oldweights[oldname]
  assert(len(old.shape)==4)
  assert(len(cmapping) >= old.shape[2])
  assert(old.dtype == np.float32)
  #dx,dy,ic,oc
  new = np.zeros([old.shape[0],old.shape[1],len(cmapping),old.shape[3]],dtype=old.dtype)
  for dx in range(old.shape[0]):
    for dy in range(old.shape[1]):
      for ic in range(len(cmapping)):
        for oc in range(old.shape[3]):
          new[dx,dy,ic,oc] = cscaling[ic] * old[dx,dy,cmapping[ic],oc]
  newweights[newname] = new

def upgrade_conv_out_channels(oldweights,newweights,oldname,newname,cmapping):
  old = newweights[newname] if newname in newweights else oldweights[oldname]
  assert(len(old.shape)==4)
  assert(len(cmapping) >= old.shape[3])
  assert(old.dtype == np.float32)
  #dx,dy,ic,oc
  new = np.zeros([old.shape[0],old.shape[1],old.shape[2],len(cmapping)],dtype=old.dtype)
  for dx in range(old.shape[0]):
    for dy in range(old.shape[1]):
      for ic in range(old.shape[2]):
        for oc in range(len(cmapping)):
          new[dx,dy,ic,oc] = old[dx,dy,ic,cmapping[oc]]
  newweights[newname] = new

def noise_conv(newweights,newname,magnitude):
  new = newweights[newname]
  newweights[newname] = new + truncated_normal(new.shape,new.shape[0]*new.shape[1]*new.shape[2])*magnitude

def upgrade_beta(oldweights,newweights,oldname,newname,cmapping):
  old = newweights[newname] if newname in newweights else oldweights[oldname]
  assert(len(old.shape)==1)
  assert(len(cmapping) >= old.shape[0])
  assert(old.dtype == np.float32)
  #oc
  new = np.zeros([len(cmapping)],dtype=old.dtype)
  for oc in range(len(cmapping)):
    new[oc] = old[cmapping[oc]]
  newweights[newname] = new

def upgrade_residual_block(oldweights,newweights,oldname,newname,trunk_cmapping,trunk_cscaling,oldcfg,newcfg,noise_mag):
  (mid_cmapping,mid_cscaling) = make_mapping_scaling("mid_num_channels",oldcfg,newcfg)

  upgrade_beta(oldweights,newweights,oldname+"/norm1/beta:0",newname+"/norm1/beta:0",trunk_cmapping)

  upgrade_conv_in_channels(oldweights,newweights,oldname+"/w1:0",newname+"/w1:0",trunk_cmapping,trunk_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,oldname+"/w1:0",newname+"/w1:0",mid_cmapping)
  noise_conv(newweights,newname+"/w1:0",noise_mag)

  upgrade_beta(oldweights,newweights,oldname+"/norm2/beta:0",newname+"/norm2/beta:0",mid_cmapping)

  upgrade_conv_in_channels(oldweights,newweights,oldname+"/w2:0",newname+"/w2:0",mid_cmapping,mid_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,oldname+"/w2:0",newname+"/w2:0",trunk_cmapping)
  noise_conv(newweights,newname+"/w2:0",noise_mag)

def upgrade_gpool_block(oldweights,newweights,oldname,newname,trunk_cmapping,trunk_cscaling,oldcfg,newcfg,noise_mag):
  (regular_cmapping,regular_cscaling) = make_mapping_scaling("regular_num_channels",oldcfg,newcfg)
  (gpool_cmapping,gpool_cscaling) = make_mapping_scaling("gpool_num_channels",oldcfg,newcfg)

  upgrade_beta(oldweights,newweights,oldname+"/norm1/beta:0",newname+"/norm1/beta:0",trunk_cmapping)

  upgrade_conv_in_channels(oldweights,newweights,oldname+"/w1a:0",newname+"/w1a:0",trunk_cmapping,trunk_cscaling)
  upgrade_conv_in_channels(oldweights,newweights,oldname+"/w1b:0",newname+"/w1b:0",trunk_cmapping,trunk_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,oldname+"/w1a:0",newname+"/w1a:0",regular_cmapping)
  upgrade_conv_out_channels(oldweights,newweights,oldname+"/w1b:0",newname+"/w1b:0",gpool_cmapping)
  noise_conv(newweights,newname+"/w1a:0",noise_mag)
  noise_conv(newweights,newname+"/w1b:0",noise_mag)

  upgrade_beta(oldweights,newweights,oldname+"/norm1b/beta:0",newname+"/norm1b/beta:0",gpool_cmapping)

  upgrade_triplemat_in_channels(oldweights,newweights,oldname+"/w1r:0",newname+"/w1r:0",gpool_cmapping,gpool_cscaling)
  upgrade_mat_out_channels(oldweights,newweights,oldname+"/w1r:0",newname+"/w1r:0",regular_cmapping)
  noise_mat(newweights,newname+"/w1r:0",noise_mag)

  upgrade_beta(oldweights,newweights,oldname+"/norm2/beta:0",newname+"/norm2/beta:0",regular_cmapping)

  upgrade_conv_in_channels(oldweights,newweights,oldname+"/w2:0",newname+"/w2:0",regular_cmapping,regular_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,oldname+"/w2:0",newname+"/w2:0",trunk_cmapping)
  noise_conv(newweights,newname+"/w2:0",noise_mag)

def new_residual_block(newweights,newname,newcfg,noise_mag):
  trunk_num_channels = newcfg["trunk_num_channels"]
  mid_num_channels = newcfg["mid_num_channels"]
  newweights[newname+"/norm1/beta:0"] = np.zeros(trunk_num_channels)
  newweights[newname+"/w1:0"] = truncated_normal([3,3,trunk_num_channels,mid_num_channels],3*3*trunk_num_channels)*noise_mag
  newweights[newname+"/norm2/beta:0"] = np.zeros(mid_num_channels)
  newweights[newname+"/w2:0"] = np.zeros([3,3,mid_num_channels,trunk_num_channels])


def upgrade_net(oldweights,newweights,oldcfg,newcfg,block_map,noise_mag):
  (trunk_cmapping,trunk_cscaling) = make_mapping_scaling("trunk_num_channels",oldcfg,newcfg)

  upgrade_conv_out_channels(oldweights,newweights,"conv1/w:0","conv1/w:0",trunk_cmapping)
  upgrade_mat_out_channels(oldweights,newweights,"ginputw:0","ginputw:0",trunk_cmapping)
  noise_conv(newweights,"conv1/w:0",noise_mag)
  noise_mat(newweights,"ginputw:0",noise_mag)

  for newname in block_map:
    (kind,oldname) = block_map[newname]
    if kind == "regular":
      if oldname == "new":
        new_residual_block(newweights,newname,newcfg,noise_mag=1.0)
      else:
        upgrade_residual_block(oldweights,newweights,oldname,newname,trunk_cmapping,trunk_cscaling,oldcfg,newcfg,noise_mag)
    elif kind == "gpool":
      if oldname == "new":
        assert(False) # Not implemented
      else:
        upgrade_gpool_block(oldweights,newweights,oldname,newname,trunk_cmapping,trunk_cscaling,oldcfg,newcfg,noise_mag)
    else:
      assert(False) # No other block types implemented

  upgrade_beta(oldweights,newweights,"trunk/norm/beta:0","trunk/norm/beta:0",trunk_cmapping)

  (p1_cmapping,p1_cscaling) = make_mapping_scaling("p1_num_channels",oldcfg,newcfg)
  (g1_cmapping,g1_cscaling) = make_mapping_scaling("g1_num_channels",oldcfg,newcfg)

  upgrade_conv_in_channels(oldweights,newweights,"p1/intermediate_conv/w:0","p1/intermediate_conv/w:0",trunk_cmapping,trunk_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,"p1/intermediate_conv/w:0","p1/intermediate_conv/w:0",p1_cmapping)
  noise_conv(newweights,"p1/intermediate_conv/w:0",noise_mag)

  upgrade_conv_in_channels(oldweights,newweights,"g1/w:0","g1/w:0",trunk_cmapping,trunk_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,"g1/w:0","g1/w:0",g1_cmapping)
  noise_conv(newweights,"g1/w:0",noise_mag)

  upgrade_beta(oldweights,newweights,"g1/norm/beta:0","g1/norm/beta:0",g1_cmapping)
  upgrade_triplemat_in_channels(oldweights,newweights,"matmulg2w:0","matmulg2w:0",g1_cmapping,g1_cscaling)
  upgrade_mat_out_channels(oldweights,newweights,"matmulg2w:0","matmulg2w:0",p1_cmapping)
  noise_mat(newweights,"matmulg2w:0",noise_mag)

  upgrade_beta(oldweights,newweights,"p1/norm/beta:0","p1/norm/beta:0",p1_cmapping)
  upgrade_conv_in_channels(oldweights,newweights,"p2/w:0","p2/w:0",p1_cmapping,p1_cscaling)
  noise_conv(newweights,"p2/w:0",noise_mag)

  upgrade_triplemat_in_channels(oldweights,newweights,"matmulpass:0","matmulpass:0",g1_cmapping,g1_cscaling)
  noise_mat(newweights,"matmulpass:0",noise_mag)

  (v1_cmapping,v1_cscaling) = make_mapping_scaling("v1_num_channels",oldcfg,newcfg)
  upgrade_conv_in_channels(oldweights,newweights,"v1/w:0","v1/w:0",trunk_cmapping,trunk_cscaling)
  upgrade_conv_out_channels(oldweights,newweights,"v1/w:0","v1/w:0",v1_cmapping)
  noise_conv(newweights,"v1/w:0",noise_mag)
  upgrade_beta(oldweights,newweights,"v1/norm/beta:0","v1/norm/beta:0",v1_cmapping)

  (v2_cmapping,v2_cscaling) = make_mapping_scaling("v2_size",oldcfg,newcfg)
  upgrade_triplemat_in_channels(oldweights,newweights,"v2/w:0","v2/w:0",v1_cmapping,v1_cscaling)
  upgrade_mat_out_channels(oldweights,newweights,"v2/w:0","v2/w:0",v2_cmapping)
  noise_mat(newweights,"v2/w:0",noise_mag)
  upgrade_beta(oldweights,newweights,"v2/b:0","v2/b:0",v2_cmapping)

  v3_cmapping = np.array([0,1,2],dtype=np.int32)
  upgrade_mat_in_channels(oldweights,newweights,"v3/w:0","v3/w:0",v2_cmapping,v2_cscaling)
  noise_mat(newweights,"v3/w:0",noise_mag)
  upgrade_beta(oldweights,newweights,"v3/b:0","v3/b:0",v3_cmapping)

  mv3_cmapping = np.array([0,1,2,3,4],dtype=np.int32)
  upgrade_mat_in_channels(oldweights,newweights,"mv3/w:0","mv3/w:0",v2_cmapping,v2_cscaling)
  noise_mat(newweights,"mv3/w:0",noise_mag)
  upgrade_beta(oldweights,newweights,"mv3/b:0","mv3/b:0",mv3_cmapping)

  (sb2_cmapping,sb2_cscaling) = make_mapping_scaling("sbv2_num_channels",oldcfg,newcfg)
  upgrade_triplemat_in_channels(oldweights,newweights,"sb2/w:0","sb2/w:0",v1_cmapping,v1_cscaling)
  upgrade_mat_out_channels(oldweights,newweights,"sb2/w:0","sb2/w:0",sb2_cmapping)
  noise_mat(newweights,"sb2/w:0",noise_mag)
  upgrade_beta(oldweights,newweights,"sb2/b:0","sb2/b:0",sb2_cmapping)
  upgrade_mat_in_channels(oldweights,newweights,"sb2_offset/w:0","sb2_offset/w:0",[0],[1.0])
  upgrade_mat_out_channels(oldweights,newweights,"sb2_offset/w:0","sb2_offset/w:0",sb2_cmapping)
  upgrade_mat_in_channels(oldweights,newweights,"sb3/w:0","sb3/w:0",sb2_cmapping,sb2_cscaling)
  noise_mat(newweights,"sb3/w:0",noise_mag)

  (bb2_cmapping,bb2_cscaling) = make_mapping_scaling("bbv2_num_channels",oldcfg,newcfg)
  upgrade_triplemat_in_channels(oldweights,newweights,"bb2/w:0","bb2/w:0",v1_cmapping,v1_cscaling)
  upgrade_mat_out_channels(oldweights,newweights,"bb2/w:0","bb2/w:0",bb2_cmapping)
  noise_mat(newweights,"bb2/w:0",noise_mag)
  upgrade_beta(oldweights,newweights,"bb2/b:0","bb2/b:0",bb2_cmapping)
  upgrade_mat_in_channels(oldweights,newweights,"bb2_offset/w:0","bb2_offset/w:0",[0],[1.0])
  upgrade_mat_out_channels(oldweights,newweights,"bb2_offset/w:0","bb2_offset/w:0",bb2_cmapping)
  upgrade_mat_in_channels(oldweights,newweights,"bb3/w:0","bb3/w:0",bb2_cmapping,bb2_cscaling)
  noise_mat(newweights,"bb3/w:0",noise_mag)

  upgrade_conv_in_channels(oldweights,newweights,"vownership/w:0","vownership/w:0",v1_cmapping,v1_cscaling)
  noise_conv(newweights,"vownership/w:0",noise_mag)

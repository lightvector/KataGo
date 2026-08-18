#!/usr/bin/env python3
# parse_insts.py -- decode KataGo ryzenai golden .insts.bin (TXN control code)
# Field layouts follow FastFlowLM npu_utils/instr_utils/*.hpp (MIT).
import json, struct, sys, os
from pathlib import Path

ROOT = Path(r"c:\Users\lizel\OneDrive\Desktop\Codes\Github\KataGo-Multi-backends")
ART = ROOT / "cpp/neuralnet/ryzenai/artifacts"

OP = {
    0: "WRITE32",
    1: "BLOCKWRITE",
    3: "MASKWRITE",
    6: "PREEMPT",
    0x80: "WAIT_TCT",
    0x81: "DDR_PATCH",
}

def u32s(data):
    return list(struct.unpack("<%dI" % (len(data)//4), data))

def decode(words, verbose=True):
    out = []
    w0, w1, ncmd, nbytes = words[0], words[1], words[2], words[3]
    hdr = dict(n_rows=w0>>24, gen=(w0>>16)&0xFF, minor=(w0>>8)&0xFF, major=w0&0xFF,
               mem_tile_rows=(w1>>8)&0xFF, num_cols=w1&0xFF, n_cmds=ncmd, n_bytes=nbytes)
    out.append(f"HDR rows={hdr['n_rows']} gen={hdr['gen']} minor={hdr['minor']} major={hdr['major']} "
               f"mem_tile_rows={hdr['mem_tile_rows']} num_cols={hdr['num_cols']} n_cmds={ncmd} n_bytes={nbytes}")
    i = 4
    cmd_idx = 0
    while i < len(words):
        op = words[i]
        if op == 1:  # BLOCKWRITE: 12 words
            bd = words[i:i+12]
            loc = bd[2]
            col = (loc>>25)&0x7F; row=(loc>>20)&0x1F; bd_id=(loc>>5)&0xF; addr = loc & 0x1FFFF
            opsize = bd[3]>>2
            buflen = bd[4]; bufoff = bd[5]
            pkt = bd[6]
            en_pkt = (pkt>>30)&1; ooo=(pkt>>24)&0x3F; pkt_id=(pkt>>19)&0x1F; pkt_ty=(pkt>>16)&7
            d0 = bd[7]; d1 = bd[8]; d2 = bd[9]; it = bd[10]; nb = bd[11]
            is_lin = (d0 == 0)
            d0s = (d0>>20)&0x3FF; d0st = (d0 & 0xFFFFF)+1
            d1s = (d1>>20)&0x3FF; d1st = (d1 & 0xFFFFF)+1
            d2st = (d2 & 0xFFFFF)+1; cache=(d2>>24)&0xF
            its = ((it>>20)&0x3FF)+1; itst = (it & 0xFFFFF)+1
            nextbd=(nb>>27)&0xF; use_next=(nb>>26)&1; valid=(nb>>25)&1
            relval=(nb>>18)&0xFF; relid=(nb>>13)&0xF; acqen=(nb>>12)&1; acqval=(nb>>5)&0x7F; acqid=nb&0xF
            d2s = buflen//(d0s*d1s) if (not is_lin and d0s*d1s) else 0
            out.append(f"[{cmd_idx:3d}@{i:4d}] BLOCKWRITE col={col} row={row} bd={bd_id} addr=0x{addr:x} opsz={opsize} "
                       f"len={buflen} bufoff={bufoff} pkt(en={en_pkt},ooo={ooo},id={pkt_id},ty={pkt_ty}) "
                       f"D0=({d0s},{d0st}) D1=({d1s},{d1st}) D2=({d2s},{d2st}) iter=({its},{itst}) cache={cache} "
                       f"next={nextbd} use_next={use_next} valid={valid} lock(rel={relval},{relid};acq={acqen},{acqval},{acqid})")
            i += 12
        elif op == 0x81:  # DDR_PATCH: 12 words
            bd = words[i:i+12]
            opsize = bd[1]>>2
            loc = bd[6]
            col=(loc>>25)&0x7F; row=(loc>>20)&0x1F; bd_id=((loc-0x04)>>5)&0x1F; addr=loc&0x1FFFF
            arg_idx = bd[8]; arg_off = bd[10]
            out.append(f"[{cmd_idx:3d}@{i:4d}] DDR_PATCH col={col} row={row} bd={bd_id} addr=0x{addr:x} arg_idx={arg_idx} arg_off={arg_off}")
            i += 12
        elif op == 0:  # WRITE32: 6 words
            bd = words[i:i+6]
            loc = bd[2]
            col=(loc>>25)&0x7F; row=(loc>>20)&0x1F; addr=loc&0xFFFFF
            val = bd[4]
            is_queue = (addr & 0x1FE00) == 0x1d200
            if is_queue:
                ch = (loc>>3)&1; direction = "MM2S" if (addr&0x10) else "S2MM"
                rep = (val>>16)&0xFF; itok=(val>>31)&1; bid=val&0xF
                out.append(f"[{cmd_idx:3d}@{i:4d}] QUEUE_PUSH col={col} row={row} {direction} ch={ch} bd={bid} repeat={rep} issue_token={itok}")
            else:
                out.append(f"[{cmd_idx:3d}@{i:4d}] WRITE32 col={col} row={row} addr=0x{addr:05x} val=0x{val:08x}")
            i += 6
        elif op == 3:  # MASKWRITE: 7 words
            bd = words[i:i+7]
            loc = bd[2]
            col=(loc>>25)&0x7F; row=(loc>>20)&0x1F; addr=loc&0xFFFFF
            val=bd[4]; mask=bd[5]
            # issue-token form: addr 0x1D2xx
            if (addr & 0x1FE00) == 0x1d200:
                ch=(loc>>3)&1; direction = "MM2S" if (addr&0x10) else "S2MM"
                pktid = val>>8
                out.append(f"[{cmd_idx:3d}@{i:4d}] ISSUE_TOKEN col={col} row={row} {direction} ch={ch} pkt_id={pktid} mask=0x{mask:x}")
            else:
                out.append(f"[{cmd_idx:3d}@{i:4d}] MASKWRITE col={col} row={row} addr=0x{addr:05x} val=0x{val:08x} mask=0x{mask:08x}")
            i += 7
        elif op == 0x80:  # WAIT_TCT: 4 words
            bd = words[i:i+4]
            opsize=bd[1]>>2
            x=bd[2]; y=bd[3]
            direction = "MM2S" if (x&1) else "S2MM"
            row=(x>>8)&0xFF; col=(x>>16)&0xFF; ch=(y>>24)&0xFF
            out.append(f"[{cmd_idx:3d}@{i:4d}] WAIT_TCT col={col} row={row} {direction} ch={ch} (w3=0x{y:x})")
            i += 4
        elif op == 6:
            out.append(f"[{cmd_idx:3d}@{i:4d}] PREEMPT level={(words[i]>>8)&3}")
            i += 1
        else:
            out.append(f"[{cmd_idx:3d}@{i:4d}] UNKNOWN op=0x{op:x} words={['%08x'%w for w in words[i:i+4]]}")
            i += 1
        cmd_idx += 1
    return hdr, out

def main():
    manifest = json.loads((ART/"manifest.json").read_text())
    seen = {}
    for a in manifest["artifacts"]:
        p = ART / a["insts"]
        key = (a["arch"], a["n_aie_cols"], a["M"], a["K"], a["N"])
        if key in seen:
            continue
        seen[key] = True
        data = p.read_bytes()
        words = u32s(data)
        hdr, lines = decode(words)
        tag = f"{a['arch']}_{a['n_aie_cols']}col M{a['M']}K{a['K']}N{a['N']} tile={a['tile']['m']}x{a['tile']['k']}x{a['tile']['n']}"
        print("="*100)
        print(f"{p.relative_to(ART)}  {tag}")
        print("="*100)
        for l in lines:
            print(l)

if __name__ == "__main__":
    main()

import os
import sys
import random
import numpy as np

from sgfmetadata import SGFMetadata

expected_keys = [
    "binaryInputNCHWPacked",
    "globalInputNC",
    "policyTargetsNCMove",
    "globalTargetsNC",
    "scoreDistrN",
    "valueTargetsNCHW",
]

def process_npz_files(in_dir, out_dir):
    rand = random.Random()
    for filename in os.listdir(in_dir):
        if filename.endswith(".npz"):
            print("Processing " + filename, flush=True)
            with np.load(os.path.join(in_dir, filename)) as npz:
                assert(set(npz.keys()) == set(expected_keys))
                binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
                globalInputNC = npz["globalInputNC"]
                policyTargetsNCMove = npz["policyTargetsNCMove"]
                globalTargetsNC = npz["globalTargetsNC"]
                scoreDistrN = npz["scoreDistrN"]
                valueTargetsNCHW = npz["valueTargetsNCHW"]

                pos_len = valueTargetsNCHW.shape[2]

                binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
                assert len(binaryInputNCHW.shape) == 3
                assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
                binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
                binaryInputNCHW = np.reshape(binaryInputNCHW, (
                    binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
                )).astype(np.float32)

                metadataInputNC = np.zeros((binaryInputNCHW.shape[0],SGFMetadata.METADATA_INPUT_NUM_CHANNELS), dtype=np.float32)
                for i in range(binaryInputNCHW.shape[0]):
                    board_area = int(np.sum(binaryInputNCHW[i,0]))
                    sgfmeta = SGFMetadata.get_katago_selfplay_metadata(board_area=board_area,rand=rand)
                    # nextPlayer doesn't matter in selfplay because it only is used to swap pla/opp metadata
                    # and in this case both sides are identically katago
                    metadataInputNC[i] = sgfmeta.get_metadata_row(nextPlayer=Board.WHITE)

                np.savez_compressed(
                    os.path.join(out_dir, filename),
                    binaryInputNCHWPacked = binaryInputNCHWPacked,
                    globalInputNC = globalInputNC,
                    policyTargetsNCMove = policyTargetsNCMove,
                    globalTargetsNC = globalTargetsNC,
                    scoreDistrN = scoreDistrN,
                    valueTargetsNCHW = valueTargetsNCHW,
                    metadataInputNC = metadataInputNC,
                )
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_kata_selfplay_meta_to_npzs.py <in_dir> <out_dir>")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    if os.path.exists(out_dir):
        raise Exception(out_dir + " already exists")
    os.mkdir(out_dir)

    process_npz_files(in_dir, out_dir)

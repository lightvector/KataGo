import json
import random
import argparse
import os
import sgfmill
import math
import sys
from dataclasses import dataclass
from collections import defaultdict

from PIL import Image, ImageDraw

from katago.game.board import Board

# . stone or rest of board
# e empty
# s stone
# c critical
eyepatterns = [
    (50, [
        "s",
    ]),
    (20, [
        "e",
    ]),
    (10, [
        "ee",
    ]),
    (5, [
        "ece",
    ]),
    (5, [
        "ec",
        ".e",
    ]),
    (1, [
        "ecce",
    ]),
    (1, [
        "ec.",
        ".ce",
    ]),
    (1, [
        "ecc",
        "..e",
    ]),
    (2, [
        "ee",
        "ee",
    ]),
    (3, [
        "ee.",
        "ece",
    ]),
    (1, [
        ".e.",
        "ece",
        ".e.",
    ]),
    (1, [
        "eccce",
    ]),
    (1, [
        "eccc",
        "...e"
    ]),
    (1.5, [
        ".ee",
        "ece",
        ".e.",
    ]),
    (1, [
        "ece",
        "ece",
        ".e.",
    ]),
    (1, [
        ".ee",
        "ccc",
        "ee.",
    ]),
    (1, [
        "ece",
        "ece",
    ]),
]

def randint_exponential(scale):
    r = 0
    while r <= 0:
        r = random.random()
    return int(math.floor(-math.log(r) * scale))

@dataclass
class GroupInfo:
    has_eye: bool = False
    pla: int = 0
    size: int = 0

def gen(xsize,ysize):
    stones = [ [ 0 for _ in range(xsize) ] for _ in range(ysize) ]
    owned = [ [ 0 for _ in range(xsize) ] for _ in range(ysize) ]
    kinds = [ [ '.' for _ in range(xsize) ] for _ in range(ysize) ]
    patternids = [ [ 0 for _ in range(xsize) ] for _ in range(ysize) ]

    next_pattern_id = 1
    def place_pattern(pla):
        nonlocal next_pattern_id
        pattern = random.choices(
            [pattern for (weight,pattern) in eyepatterns],
            weights=[weight for (weight,pattern) in eyepatterns],
            k=1
        )[0]
        pattern = [ [ c for c in row ] for row in pattern ]
        if random.random() >= 0.5:
            pattern = list(zip(*pattern))
        if random.random() >= 0.5:
            pattern = list(list(reversed(row)) for row in pattern)
        if random.random() >= 0.5:
            pattern = list(reversed(pattern))

        pysize = len(pattern)
        pxsize = len(pattern[0])

        # Generate too large and clamp to border, to make borders more likely
        yoffset = random.randint(-1,ysize-pysize+1)
        xoffset = random.randint(-1,xsize-pxsize+1)
        yoffset = min(yoffset,ysize-pysize)
        xoffset = min(xoffset,xsize-pxsize)
        yoffset = max(yoffset,0)
        xoffset = max(xoffset,0)

        numcs = sum(kind == 'c' for row in pattern for kind in row)
        for dy in range(pysize):
            for dx in range(pxsize):
                kind = pattern[dy][dx]
                if kind == 'c':
                    cpla = 0
                    if numcs == 1:
                        r = random.random()
                        if r < 0.7:
                            cpla = 3-pla
                        elif r < 0.9:
                            cpla = 0
                        else:
                            cpla = pla
                    elif numcs == 2:
                        r = random.random()
                        if r < 0.3:
                            cpla = 3-pla
                        elif r < 0.95:
                            cpla = 0
                        else:
                            cpla = pla
                    kinds[yoffset+dy][xoffset+dx] = 'e'
                    owned[yoffset+dy][xoffset+dx] = pla
                    stones[yoffset+dy][xoffset+dx] = cpla
                    patternids[yoffset+dy][xoffset+dx] = next_pattern_id
                else:
                    kinds[yoffset+dy][xoffset+dx] = kind
                    owned[yoffset+dy][xoffset+dx] = pla
                    patternids[yoffset+dy][xoffset+dx] = next_pattern_id

        next_pattern_id += 1

    num_b_patterns = 1
    num_w_patterns = 1
    if random.random() < 0.5:
        place_pattern(1)
        place_pattern(2)
    else:
        place_pattern(2)
        place_pattern(1)

    num_extra_patterns = 0
    num_extra_patterns = randint_exponential(10) + randint_exponential(3)
    for _ in range(num_extra_patterns):
        if random.random() < num_b_patterns / (num_b_patterns + num_w_patterns):
            num_w_patterns += 1
            place_pattern(2)
        else:
            num_b_patterns += 1
            place_pattern(1)

    def filleyewall(y,x,pla):
        if owned[y][x] != 0 and owned[y][x] != pla:
            stones[y][x] = random.randint(1,2)
            owned[y][x] = 3
            kinds[y][x] = 'x'
        else:
            owned[y][x] = pla
            stones[y][x] = pla

    # Fill in single stones
    for y in range(ysize):
        for x in range(xsize):
            if kinds[y][x] == 's':
                stones[y][x] = owned[y][x]

    # Fill in sides
    for y in range(ysize):
        for x in range(xsize-1):
            if kinds[y][x] == 'e' and kinds[y][x+1] == '.':
                filleyewall(y,x+1,owned[y][x])
            elif kinds[y][x+1] == 'e' and kinds[y][x] == '.':
                filleyewall(y,x,owned[y][x+1])
    for x in range(xsize):
        for y in range(ysize-1):
            if kinds[y][x] == 'e' and kinds[y+1][x] == '.':
                filleyewall(y+1,x,owned[y][x])
            elif kinds[y+1][x] == 'e' and kinds[y][x] == '.':
                filleyewall(y,x,owned[y+1][x])

    # Fill in corners
    corneroppprob = random.random() * 0.24 + 0.18
    corneropps_by_patternid = defaultdict(list)
    for y in range(ysize-1):
        for x in range(xsize-1):
            ecount = (
                int(kinds[y][x] == 'e') +
                int(kinds[y][x+1] == 'e') +
                int(kinds[y+1][x] == 'e') +
                int(kinds[y+1][x+1] == 'e')
            )
            ccount = (
                int(kinds[y][x] == 'c') +
                int(kinds[y][x+1] == 'c') +
                int(kinds[y+1][x] == 'c') +
                int(kinds[y+1][x+1] == 'c')
            )
            dotcount = (
                int(kinds[y][x] == '.') +
                int(kinds[y][x+1] == '.') +
                int(kinds[y+1][x] == '.') +
                int(kinds[y+1][x+1] == '.')
            )
            scount = (
                int(kinds[y][x] == 's') +
                int(kinds[y][x+1] == 's') +
                int(kinds[y+1][x] == 's') +
                int(kinds[y+1][x+1] == 's')
            )
            if ecount + ccount == 1 and dotcount + scount == 3:
                if kinds[y][x] == 'e':
                    cornerx = x+1
                    cornery = y+1
                    cornerpla = owned[y][x]
                    patternid = patternids[y][x]
                elif kinds[y+1][x] == 'e':
                    cornerx = x+1
                    cornery = y
                    cornerpla = owned[y+1][x]
                    patternid = patternids[y+1][x]
                elif kinds[y][x+1] == 'e':
                    cornerx = x
                    cornery = y+1
                    cornerpla = owned[y][x+1]
                    patternid = patternids[y][x+1]
                elif kinds[y+1][x+1] == 'e':
                    cornerx = x
                    cornery = y
                    cornerpla = owned[y+1][x+1]
                    patternid = patternids[y+1][x+1]
                else:
                    assert False

                if random.random() < corneroppprob:
                    corneropps_by_patternid[patternid].append((cornerx,cornery))
                    filleyewall(cornery,cornerx,3-cornerpla)
                elif random.random() < 0.60:
                    pass
                else:
                    filleyewall(cornery,cornerx,cornerpla)

    info_by_label = {}
    owned_labels = [ [ 0 for _ in range(xsize) ] for _ in range(ysize) ]
    def label_owned_region(label, pla, x, y):
        if owned_labels[y][x] == 0 and owned[y][x] == pla:
            if stones[y][x] == 0:
                info_by_label[label].has_eye = True
            owned_labels[y][x] = label
            info_by_label[label].size += 1
            if x > 0:
                label_owned_region(label, pla, x-1, y)
            if x < xsize-1:
                label_owned_region(label, pla, x+1, y)
            if y > 0:
                label_owned_region(label, pla, x, y-1)
            if y < ysize-1:
                label_owned_region(label, pla, x, y+1)

    next_owned_label = 1
    for y in range(ysize):
        for x in range(xsize):
            if owned_labels[y][x] == 0 and owned[y][x] != 0:
                info_by_label[next_owned_label] = GroupInfo()
                info_by_label[next_owned_label].pla = owned[y][x]
                label_owned_region(next_owned_label, owned[y][x], x, y)
                next_owned_label += 1

    # print(corneropps_by_patternid)

    def merge(label1, label2):
        assert info_by_label[label1].pla == info_by_label[label2].pla
        info_by_label[label1].has_eye = (info_by_label[label1].has_eye or info_by_label[label2].has_eye)
        info_by_label[label1].size = (info_by_label[label1].size + info_by_label[label2].size)
        for y in range(ysize):
            for x in range(xsize):
                if owned_labels[y][x] == label2:
                    owned_labels[y][x] = label1
        del info_by_label[label2]

    num_merges = random.random() * 0.25 * len(info_by_label)
    num_merges = int(num_merges) + (1 if random.random() < (num_merges - int(num_merges)) else 0)
    for _ in range(num_merges):
        keys = list(info_by_label.keys())
        key1 = random.choice(keys)
        otherkeys = [key for key in keys if info_by_label[key].pla == info_by_label[key1].pla and key != key1]
        if len(otherkeys) > 0:
            key2 = random.choice(otherkeys)
            merge(key1,key2)

    for corneropps in corneropps_by_patternid.values():
        if random.random() < 0.7:
            for (x1,y1) in corneropps:
                for (x2,y2) in corneropps:
                    label1 = owned_labels[y1][x1]
                    label2 = owned_labels[y2][x2]
                    if label1 != label2:
                        if info_by_label[label1].pla == info_by_label[label2].pla:
                            merge(label1,label2)

    def adj(x,y):
        if x > 0:
            yield (x-1,y)
        if x < xsize-1:
            yield (x+1,y)
        if y > 0:
            yield (x,y-1)
        if y < ysize-1:
            yield (x,y+1)
    def diag_reachable(x,y):
        if x > 0:
            if y > 0:
                if owned[y][x-1] == 0 or owned[y-1][x] == 0:
                    yield (x-1,y-1)
            if y < ysize-1:
                if owned[y][x-1] == 0 or owned[y+1][x] == 0:
                    yield (x-1,y+1)
        if x < xsize-1:
            if y > 0:
                if owned[y][x+1] == 0 or owned[y-1][x] == 0:
                    yield (x+1,y-1)
            if y < ysize-1:
                if owned[y][x+1] == 0 or owned[y+1][x] == 0:
                    yield (x+1,y+1)

    # Is this a location with 2 adjacent opponents and 1 adjacent player that
    # is NOT the given label
    def is_crossroads(x,y,pla,label):
        if owned[y][x] != 0:
            return False
        pcount = 0
        ocount = 0
        opp = 3-pla
        for (ax,ay) in adj(x,y):
            if owned[ay][ax] == pla and owned_labels[ay][ax] != label:
                pcount += 1
            elif owned[ay][ax] == opp:
                ocount += 1
        if pcount == 1 and ocount == 2:
            return True
        return False

    black_grow_prob = sum(random.random() for i in range(8)) / 8
    def grow(allow_diag_blocker):
        pla = 1 if random.random() < black_grow_prob else 2
        growspots = []
        for y in range(ysize):
            for x in range(xsize):
                if owned[y][x] == 0:
                    adjacent_labels = []
                    blocker_labels = []
                    for (ax,ay) in adj(x,y):
                        if owned[ay][ax] == pla:
                            adjacent_labels.append(owned_labels[ay][ax])
                    for (ax,ay) in diag_reachable(x,y):
                        if owned[ay][ax] == pla:
                            blocker_labels.append(owned_labels[ay][ax])

                    adjacent_labels = list(set(adjacent_labels))
                    blocker_labels = list(set(blocker_labels).difference(set(adjacent_labels)))
                    if len(adjacent_labels) == 1 and (allow_diag_blocker or len(blocker_labels) == 0):
                        label = adjacent_labels[0]
                        has_crossroads = False
                        for (ax,ay) in adj(x,y):
                            if is_crossroads(ax,ay,pla,label):
                                has_crossroads = True
                        if not has_crossroads:
                            weight = 1.0 / math.sqrt(0.1 + info_by_label[label].size)
                            growspots.append(((x,y),label,weight))

        # print(growspots)
        if len(growspots) > 0:
            ((x,y),label,weight) = random.choices(growspots,weights=(weight for (_,_,weight) in growspots))[0]
            owned[y][x] = pla
            owned_labels[y][x] = label
            stones[y][x] = pla
            info_by_label[label].size += 1

    for i in range(700):
        grow(allow_diag_blocker=False)
    for i in range(100):
        grow(allow_diag_blocker=True)

    def shrink(allow_far_shrink, allow_disconnect, replace_with_opp):
        shrinkables = []
        for y in range(ysize):
            for x in range(xsize):
                if stones[y][x] != 0:
                    pla = stones[y][x]
                    opp = 3-pla
                    adjplalabels = []
                    adjopplabels = []
                    for (ax,ay) in adj(x,y):
                        if owned[ay][ax] == pla:
                            adjplalabels.append(owned_labels[ay][ax])
                        if owned[ay][ax] == opp and (allow_far_shrink or stones[ay][ax] == opp):
                            adjopplabels.append(owned_labels[ay][ax])

                    adjplalabelset = set(adjplalabels)
                    adjopplabelset = set(adjopplabels)
                    if not (len(adjplalabelset) == 1 and len(adjopplabelset) == 1):
                        continue
                    if replace_with_opp and len(adjopplabels) != 1:
                        continue

                    allow = True
                    if not allow_disconnect:
                        # Set temporarily
                        stones[y][x] = 0
                        for (ax,ay) in adj(x,y):
                            if stones[ay][ax] == pla:
                                floodfill = [ [ 0 for _ in range(xsize) ] for _ in range(ysize) ]
                                queue = [(ax,ay)]
                                idx = 0
                                while idx < len(queue):
                                    fx, fy = queue[idx]
                                    idx += 1
                                    if floodfill[fy][fx] == 0:
                                        floodfill[fy][fx] = 1
                                        for (fx2,fy2) in adj(fx,fy):
                                            if stones[fy2][fx2] == pla:
                                                queue.append((fx2,fy2))
                                break
                        disconnected = False
                        for (ax,ay) in adj(x,y):
                            if stones[ay][ax] == pla and floodfill[ay][ax] != 1:
                                disconnected = True
                                break
                        if disconnected:
                            allow = False
                        # Undo
                        stones[y][x] = pla

                    if allow:
                        shrinkables.append((x,y,pla,adjopplabels[0]))

        if len(shrinkables) > 0:
            (x,y,pla,adjopplabel) = random.choice(shrinkables)
            if replace_with_opp:
                stones[y][x] = 3-pla
                owned_labels[y][x] = adjopplabel
                owned[y][x] = 3-pla
            else:
                stones[y][x] = 0

    num_replaces = randint_exponential(25)
    for _ in range(num_replaces):
        shrink(allow_far_shrink=False, allow_disconnect=False, replace_with_opp=True)

    shrinks = []
    num_shrinks = randint_exponential(22 + int(len(info_by_label) * 0.3))
    num_far_shrinks = randint_exponential(20)
    for _ in range(num_shrinks):
        shrinks.append("shrink")
    for _ in range(num_far_shrinks):
        shrinks.append("far_shrink")
    random.shuffle(shrinks)
    for s in shrinks:
        if s == "shrink":
            shrink(allow_far_shrink=False, allow_disconnect=False, replace_with_opp=False)
        elif s == "far_shrink":
            shrink(allow_far_shrink=True, allow_disconnect=False, replace_with_opp=False)

    num_disconnect_shrinks = randint_exponential(5)
    for _ in range(num_disconnect_shrinks):
        shrink(allow_far_shrink=True, allow_disconnect=True, replace_with_opp=False)

    # print("=========================================================")
    # # print("\n".join("".join(row) for row in kinds))
    # # print("-----------------------------------------")
    # print("\n".join("".join(str("x" if elt == 1 else "o" if elt == 2 else ".") for elt in row) for row in owned))
    # print("-----------------------------------------")
    # print("\n".join("".join(str("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[elt]) for elt in row) for row in owned_labels))
    # print(info_by_label)
    # print("-----------------------------------------")
    # print("\n".join("".join(str("x" if elt == 1 else "o" if elt == 2 else ".") for elt in row) for row in stones))

    locs = []
    for y in range(ysize):
        for x in range(xsize):
            locs.append((x,y))

    assert xsize == ysize
    b = Board(xsize)
    for (x,y) in locs:
        loc = b.loc(x,y)
        # Final slight omission to poke occasional eyes into huge solid stone areas.
        if random.random() < 1.0 / 2000.0:
            stones[y][x] = 0

        pla = stones[y][x]
        if b.would_be_legal(pla,loc):
            b.play(pla,loc)

    num_diffs = 0
    for (x,y) in locs:
        loc = b.loc(x,y)
        if b.board[loc] != stones[y][x]:
            num_diffs += 1

    if num_diffs > 10:
        return None

    return b


def save_image(board, xsize, ysize):
    CELL_SIZE = 20
    STONE_RADIUS = 8

    # Create blank board image
    img = Image.new('RGB', (CELL_SIZE*ysize, CELL_SIZE*xsize), color=(160,82,45))

    # Draw grid lines
    draw = ImageDraw.Draw(img)
    for x in range(xsize):
      draw.line((CELL_SIZE*x, 0, CELL_SIZE*x, CELL_SIZE*ysize), fill=(0,0,0))
    for y in range(ysize):
      draw.line((0, CELL_SIZE*y, CELL_SIZE*xsize, CELL_SIZE*y), fill=(0,0,0))

    # Draw stones
    for y in range(ysize):
      for x in range(xsize):
        if board[y][x] == 1:
          draw.ellipse((CELL_SIZE*y+CELL_SIZE/2-STONE_RADIUS, CELL_SIZE*x+CELL_SIZE/2-STONE_RADIUS, CELL_SIZE*y+CELL_SIZE/2+STONE_RADIUS, CELL_SIZE*x+CELL_SIZE/2+STONE_RADIUS), fill=(0,0,0))
        elif board[y][x] == 2:
          draw.ellipse((CELL_SIZE*y+CELL_SIZE/2-STONE_RADIUS, CELL_SIZE*x+CELL_SIZE/2-STONE_RADIUS, CELL_SIZE*y+CELL_SIZE/2+STONE_RADIUS, CELL_SIZE*x+CELL_SIZE/2+STONE_RADIUS), fill=(255,255,255))

    # Save image
    img.save('tmpclumpypos.png')


def gen_sgfs_debug():
    for i in range(100):
        xsize = 19
        ysize = 19
        b = None
        while b is None:
            b = gen(xsize,ysize)
        initial_pla = random.choice([1,2])

        with open(f"tmp/clumpysgfs/{i}.sgf","w") as f:
            f.write("(;FF[4]GM[1]SZ[19]HA[0]KM[6.5]")
            whitepieces = []
            blackpieces = []
            for y in range(ysize):
                for x in range(xsize):
                    loc = b.loc(x,y)
                    color = b.board[loc]
                    if color == 1:
                        xc = "abcdefghijklmnopqrstuvwxyz"[x]
                        yc = "abcdefghijklmnopqrstuvwxyz"[y]
                        blackpieces.append(f"[{xc}{yc}]")
                    elif color == 2:
                        xc = "abcdefghijklmnopqrstuvwxyz"[x]
                        yc = "abcdefghijklmnopqrstuvwxyz"[y]
                        whitepieces.append(f"[{xc}{yc}]")
            if len(whitepieces) > 0:
                f.write("AW"+"".join(whitepieces))
            if len(blackpieces) > 0:
                f.write("AB"+"".join(blackpieces))
            f.write("PL[W]" if initial_pla == 2 else "")
            f.write(")")

        print(i)
        print(b.to_string())

        # board = [ [ 0 for _ in range(xsize) ] for _ in range(ysize) ]
        # for y in range(ysize):
        #     for x in range(xsize):
        #         loc = b.loc(x,y)
        #         board[y][x] = b.board[loc]
        # save_image(board,xsize,ysize)


def maybe_generate_one_pos(out, training_weight, max_lopsidedness, soft_filter_large_group_scale):
    xsize = 19
    ysize = 19
    board = None
    while board is None:
        board = gen(xsize,ysize)

    soft_keep_prob = 1.0
    visited_group_heads = set()
    for y in range(ysize):
        for x in range(xsize):
            loc = board.loc(x,y)
            if board.board[loc] == 1 or board.board[loc] == 2:
                head = board.group_head[loc]
                if head not in visited_group_heads:
                    visited_group_heads.add(head)
                    stonecount = board.group_stone_count[head]
                    if stonecount > 2 * soft_filter_large_group_scale:
                        soft_keep_prob = 0.0
                    elif stonecount > soft_filter_large_group_scale:
                        excess = stonecount - soft_filter_large_group_scale
                        soft_keep_prob *= math.exp(-excess * 3 / soft_filter_large_group_scale)

    initial_pla = random.choice([1,2])

    to_write = {}
    to_write["board"] = ""
    num_stones = 0
    num_black = 0
    num_white = 0
    for y in range(ysize):
        for x in range(xsize):
            loc = board.loc(x,y)
            if board.board[loc] == 1:
                to_write["board"] += "X"
                num_stones += 1
                num_black += 1
            elif board.board[loc] == 2:
                to_write["board"] += "O"
                num_stones += 1
                num_white += 1
            else:
                to_write["board"] += "."
        to_write["board"] += "/"

    if abs(num_black-num_white) > max_lopsidedness:
        return False

    # print(str(soft_keep_prob) + "\n" + to_write["board"].replace("/","\n"))
    if soft_keep_prob < 0.05:
        return False
    if random.random() >= soft_keep_prob:
        return False

    to_write["hintLoc"] = "null"
    to_write["nextPla"] = ("B" if initial_pla == 1 else "W")
    to_write["initialTurnNumber"] = num_stones
    to_write["moveLocs"] = []
    to_write["movePlas"] = []
    to_write["weight"] = 1.0
    to_write["trainingWeight"] = training_weight
    to_write["xSize"] = board.size
    to_write["ySize"] = board.size
    out.write(json.dumps(to_write) + "\n")
    out.flush()
    return True

def main(num_to_generate, out_file, training_weight, max_lopsidedness, soft_filter_large_group_scale):
    with open(out_file,"w") as out:
        num_kept = 0
        while True:
            suc = maybe_generate_one_pos(out, training_weight, max_lopsidedness, soft_filter_large_group_scale)
            if suc:
                num_kept += 1
            if num_kept >= num_to_generate:
                break
            if num_kept % 100 == 0:
                print(f"Kept {num_kept} so far")
                sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-num-to-generate", required=True, type=int)
    parser.add_argument("-out-file", required=True)
    parser.add_argument("-training-weight", required=True, type=float)
    parser.add_argument("-max-lopsidedness", required=True, type=int)
    parser.add_argument("-soft-filter-large-group-scale", required=True, type=float)
    args = vars(parser.parse_args())

    main(
        args["num_to_generate"],
        args["out_file"],
        args["training_weight"],
        args["max_lopsidedness"],
        args["soft_filter_large_group_scale"],
    )



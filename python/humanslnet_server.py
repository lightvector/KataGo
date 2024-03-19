import sys
import json
import numpy as np
from load_model import load_model
from gamestate import GameState, SGFMetadata
import argparse

def numpy_array_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', help='Checkpoint to test', required=True)
    args = parser.parse_args()

    model, _, _ = load_model(args.checkpoint, use_swa=False, device="cpu", pos_len=19, verbose=False)
    game_state = None

    def write(output):
        sys.stdout.write(json.dumps(output,default=numpy_array_encoder) + "\n")
        sys.stdout.flush()

    # DEBUGGING
    # game_state = GameState(board_size=19, rules=GameState.RULES_JAPANESE)
    # sgfmeta = SGFMetadata()
    # outputs = game_state.get_model_outputs(model, sgfmeta=sgfmeta)
    # write(dict(outputs=outputs))

    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            data = json.loads(line)

            if data["command"] == "start":
                board_size = data["board_size"]
                rules = data["rules"]
                game_state = GameState(board_size, rules)
                write(dict(outputs=""))

            elif data["command"] == "play":
                pla = data["pla"]
                loc = data["loc"]
                game_state.play(pla, loc)
                write(dict(outputs=""))

            elif data["command"] == "undo":
                game_state.undo()
                write(dict(outputs=""))

            elif data["command"] == "get_model_outputs":
                sgfmeta = SGFMetadata.of_dict(data["sgfmeta"])
                outputs = game_state.get_model_outputs(model, sgfmeta=sgfmeta)
                write(dict(outputs=outputs))

            else:
                raise ValueError(f"Unknown command: {data['command']}")

        except (KeyboardInterrupt, EOFError, BrokenPipeError):
            break

        except Exception as e:
            raise RuntimeError(f"Error processing command: {e}")

if __name__ == "__main__":
    main()

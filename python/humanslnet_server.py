import sys
import json
import numpy as np
from load_model import load_model
from gamestate import GameState
from features import Features
from sgfmetadata import SGFMetadata
import argparse

def numpy_array_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def write(output):
        sys.stdout.write(json.dumps(output,default=numpy_array_encoder) + "\n")
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', help='Checkpoint to test', required=True)
    parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
    parser.add_argument('-device', help='Device to use, such as cpu or cuda:0', required=True)
    parser.add_argument('-webserver', help='set if is used for the flask wrapper', required=False)
    args = parser.parse_args()

    model, swa_model, _ = load_model(args.checkpoint, use_swa=args.use_swa, device=args.device, pos_len=19, verbose=False)
    if swa_model is not None:
        model = swa_model
    game_state = None

    if args.webserver:
        write("Ready to receive input")

    
    # DEBUGGING
    # game_state = GameState(board_size=19, rules=GameState.RULES_JAPANESE)
    # sgfmeta = SGFMetadata()
    # outputs = game_state.get_model_outputs(model, sgfmeta=sgfmeta)
    # write(dict(outputs=outputs))

    for line in sys.stdin:
        line = line.strip()
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

        elif data["command"] == "redo":
            game_state.redo()
            write(dict(outputs=""))

        elif data["command"] == "get_model_outputs":
            sgfmeta = SGFMetadata.of_dict(data["sgfmeta"])
            outputs = game_state.get_model_outputs(model, sgfmeta=sgfmeta)
            filtered_outputs = {}
            for key in outputs:
                if key in ["moves_and_probs0", "value", "lead", "scorestdev"]:
                    filtered_outputs[key] = outputs[key]
            write(dict(outputs=filtered_outputs))

        elif data["command"] == "get_best_move":
            sgfmeta = SGFMetadata.of_dict(data["sgfmeta"])

            # Run Monte Carlo Tree Search with a specified number of visits
            visits = data.get("visits", 100)  # Default to 100 visits if not specified
            game_state.run_monte_carlo_tree_search(model, sgfmeta, visits)

            # Write the refined outputs back to the output
            write({"best_move": game_state.best_move})

        else:
            raise ValueError(f"Unknown command: {data['command']}")


if __name__ == "__main__":
    main()

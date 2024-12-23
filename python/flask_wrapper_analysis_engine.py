from flask import Flask, request, jsonify
from subprocess import Popen, PIPE, STDOUT
import threading
import json
import os

app = Flask(__name__)

# Store the process globally
cli_process = None
lock = threading.Lock()

def start_cli():
    global cli_process
    command = os.environ.get('KATAGO_COMMAND')

    if not command:
        raise ValueError('Environment variable KATAGO_COMMAND not set')

    with lock:
        if cli_process is None:
            cli_process = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=True)
            wait_for_ready_state(cli_process)

def wait_for_ready_state(process):
    """Reads lines from the CLI program until it outputs the ready state message."""
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(f"Initial CLI Output: {line.strip()}")  # Print the initial output for debugging
            
            if "Started, ready to begin handling requests" in line:
                print("CLI program is ready.")
                break  # Stop reading once the ready state is detected
    except Exception as e:
        print(f"Error reading initial output: {str(e)}")

@app.route('/run', methods=['POST'])
def run_cli():
    global cli_process
    input_data = request.json

    if not input_data:
        return jsonify({'error': 'No input provided'}), 400

    analyze_turns = input_data.get("analyzeTurns", [])
    num_responses_expected = len(analyze_turns)

    with lock:
        if cli_process is None:
            return jsonify({'error': 'CLI program is not running'}), 400

        try:
            # Send input to the process
            input_json = json.dumps(input_data)
            cli_process.stdin.write(input_json + '\n')
            cli_process.stdin.flush()

            # Read the output from the process until all responses are received
            responses = []
            while len(responses) < num_responses_expected:
                output_line = cli_process.stdout.readline()
                if not output_line:
                    break
                output_line = output_line.strip()
                
                try:
                    # Attempt to parse the output as JSON
                    response_data = json.loads(output_line)
                    responses.append(response_data)
                except json.JSONDecodeError:
                    # Ignore lines that are not valid JSON
                    print(f"Non-JSON line ignored: {output_line}")

            if len(responses) != num_responses_expected:
                return jsonify({'error': 'Did not receive the expected number of responses from CLI program'}), 500

            return jsonify(responses)
        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    start_cli()  # Start the CLI program when the server starts
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from subprocess import Popen, PIPE
import time
import threading
import json
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__)

# Store the process globally
cli_process = None
lock = threading.Lock()

def start_cli():
    global cli_process
    global monitor_output_thread
    command = "python ./humanslnet_server.py -checkpoint ./b18c384nbt-humanv0.ckpt -device cpu -webserver True"

    with lock:
        if cli_process is None:
            cli_process = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)

            wait_for_ready_state(cli_process)
            stderr_thread = threading.Thread(target=monitor_stderr, args=(cli_process,))
            
            stderr_thread.daemon = True
            
            stderr_thread.start()


def wait_for_ready_state(process):
    """Reads lines from the CLI program until it outputs the ready state message."""
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(f"Initial CLI Output: {line.strip()}")  # Print the initial output for debugging
            
            if "Ready to receive input" in line:
                print("CLI program is ready.")
                break  # Stop reading once the ready state is detected
    except Exception as e:
        print(f"Error reading initial output: {str(e)}")

def monitor_stderr(process):
    global cli_process
    while True:
        error_output = process.stderr.readline()
        if error_output == '' and process.poll() is not None:
            break
        if error_output:
            print(f"ERROR: {error_output.strip()}")
            if "ERROR" in error_output or "Exception" in error_output or "Error" in error_output:
                print("Error detected, quitting the process...")
                os._exit(1)
        time.sleep(0.1)


def restart_process():
    global cli_process
    with lock:
        if cli_process:
            cli_process.kill()
            cli_process = None
            start_cli()

@app.route('/', methods=['POST'])
@cross_origin()
def run_cli():
    global cli_process
    input_data = request.json

    if not input_data:
        return jsonify({'error': 'No input provided'}), 400

    with lock:
        if cli_process is None:
            start_cli()
            return jsonify({'error': 'CLI program was not running, restarted'}), 400

        try:
            # Send input to the process
            input_json = json.dumps(input_data)
            cli_process.stdin.write(input_json + '\n')
            cli_process.stdin.flush()

            # Read the output from the process until all responses are received
            num_responses_expected = 1
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

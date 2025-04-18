#!/bin/bash -eu
set -o pipefail
{

# Adds noise to a KataGo model and exports it for selfplay.
# This script adds specified noise to a checkpoint model and prepares it for use in selfplay.

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODEL_KIND NOISE_SCALE"
    echo "BASEDIR base directory of the training run (e.g. '/d/Projects/KataGo/Training/BaseDir')"
    echo "TRAININGNAME name of the training run (e.g. 'kata1-b28c512nbt')"
    echo "NOISE_SCALE amount of noise to add to the model (e.g. 0.1)"
    echo "MODEL_KIND what size model (e.g. 'b28c512nbt')"
    exit 0
fi

# ./selfplay/noise.sh /g/Projects/KataGo/Training/BaseDir kata1-b28c512nbt b28c512nbt 0.1


BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODEL_KIND="$1"
shift
NOISE_SCALE="$1"
shift


# Make sure paths are properly formatted for the system
# Convert Windows-style paths to Unix-style if needed
BASEDIR=$(echo "$BASEDIR" | sed 's/\\/\//g')

CHECKPOINT="$BASEDIR/train"

# Display what we're about to do
echo "Base directory: $BASEDIR"
echo "Training name: $TRAININGNAME"
echo "Model kind: $MODEL_KIND"
echo "Adding noise to model with scale $NOISE_SCALE"

# Run the noise script to add noise to the model
time python noise.py \
    --base-dir "$BASEDIR" \
    --training-name "$TRAININGNAME" \
    --model-kind "$MODEL_KIND" \
    --noise-scale "$NOISE_SCALE" \
    --export-prefix "noisy"

# Check if noise.py completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to add noise to the model"
    exit 1
fi

# Export the noisy model for selfplay
echo "Exporting noisy model for selfplay"
(
    time ./export_model_for_selfplay.sh "noisy-$NOISE_SCALE" "$BASEDIR" "1"
)

echo "Noisy model created and exported successfully"

exit 0
}

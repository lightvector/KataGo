#!/bin/bash -eu
set -o pipefail
{

# Merges multiple checkpoint models to the latest and exports it for selfplay.

# Accept 3 arguments
if [[ $# -lt 3 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODEL_KIND"
    echo "BASEDIR base directory of the training run (e.g. '/d/Projects/KataGo-Noise/Training/BaseDir')"
    echo "TRAININGNAME name of the training run (e.g. 'kata1-b28c512nbt')"
    echo "MODEL_KIND what size model (e.g. 'b28c512nbt')"
    exit 0
fi

# cd /g/Projects/KataGo-Noise/python
# ./selfplay/merge.sh /g/Projects/KataGo-Noise/Training/BaseDir kata1-b28c512nbt b28c512nbt


BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODEL_KIND="$1"
shift


# Make sure paths are properly formatted for the system
# Convert Windows-style paths to Unix-style if needed
BASEDIR=$(echo "$BASEDIR" | sed 's/\\/\//g')

# CHECKPOINT="$BASEDIR/train"

# Display what we're about to do
echo "Base directory: $BASEDIR"
echo "Training name: $TRAININGNAME"
echo "Model kind: $MODEL_KIND"

# Run the merge script to merge the models
time python merge.py \
    --base-dir "$BASEDIR" \
    --training-name "$TRAININGNAME" \
    --model-kind "$MODEL_KIND" \
    --export-prefix "merged"

# Check if merge.py completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to merge the models"
    exit 1
fi

# Export the merged model for selfplay
echo "Exporting noisy model for selfplay"
(
    time ./export_model_for_selfplay.sh "merged" "$BASEDIR" "1"
)

echo "Models merged and exported successfully."

exit 0
}

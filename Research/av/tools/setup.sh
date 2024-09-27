# Call source tools/setup.sh

if [ -n "${PROJECT_ROOT_DIR+x}" ]; then
    # We already ran setup.sh so just return
    echo "Already ran setup.sh. Skipping..."
    return
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT_DIR"

echo "Finished setting up"
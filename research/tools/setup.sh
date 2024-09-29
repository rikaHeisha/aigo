# Call source tools/setup.sh

# if [ -n "${PROJECT_ROOT_DIR}" ]; then
#     echo "Already ran setup.sh. Skipping..."
# fi

export PROJECT_ROOT_DIR=$( cd "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
export PYTHONPATH="$PROJECT_ROOT_DIR"

echo "Finished setting up"
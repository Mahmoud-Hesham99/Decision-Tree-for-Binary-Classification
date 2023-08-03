#!/bin/bash

# Get the first argument passed to the script (the command)
command=$1

# Shift all arguments down by one (removing the first argument)
shift

echo "Running $command"

# Perform the command specified by the first argument
case $command in
  # If the command is "train", run the train.py script with all remaining arguments
  train)
    python /opt/src/train.py -t"$@"
    ;;

  # If the command is "predict", run the predict.py script with all remaining arguments
  predict)
    python /opt/src/predict.py "$@"
    ;;

  # If the command is "serve", run the serve.py script with all remaining arguments
  serve)
    python /opt/src/serve.py "$@"
    ;;

  # If the command is "standby", keep the container running without doing anything
  standby)
    tail -f /dev/null
    ;;

  # If an unknown command is passed, display an error message and exit with a non-zero status code
  *)
    echo "Invalid or missing command. Please specify one of the following commands:"
    echo "train: Train the model"
    echo "predict: Make predictions with the model"
    echo "serve: Serve the model for inference"
    echo "standby: Run the container in standby mode to keep the container running without doing anything"
    exit 1
    ;;
esac
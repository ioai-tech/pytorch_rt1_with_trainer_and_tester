#!/bin/bash

# Assigning the Python script name to a variable
python_script="test_model.py"

# Getting input arguments from the command line
epoch=$1
task="pick"
ckpt_dir=$2
model_name=$3
gpu_name=$4
cam_view=$5
using_pos_and_orn=$6
num_threads=$7
initial_c_value=0
override_exist=1

# Function to run the Python script with specified arguments
run_python_script() {
  local c_value=$1
  python $python_script -t $task -d $ckpt_dir -m $model_name -c $cam_view -e $epoch -g $gpu_name -u $using_pos_and_orn -p $c_value -n $num_threads
}

# Function to handle cleanup when Ctrl+C is detected
cleanup() {
  echo "Ctrl+C detected. Killing all threads..."
  kill 0
  exit 1
}

# Function to run multiple threads of the Python script
run_threads() {
  local num_threads=$1
  local c_value=$2
  local increment=1

  # Trap Ctrl+C to call the cleanup function
  trap cleanup INT

  # Run the Python script in multiple threads
  for ((i=0; i<num_threads; i++))
  do
    run_python_script $c_value &  # Run Python script with specified arguments in the background
    c_value=$((c_value+increment))  # Increment the value of 'c_value'
  done

  wait  # Wait for all background processes to finish
}

# Run the specified number of threads with initial 'c_value'
run_threads $num_threads $initial_c_value

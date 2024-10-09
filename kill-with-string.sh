#!/bin/bash

# Assign command-line arguments to variables
string1="$1"
string2="$2"
dryrun="$3"

# Check if string1 is provided
if [[ -z "$string1" ]]; then
    echo "Usage: $0 <string1> [string2] [dryrun]"
    echo "  <string1>: Required. Processes containing this string will be targeted."
    echo "  [string2]: Optional. Exclude processes containing this string."
    echo "  [dryrun]: Optional. If set to 'dryrun', the script will only display the processes without killing them."
    exit 1
fi

# Iterate over all PIDs matching 'string1'
for pid in $(pgrep -f "$string1"); do
    # Get the full command line of the process
    cmdline=$(ps -p "$pid" -o cmd=)
    
    # Initialize a flag to determine if the process should be killed
    should_kill=true
    
    # If 'string2' is provided, check that it is not in the command line
    if [[ -n "$string2" && "$cmdline" == *"$string2"* ]]; then
        should_kill=false
    fi
    
    if [[ "$should_kill" == true ]]; then
        if [[ "$dryrun" == "dryrun" ]]; then
            # Dry run: just echo the PID and command line
            echo "Would kill process $pid ($cmdline)"
        else
            # Kill the process
            kill "$pid"
            echo "Killed process $pid ($cmdline)"
        fi
    fi
done

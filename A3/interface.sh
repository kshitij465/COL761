#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <question_number> [part] [path_to_dataset]"
    exit 1
fi

# Run code for question 1
if [ "$1" == "1" ]; then
    python q1.py  # Replace with the actual command to run code for question 1

# Run code for question 2
elif [ "$1" == "2" ]; then
    # Check if part and path to dataset are provided
    if [ $# -lt 3 ]; then
        echo "Usage: $0 2 <part> <path_to_dataset>"
        exit 1
    fi

    part="$2"
    dataset="$3"

    # Run code based on the provided part
    case "$part" in
        a)
            python code_for_question2a.py "$dataset"
            ;;
        b)
            python code_for_question2b.py "$dataset"
            ;;
        c)
            python q2.py "$dataset"
            python partb_lsh.py "$dataset" c
            python partb_sequential.py "$dataset"
            ;;
        d)
            python partb_lsh.py "$dataset" d
            ;;
        *)
            echo "Invalid part. Valid parts are: a, b, c, d"
            exit 1
            ;;
    esac
else
    echo "Invalid question number. Valid numbers are: 1, 2"
    exit 1
fi

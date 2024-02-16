#!/bin/bash


# Get the filename from the command line argument
filename="$1"
resultpath="$2"

# Check if the file exists


# Count the number of '#' characters in the file and store it in a variable
count=$(grep -o "#" "$filename" | wc -l)

# Print the result
echo "Number of Graphs in $filename: $count"


python3 ./format.py "$filename"

# Define the output file for 'format_change'

support_values=("95" "50" "25" "10" "5")  # Add more values as needed

for support_fsg in "${support_values[@]}"; do
    support_gspan=$(awk "BEGIN { printf \"%.2f\", $support_fsg / 100 }")
    support_gaston=$(awk "BEGIN { printf \"%.2f\", $support_gspan * $count }")
    echo "Support $support_fsg" >> "time_outputfsg.txt"
    { time  ./fsg/fsg -s "$support_fsg" "$filename-fsg" ; } 2>> "time_outputfsg.txt"
    { time  ./gspan/gSpan-64 -s "$support_gspan" -o -f "$filename-gspan"; } 2>> "time_outputgspan.txt"
    { time  ./gaston-1.1/gaston "$support_gaston" "$filename-gaston"; } 2>> "time_outputgaston.txt"
done

python3 make_plots.py "$resultpath"
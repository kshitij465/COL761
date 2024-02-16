import re
import os
import matplotlib.pyplot as plt

def extract_support(line):
    match = re.match(r"Support (\d+)", line)
    if match:
        return int(match.group(1))
    return None

def extract_time(line):
    match = re.match(r"real\t(\d+m\d+\.\d+s)", line)
    if match:
        time_str = match.group(1)
        minutes, seconds = float(time_str.split("m")[0]), float(time_str.split("m")[1][:-1])
        return minutes * 60 + seconds
    return None

def parse_results(file_path):
    support_values = []
    fsg_times = []
    gspan_times = []
    gaston_times = []

    current_support = None
    current_algorithm = None
    algorithms = ("fsg", "gspan", "gaston")

    with open(file_path+"fsg.txt", "r") as file:
        for line in file:
            support = extract_support(line)
            if support is not None:
                current_support = support
                support_values.append(current_support)

            time = extract_time(line)
            if time is not None:
                fsg_times.append(time)
    with open(file_path+"gaston.txt", "r") as file:
        for line in file:

            time = extract_time(line)
            if time is not None:
                gaston_times.append(time)
    with open(file_path+"gspan.txt", "r") as file:
        for line in file:

            time = extract_time(line)
            if time is not None:
                gspan_times.append(time)

    return support_values, fsg_times, gspan_times, gaston_times

def plot_running_time(support_values, fsg_times, gspan_times, gaston_times, result_path):
    plt.figure(figsize=(10, 10))
    plt.plot(support_values, fsg_times, marker='o', label='FSG')
    plt.plot(support_values, gspan_times, marker='o', label='gSpan')
    plt.plot(support_values, gaston_times, marker='o', label='Gaston')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Support Threshold')
    plt.ylabel('Execution Time (seconds)')
    plt.savefig(os.path.join(result_path, 'q1_plot_cs1200340.png'))
    plt.close()

def main(result_path):
    support_values, fsg_times, gspan_times, gaston_times = parse_results("time_output")
    plot_running_time(support_values, fsg_times, gspan_times, gaston_times, result_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <result_path>")
        sys.exit(1)
    main(sys.argv[1])
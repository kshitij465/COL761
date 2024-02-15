import sys
import os
import time
import numpy as np
import subprocess
from collections import OrderedDict
import networkx as nx
import networkx.algorithms.isomorphism as isomorph

def main():
    input_file_path = sys.argv[1]
    encoding_file = os.path.join(sys.argv[2], 'features_cs1200340.txt')
    input_file = open(input_file_path, 'r')

    gspan_input_active_path = input_file_path + "-gspan_a"
    gspan_input_inactive_path = input_file_path + "-gspan_i"
    gspan_input_active = open(gspan_input_active_path, 'w')
    gspan_input_inactive = open(gspan_input_inactive_path, 'w')
    graph_map_i = []
    graph_map_a = []
    node_id_to_int = {}
    node_count = 0
    active_graph_num = 0
    inactive_graph_num = 0
    graph_num = 0

    labels = OrderedDict()
    total_graph_data = []


    graph_ids = []
    graph_ids_i = OrderedDict()
    graph_ids_a = OrderedDict()

    while True:
        line = input_file.readline()
        if not line:
            break
        s = line.strip()
        parts = s.split()

        graph_id, label = [int(part) for part in parts[1:]]
        tst = [int(part) for part in parts[1:]]
        graph_ids.append(graph_id)
        labels[graph_id] = label
        G = nx.Graph()
        if label == 1:
            gspan_input_active.write("t # " + str(active_graph_num) + "\n")
            graph_ids_a[int(graph_num)] = active_graph_num
            active_graph_num += 1
            while True:
                current_pos = input_file.tell()
                next_line_first_char = input_file.read(1)
                input_file.seek(current_pos)
                if (not next_line_first_char) or next_line_first_char == '#':
                    break
                line = input_file.readline()
                gspan_input_active.write(line)

                integers = [int(x) for x in line.split()[1:]]
                line = line.split(" ")
                if line[0]=='v':
                    G.add_node(integers[0], label=integers[1])
                elif line[0]=='e':
                    G.add_edge(integers[0], integers[1], label=integers[2])
                
        else :
            gspan_input_inactive.write("t # " + str(inactive_graph_num) + "\n")
            graph_ids_i[int(graph_num)] = inactive_graph_num
            inactive_graph_num+=1
            while True:
                current_pos = input_file.tell()
                next_line_first_char = input_file.read(1)
                input_file.seek(current_pos)
                if (not next_line_first_char) or next_line_first_char == '#':
                    break
                line = input_file.readline()
                gspan_input_inactive.write(line)

                integers = [int(x) for x in line.split()[1:]]
                line = line.split(" ")
                if line[0]=='v':
                    G.add_node(integers[0], label=integers[1])
                elif line[0]=='e':
                    G.add_edge(integers[0], integers[1], label=integers[2])
        total_graph_data.append(G)
        graph_num += 1
    print(graph_num, active_graph_num, inactive_graph_num)

    min_sup_a = 0.5
    min_sup_i = 0.6
    gspan_input_active.close()
    gspan_input_inactive.close()

    log_file_path = "gspan_log.txt"
    log_file = open(log_file_path, "w") 
    
    # run gspan to get subgraph_file

    subprocess.run("./gspan/gSpan-64 -f "+gspan_input_active_path+" -s "+str(min_sup_a)+" -o -i", stdout=log_file, stderr=subprocess.STDOUT, shell=True, check=True)
    subprocess.run("./gspan/gSpan-64 -f "+gspan_input_inactive_path+" -s "+str(min_sup_i)+" -o -i", stdout=log_file, stderr=subprocess.STDOUT, shell=True, check=True)

    log_file.close()
    subgraph_file_i = gspan_input_inactive.name+".fp"
    subgraph_file_a = gspan_input_active.name+".fp"

    total_graphs = inactive_graph_num + active_graph_num
    num_active = active_graph_num
    num_inactive = inactive_graph_num
    sup_active = round((num_active/total_graphs),3)
    sup_inactive = round((num_inactive/total_graphs),3)

    print(sup_active)
    print(sup_inactive)


    active_subgraphs = []
    label_1_graph = []

    sgfa = open(subgraph_file_a, 'r')
    line = sgfa.readline()

    while line: 
        
        line = line.strip()
        line = line.split()
        if len(line)!=0:
            if line[0]=='t':
                line = sgfa.readline()
                integers = [int(x) for x in line.split()[1:]]
                G = nx.Graph()
                while line[0]!='x':
                    if line[0]=='v':
                        G.add_node(integers[0], label=integers[1])
                    elif line[0]=='e':
                        G.add_edge(integers[0], integers[1], label=integers[2])
                    line = sgfa.readline()
                    integers = [int(x) for x in line.split()[1:]]
                if line[0]=='x':
                    label_1_graph.append(integers)
                active_subgraphs.append(G)
        line = sgfa.readline()
    sgfa.close()



    label_0_graph = []
    inactive_subgraphs = []
    sgfi = open(subgraph_file_i, 'r')

    line = sgfi.readline()

    while line: 
        line = line.strip()
        line = line.split()
        if len(line)!=0:
            if line[0]=='t':
                line = sgfi.readline()
                integers = [int(x) for x in line.split()[1:]]
                G = nx.Graph()
                while line[0]!='x':
                    if line[0]=='v':
                        G.add_node(integers[0], label=integers[1])
                    elif line[0]=='e':
                        G.add_edge(integers[0], integers[1], label=integers[2])
                    line = sgfi.readline()
                    integers = [int(x) for x in line.split()[1:]]
                if line[0]=='x':
                    label_0_graph.append(integers)
                inactive_subgraphs.append(G)
        line = sgfi.readline()
    sgfi.close()



            # Finding Not_in_other active graphs
    Not_in_other_a = [
        i  # Index of active graph
        for i, active_graph in enumerate(active_subgraphs)  # Loop over indices and active graphs
        if not any(
            nx.is_isomorphic(active_graph, inactive_graph, node_match=isomorph.categorical_node_match('label', ''), edge_match=isomorph.categorical_node_match('label', ''))
            for inactive_graph in inactive_subgraphs  # Loop over inactive graphs
        )
    ]

    # Finding Not_in_other inactive graphs
    Not_in_other_i = list(range(len(inactive_subgraphs)))

    # Removing duplicates from Not_in_other active graphs
    Not_in_other_a = list(set(Not_in_other_a))

    print("1: Not_in_other inactive", len(Not_in_other_i))
    print("1: Not_in_other active",len(Not_in_other_a))

   # Graph objects
    Not_in_other_subgraphs_1 = [inactive_subgraphs[Not_in_other_i[i]] for i in range(min(40,len(Not_in_other_i)))]
    Not_in_other_subgraphs_2=[active_subgraphs[Not_in_other_a[i]] for i in range(min(40,len(Not_in_other_a)))]
    Not_in_other_subgraphs=Not_in_other_subgraphs_1+Not_in_other_subgraphs_2
    # indexes
    Graphs_1 = [label_0_graph[Not_in_other_i[i]] for i in range(min(40,len(Not_in_other_i)))]
    Graphs_2 = [label_1_graph[Not_in_other_a[i]] for i in range(min(40,len(Not_in_other_a)))]
    Graphs = Graphs_1+Graphs_2
    labels_featues = [0*i for i in range(min(40,len(Not_in_other_i)))]


    for i in range(min(40,len(Not_in_other_a))):
        labels_featues.append(1)

    print(f'Found Not_in_other subgraphs')


    # read subgraphs and form binary feature vector for each graph
    total_subgraphs = len(Not_in_other_subgraphs)
    print("total Not_in_other subgraphs: ", total_subgraphs)

    zero_arr = np.zeros(total_subgraphs, dtype=np.int8)
    itera = np.arange(total_subgraphs)
    labelings = {i: dict(zip(itera, zero_arr)) for i in range(total_graphs)}

    
    items_graph_ids_i = list(graph_ids_i.items())
    items_graph_ids_a = list(graph_ids_a.items())

    for i in range(total_subgraphs):
        label = labels_featues[i]
        ord_graph_id=0
        for freq_graph in Graphs[i]:
            if label==0:
                ord_graph_id = items_graph_ids_i[freq_graph][0]
            else:
                ord_graph_id = items_graph_ids_a[freq_graph][0]
            labelings[ord_graph_id][i]=1

    items_labels = list(labels.items())
    # check for inactive freq subgraphs in active graphs
    for i in range(len(total_graph_data)):
        G = total_graph_data[i]
        for j in range(len(labels_featues)):
            if labels_featues[j]==items_labels[i][1]:
                GM = isomorph.GraphMatcher(G, Not_in_other_subgraphs[j],node_match=isomorph.categorical_node_match('label', ''),  edge_match=isomorph.categorical_node_match('label', ''))
                is_isomorph = GM.subgraph_is_isomorphic()
                if is_isomorph:
                    labelings[i][j]=1

    print(f'Made binary features for Train')

    mappings=[]
    for i in range(total_graphs):
        actual_graph_id = graph_ids[i]
        mappings.append(list(labelings[i].values()))
    with open(encoding_file, 'w') as fw:
        for i in range(total_graphs):
            temp=""
            temp+=str(graph_ids[i])
            temp+=" #"
            for x in mappings[i]:
                temp+=" "
                temp+=str(x)
            fw.write(temp+'\n')
    fw.close()


    print(f'Made binary features for Train')
    # exit()


if __name__ == "__main__":
    main()


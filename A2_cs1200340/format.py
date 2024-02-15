import sys

class Graph:
    def __init__(self):
        self.graph_id = ""
        self.num_nodes = 0
        self.node_id = []
        self.num_edges = 0
        self.edges = []
        self.edge_id = []

def main():
    input_file_path = sys.argv[1]
    input_file = open(input_file_path, 'r')
    graph_map = []
    node_id_to_int = {}
    node_count = 0

    read_file = True
    while read_file:
        line = input_file.readline()
        if not line:
            break
        s = line.strip()
        if not s:
            continue
        if not s.startswith('#'):
            break
        a = Graph()
        a.graph_id = s
        a.num_nodes = int(input_file.readline())
        for i in range(a.num_nodes):
            c = input_file.readline().strip()
            if c not in node_id_to_int:
                node_id_to_int[c] = node_count
                node_count += 1
            a.node_id.append(c)
            
        a.num_edges = int(input_file.readline())
        for _ in range(a.num_edges):
            p1, p2, p3 = map(str, input_file.readline().split())
            a.edges.append((int(p1), int(p2)))
            a.edge_id.append(p3)
        graph_map.append(a)
    
    print(len(graph_map))

    target_format = sys.argv[2]

    if target_format == "fsg":
        with open(input_file_path + "-fsg", 'w') as out_file:
            for a in graph_map:
                out_file.write("t " + a.graph_id + "\n")
                for i in range(a.num_nodes):
                    out_file.write("v " + str(i) + " " + a.node_id[i] + "\n")
                for i in range(a.num_edges):
                    out_file.write("e " + str(a.edges[i][0]) + " " + str(a.edges[i][1]) + " " + a.edge_id[i] + "\n")

    if target_format == "gspan":
        with open(input_file_path + "-gspan", 'w') as out_file:
            cnt = 0
            for a in graph_map:
                out_file.write("t # " + str(cnt) + "\n")
                for i in range(a.num_nodes):
                    out_file.write("v " + str(i) + " " + str(node_id_to_int[a.node_id[i]]) + "\n")
                for i in range(a.num_edges):
                    out_file.write("e " + str(a.edges[i][0]) + " " + str(a.edges[i][1]) + " " + a.edge_id[i] + "\n")
                cnt += 1

    if target_format == "gaston":
        with open(input_file_path + "-gaston", 'w') as out_file:
            cnt = 0
            for a in graph_map:
                out_file.write("t # " + str(cnt) + "\n")
                for i in range(a.num_nodes):
                    out_file.write("v " + str(i) + " " + str(node_id_to_int[a.node_id[i]]) + "\n")
                for i in range(a.num_edges):
                    out_file.write("e " + str(a.edges[i][0]) + " " + str(a.edges[i][1]) + " " + a.edge_id[i] + "\n")
                cnt += 1

    for a in graph_map:
        del a

if __name__ == "__main__":
    main()

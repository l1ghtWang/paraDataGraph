#pragma once

#include "cudaErrorCheck.cuh"
#include "gflags/gflags.h"

DEFINE_uint32(partition_size_MB, 32, "partition size in MB");

struct Edge
{
    uint32_t source;
    uint32_t end;
};

class Graph
{
private:
    uint32_t num_nodes;
    uint32_t num_edges;
    bool isWeighted;
    uint32_t *row_node_array;
    uint32_t *edgeList;
    std::vector<uint32_t> vec_partition_start_node;

public:
    Graph() : num_nodes(0), num_edges(0), isWeighted(false), row_node_array(nullptr), edgeList(nullptr){};
    ~Graph(){};

    std::string get_File_Extension(std::string fileName)
    {
        if (fileName.find_last_of(".") != std::string::npos)
            return fileName.substr(fileName.find_last_of(".") + 1);
        return "";
    }

    void print_partitioned_graph_info()
    {
        for (uint32_t i = 0; i < vec_partition_start_node.size() - 1; i++)
        {
            std::cout << "Partition " << i << ": ";
            std::cout << "start node: " << vec_partition_start_node[i] << " end node: " << vec_partition_start_node[i + 1] - 1 << "";
            std::cout << std::endl;
        }
    }
    void loadGraph(std::string filename)
    {

        Stopwatch sw(true);
        std::cout << "Loading graph from " << filename << std::endl;
        std::string fileExtension = get_File_Extension(filename);
        if (fileExtension == "el")
            isWeighted = false;
        else if (fileExtension == "wel")
            isWeighted = true;
        else
        {
            std::cout << "File extension not supported!" << std::endl;
            exit(1);
        }

        std::ifstream infile;
        infile.open(filename);
        std::stringstream ss;
        std::string line;
        uint32_t edgeCounter = 0;
        if (isWeighted)
        {
        }
        else
        {
            std::vector<Edge> edges;
            Edge newEdge;
            uint32_t max = 0;
            while (getline(infile, line))
            {
                ss.str("");
                ss.clear();
                ss << line;

                ss >> newEdge.source;
                ss >> newEdge.end;

                edges.push_back(newEdge);
                edgeCounter++;

                if (max < newEdge.source)
                    max = newEdge.source;
                if (max < newEdge.end)
                    max = newEdge.end;
            }
            infile.close();
            num_nodes = max + 1;
            num_edges = edgeCounter;
            row_node_array = new uint32_t[num_nodes + 1];
            CHECK_CUDA_ERROR(cudaMallocHost(&edgeList, (num_edges) * sizeof(uint32_t)));

            uint32_t *degree = new uint32_t[num_nodes];
            for (uint32_t i = 0; i < num_nodes; i++)
                degree[i] = 0;
            for (uint32_t i = 0; i < num_edges; i++)
            {
                degree[edges[i].source]++;
            }

            uint32_t counter = 0;
            uint32_t max_degree = 0;
            uint32_t num_edgesInOnePartition = FLAGS_partition_size_MB * 1024 * 1024 / sizeof(uint32_t);
            uint32_t counter_edgesInOnePartition = 0;

            vec_partition_start_node.push_back(0);

            for (uint32_t i = 0; i < num_nodes; i++)
            {
                if (max_degree < degree[i])
                    max_degree = degree[i];
                row_node_array[i] = counter;
                counter = counter + degree[i];
                counter_edgesInOnePartition = counter_edgesInOnePartition + degree[i];
                if (counter_edgesInOnePartition > num_edgesInOnePartition)
                {
                    vec_partition_start_node.push_back(i + 1);
                    counter_edgesInOnePartition = 0;
                }
            }
            if (counter_edgesInOnePartition > 0)
                vec_partition_start_node.push_back(num_nodes);
            row_node_array[num_nodes] = num_edges;
            uint32_t *outDegreeCounter = new uint32_t[num_nodes];
            uint32_t location;
            for (uint32_t i = 0; i < num_edges; i++)
            {
                location = row_node_array[edges[i].source] + outDegreeCounter[edges[i].source];
                edgeList[location] = edges[i].end;
                outDegreeCounter[edges[i].source]++;
            }

            printf("The graph has %d vertices, and %ld edges (average_degree: %f, max_degree: %d)\n", num_nodes, num_edges, (float)num_edges / num_nodes, max_degree);
            print_partitioned_graph_info();

            edges.clear();
            delete[] degree;
            delete[] outDegreeCounter;
        }
        sw.stop();
        std::cout << "Loading graph Time: " << sw.ms() << " ms" << std::endl;
    };
};
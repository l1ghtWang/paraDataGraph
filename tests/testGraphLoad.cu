#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include <gflags/gflags.h>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cuda_error(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast_cuda_error(__FILE__, __LINE__)
void checkLast_cuda_error(const char *const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}



DEFINE_string(test, "abc", "just a test");

// using namespace std;
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
public:
    Graph() : num_nodes(0), num_edges(0), isWeighted(false), row_node_array(nullptr),edgeList(nullptr){};
    ~Graph(){};

    std::string GetFileExtension(std::string fileName)
    {
        if (fileName.find_last_of(".") != std::string::npos)
            return fileName.substr(fileName.find_last_of(".") + 1);
        return "";
    }

    void loadGraph(std::string filename)
    {

        Stopwatch sw(true);
        std::cout << "Loading graph from " << filename << std::endl;
        std::string fileExtension = GetFileExtension(filename);
        if(fileExtension == "el")
            isWeighted = false;
        else if(fileExtension == "wel")
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
            for (uint32_t i = 0; i < num_nodes; i++)
            {
                if (max_degree < degree[i])
                    max_degree = degree[i];
                row_node_array[i] = counter;
                counter = counter + degree[i];
            }
            row_node_array[num_nodes] = num_edges;
            uint32_t *outDegreeCounter = new uint32_t[num_nodes];
            uint32_t location;
            for (uint32_t i = 0; i < num_edges; i++)
            {
                location = row_node_array[edges[i].source] + outDegreeCounter[edges[i].source];
                edgeList[location] = edges[i].end;
                outDegreeCounter[edges[i].source]++;
            }

            printf("The graph has %d vertices, and %ld edges (average_degree: %f, max_degree: %d)\n", num_nodes, num_edges, (float) num_edges / num_nodes, max_degree);

            edges.clear();
            delete[] degree;
            delete[] outDegreeCounter;
        }
        sw.stop();
        std::cout << "Loading graph Time: " << sw.ms() << " ms" << std::endl;
    };
};

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    Stopwatch sw(true);
    std::cout << "Testing Graph Load" << std::endl;
    Graph g;

    std::string testFilename = "/home/share/graph_data/raw/datasets/Google/web-Google.el";
    g.loadGraph(testFilename);
    sw.stop();
    std::cout << "Time: " << sw.ms() << " ms" << std::endl;
    std::cout << "FLAGS_test: " << FLAGS_test << std::endl;
    return 0;
}
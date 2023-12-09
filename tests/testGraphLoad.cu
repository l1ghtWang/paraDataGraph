#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include <gflags/gflags.h>
#include "graph.cuh"
#include "cudaErrorCheck.cuh"




DEFINE_string(test, "abc", "just a test");
DECLARE_string(graphInputfile);
// DEFINE_uint32(partition_size_MB, 32, "partition size in MB");
DECLARE_uint32(partition_size_MB);
// using namespace std;




int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    Stopwatch sw(true);
    std::cout << "Testing Graph Load" << std::endl;
    Graph g;

    std::string graphInputfile = "/home/share/graph_data/raw/datasets/Google/web-Google.el";
    g.loadGraph(graphInputfile);
    sw.stop();
    std::cout << "Time: " << sw.ms() << " ms" << std::endl;
    std::cout << "FLAGS_test: " << FLAGS_test << std::endl;
    return 0;
}
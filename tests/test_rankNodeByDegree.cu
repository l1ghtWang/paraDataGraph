#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include <gflags/gflags.h>
#include "graph.cuh"
#include "cudaErrorCheck.cuh"
#include <queue>

DECLARE_string(graphInputfile);
// DEFINE_uint32(partition_size_MB, 32, "partition size in MB");
DECLARE_uint32(partition_size_MB);
DEFINE_uint32(source_node, 0, "source node");
DECLARE_uint32(nStreams);
DEFINE_bool(check, false, "check result");
DEFINE_uint32(num_run, 10, "number of run");




int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    Graph g;
    g.rankNodeByDegree(FLAGS_graphInputfile);
    printf("source node: %u\n", FLAGS_source_node);

    return 0;
}
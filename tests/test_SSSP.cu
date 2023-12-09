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

namespace SSSP
{
    __forceinline__ __device__ bool kernel_isActiveNode(uint32_t node, uint32_t buffer, uint32_t value)
    {
        return buffer < value;
    }

    __global__ 
    void kernel_InitGraph(uint32_t work_size,
                          uint32_t *dev_node_value_datum,
                          uint32_t *dev_node_buffer_datum,
                          uint32_t source_node)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = i;
            if (node == source_node)
            {
                dev_node_buffer_datum[node] = 0;
            }
            else
            {
                dev_node_buffer_datum[node] = UINT32_MAX;
            }
            dev_node_value_datum[node] = UINT32_MAX;
        }
    }

    __global__ 
    void kernel_RebuildWorklist(uint32_t work_size,
                                uint32_t *dev_node_value_datum,
                                uint32_t *dev_node_buffer_datum,
                                uint32_t *dev_worklist,
                                uint32_t *dev_worklist_counter)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = i;
            if(SSSP::kernel_isActiveNode(node, dev_node_buffer_datum[node], dev_node_value_datum[node]))
            {
                uint32_t location = atomicAdd(dev_worklist_counter, 1);
                dev_worklist[location] = node;
            }
        }
    }

    __global__ 
    void kernel_RebuildWorklist_perPartition(uint32_t startNodeIdx,
                                            uint32_t work_size,
                                            uint32_t *dev_node_value_datum,
                                            uint32_t *dev_node_buffer_datum,
                                            uint32_t *dev_worklist, 
                                            uint32_t *dev_worklist_counter)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = i + startNodeIdx;
            if (SSSP::kernel_isActiveNode(node, dev_node_buffer_datum[node], dev_node_value_datum[node]))
            {
                uint32_t location = atomicAdd(dev_worklist_counter, 1);
                dev_worklist[location] = node;
            }
        }
    }
    __global__ void kernel_SSSP_sync_push_dd(uint32_t work_size,
                                             uint32_t *dev_node_value_datum,
                                             uint32_t *dev_node_buffer_datum,
                                             uint32_t *dev_worklist,
                                             uint32_t *dev_edgeList,
                                             uint32_t partition_start_edge_offset,
                                             uint32_t *dev_arr_node_edgeStartIndex_CSR)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = dev_worklist[i];
            if(dev_node_value_datum[node] > dev_node_buffer_datum[node])
            {
                dev_node_value_datum[node] = dev_node_buffer_datum[node];
                // uint32_t edge_loc =
                uint32_t edge_start = dev_arr_node_edgeStartIndex_CSR[node] - partition_start_edge_offset;
                uint32_t edge_end = dev_arr_node_edgeStartIndex_CSR[node + 1] - partition_start_edge_offset;
                for (uint32_t edge = edge_start; edge < edge_end; edge++)
                {
                    uint32_t dst_node = dev_edgeList[edge];
                    uint32_t new_dist = dev_node_buffer_datum[node] + 1;
                    atomicMin(&dev_node_buffer_datum[dst_node], new_dist);
                }
            }

        }
    }

    std::vector<uint32_t> host_sssp(Graph &graph, uint32_t source_node)
    {
        std::vector<uint32_t> distances(graph.get_num_nodes(), UINT32_MAX);
        std::queue<uint32_t> work;

        distances[source_node] = 0;
        work.push(source_node);
        while (!work.empty())
        {
            uint32_t node = work.front();
            work.pop();
            uint32_t edge_start = graph.get_host_array_node_edgeStartIndex_CSR(node);
            uint32_t edge_end = graph.get_host_array_node_edgeStartIndex_CSR(node + 1);
            for (uint32_t edge = edge_start; edge < edge_end; edge++)
            {
                uint32_t dst_node = graph.get_host_array_edgeList_ptr()[edge];
                uint32_t new_dist = distances[node] + 1;
                if (new_dist < distances[dst_node])
                {
                    distances[dst_node] = new_dist;
                    work.push(dst_node);
                }
            }
        }
        return distances;
    }

    int SSSPCheckErrors(const std::vector<uint32_t> &distances, const std::vector<uint32_t> &regression)
    {
        if (distances.size() != regression.size())
        {
            return std::abs((long long)distances.size() - (long long)regression.size());
        }

        int over_errors = 0, miss_errors = 0;
        std::vector<int> over_error_indices, miss_error_indices;

        uint32_t
            max_over_delta = 0,
            max_miss_delta = 0;

        for (int i = 0; i < regression.size(); ++i)
        {
            uint32_t hv = distances[i];
            uint32_t rv = regression[i];

            if (hv > rv)
            {
                ++over_errors;
                over_error_indices.push_back(i);
                max_over_delta = std::max(max_over_delta, hv - rv);
            }
            else if (hv < rv)
            {
                ++miss_errors;
                miss_error_indices.push_back(i);
                max_miss_delta = std::max(max_miss_delta, rv - hv);
            }
        }

        if (miss_errors > 0)
            printf("Miss errors: %d\n\n", miss_errors);

        if (over_errors > 0)
            printf("Over errors: %d\n\n", over_errors);

        return (miss_errors + over_errors);

    }
};

bool initial_Framework(Graph &g, cudaStream_t* streams)
{
    SSSP::kernel_InitGraph<<<g.get_num_nodes() / 512 + 1, 512>>>(g.get_num_nodes(),
                                                                 g.get_device_value_ptr(),
                                                                 g.get_device_buffer_ptr(),
                                                                 FLAGS_source_node);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    for (int i = 0; i < g.get_num_partitions(); i++)
    {
        uint32_t startNodeIdx = g.get_partition_start_node(i);
        uint32_t numNodesInPartition = g.get_partition_start_node(i + 1) - g.get_partition_start_node(i);
        SSSP::kernel_RebuildWorklist_perPartition<<<numNodesInPartition / 512 + 1, 512, 0, streams[i % FLAGS_nStreams]>>>(startNodeIdx,
                                                                                                                          numNodesInPartition,
                                                                                                                          g.get_device_value_ptr(),
                                                                                                                          g.get_device_buffer_ptr(),
                                                                                                                          g.get_vector_device_worklist_ptr()[i],
                                                                                                                          g.get_vector_device_worklist_counter_ptr()[i]);
    }
    // for (int i = 0; i < FLAGS_nStreams; i++)
    // {
    //     cudaStreamSynchronize(streams[i]);
    // }

    bool isConverged = true;
    for (int i = 0; i < g.get_num_partitions(); i++)
    {
        uint32_t numItemInWorkList = g.get_worklist_counter_value(i, streams[i % FLAGS_nStreams]);
        g.set_partition_numActiveNodes(i, numItemInWorkList);
        isConverged &= (numItemInWorkList == 0);
        // printf("partition: %d numItemInWorkList: %u\n", i, numItemInWorkList);
    }
    CHECK_LAST_CUDA_ERROR();
    return isConverged;
}

void computeGraph(Graph &g, cudaStream_t* streams)
{

}

bool rebuild_workList_check_converge(Graph &g, cudaStream_t* streams)
{
    for (int i = 0; i < g.get_num_partitions(); i++)
    {
        uint32_t streamIdx = i % FLAGS_nStreams;
        uint32_t startNodeIdx = g.get_partition_start_node(i);
        uint32_t numNodesInPartition = g.get_partition_start_node(i + 1) - g.get_partition_start_node(i);
        cudaMemsetAsync(g.get_vector_device_worklist_counter_ptr()[i], 0, sizeof(uint32_t), streams[streamIdx]);
        SSSP::kernel_RebuildWorklist_perPartition<<<numNodesInPartition / 512 + 1, 512, 0, streams[streamIdx]>>>(startNodeIdx,
                                                                                                                 numNodesInPartition,
                                                                                                                 g.get_device_value_ptr(),
                                                                                                                 g.get_device_buffer_ptr(),
                                                                                                                 g.get_vector_device_worklist_ptr()[i],
                                                                                                                 g.get_vector_device_worklist_counter_ptr()[i]);
    }
    // for (int i = 0; i < FLAGS_nStreams; i++)
    // {
    //     cudaStreamSynchronize(streams[i]);
    // }
    // CHECK_LAST_CUDA_ERROR();
    bool isConverged = true;
    for (int i = 0; i < g.get_num_partitions(); i++)
    {
        uint32_t streamIdx = i % FLAGS_nStreams;

        uint32_t numItemInWorkList = g.get_worklist_counter_value(i, streams[streamIdx]);
        g.set_partition_numActiveNodes(i, numItemInWorkList);
        isConverged &= (numItemInWorkList == 0);
    }
    return isConverged;
}

void start(Graph g, cudaStream_t* streams)
{
    bool isConverged = false;
    isConverged = initial_Framework(g, streams);

    uint32_t iterationCount = 0;
    while (!isConverged)
    {
        for (int i = 0; i < g.get_num_partitions(); i++)
        {
            uint32_t streamIdx = i % FLAGS_nStreams;
            uint32_t startNodeIdx = g.get_partition_start_node(i);
            uint32_t numNodesInPartition = g.get_partition_start_node(i + 1) - g.get_partition_start_node(i);
            uint32_t numEdgesInPartition = g.get_partition_start_edge(i + 1) - g.get_partition_start_edge(i);
            uint32_t *dev_edgeList_curPartiton = g.get_device_edgeList_ptr(streamIdx);
            cudaMemcpyAsync(dev_edgeList_curPartiton,
                            g.get_host_array_edgeList_ptr() + g.get_partition_start_edge(i),
                            numEdgesInPartition * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            streams[streamIdx]);

            SSSP::kernel_SSSP_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, streams[streamIdx]>>>(numNodesInPartition,
                                                                                                          g.get_device_value_ptr(),
                                                                                                          g.get_device_buffer_ptr(),
                                                                                                          g.get_vector_device_worklist_ptr()[i],
                                                                                                          dev_edgeList_curPartiton,
                                                                                                          g.get_partition_start_edge(i),
                                                                                                          g.get_device_node_edgeStartIndex_CSR_ptr());
        }

        // in theory, we don't need this sync, but the iteration number increase when we remove this sync
        for (int i = 0; i < FLAGS_nStreams; i++)
        {
            cudaStreamSynchronize(streams[i]);
        }
        CHECK_LAST_CUDA_ERROR();

        isConverged = rebuild_workList_check_converge(g, streams);
        iterationCount++;
        if (iterationCount == 1000)
        {
            break;
        }
    }

    std::cout << "total iteration: " << iterationCount << std::endl;
}

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    Stopwatch sw_overall_time(true);
    Graph g;
    g.loadGraph(FLAGS_graphInputfile);
    printf("source node: %u\n", FLAGS_source_node);

    cudaStream_t streams[FLAGS_nStreams];
    for (int i = 0; i < FLAGS_nStreams; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    for (int i = 0; i < FLAGS_num_run; i++)
    {
        Stopwatch sw_run_time(true);
        start(g, streams);
        sw_run_time.stop();
        printf("run %d time: %f ms\n", i, sw_run_time.ms());
    }


    if(FLAGS_check)
    {
        auto regression = SSSP::host_sssp(g, FLAGS_source_node);
        int errors = SSSP::SSSPCheckErrors(g.getherValues(), regression);
        printf("total errors: %d\n", errors);
    }
    else
    {
        printf("Warning: Result not checked\n");
    }
    sw_overall_time.stop();
    std::cout << "Over_All_Time: " << sw_overall_time.ms() << " ms" << std::endl;

    return 0;
}
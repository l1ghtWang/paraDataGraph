#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include <gflags/gflags.h>
#include "graph.cuh"
#include "cudaErrorCheck.cuh"
#include <queue>

#define SSSP_TVALUE uint32_t
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
    __global__ void kernel_SSSP_zeroCopy_sync_push_dd(uint32_t work_size,
                                             uint32_t *dev_node_value_datum,
                                             uint32_t *dev_node_buffer_datum,
                                             uint32_t *dev_worklist,
                                             uint32_t *dev_edgeList,
                                             uint32_t *dev_arr_node_edgeStartIndex_CSR)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = dev_worklist[i];
            if (dev_node_value_datum[node] > dev_node_buffer_datum[node])
            {
                dev_node_value_datum[node] = dev_node_buffer_datum[node];
                // uint32_t edge_loc =
                uint32_t edge_start = dev_arr_node_edgeStartIndex_CSR[node];
                uint32_t edge_end = dev_arr_node_edgeStartIndex_CSR[node + 1];
                for (uint32_t edge = edge_start; edge < edge_end; edge++)
                {
                    uint32_t dst_node = dev_edgeList[edge];
                    uint32_t new_dist = dev_node_buffer_datum[node] + 1;
                    atomicMin(&dev_node_buffer_datum[dst_node], new_dist);
                }
            }
        }
    }

    std::vector<uint32_t> host_sssp(Graph<SSSP_TVALUE> &graph, uint32_t source_node)
    {
        std::vector<uint32_t> distances(graph.get_num_nodes(), UINT32_MAX);
        std::queue<uint32_t> work;

        distances[source_node] = 0;
        work.push(source_node);
        while (!work.empty())
        {
            uint32_t node = work.front();
            work.pop();
            uint32_t edge_start = graph.get_edgeStartIndex(node);
            uint32_t edge_end = graph.get_edgeStartIndex(node + 1);
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

enum class dataTransferType
{
    Explicit_Filter,
    Unified_Memory,
    Zero_Copy,
    Explicit_Compaction
};

// __global__
// void kernel_print_edgeList(uint32_t *dev_edgeList)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid == 0)
//     {
//         for(int i = 0; i < 5; i++)
//         {
//             printf("%u ", dev_edgeList[i]);
//         }
//         printf("\n");
//     }

// }

class Engine_SSSP
{

private:
    cudaStream_t *m_streams;
    Graph<SSSP_TVALUE> *m_graph;
    std::vector<dataTransferType> m_vec_dataTransferType_perPartition;
public:
    Engine_SSSP() : m_graph(nullptr), m_streams(nullptr)
    {
        init_cuda_streams();
    }

    ~Engine_SSSP()
    {
        if(m_streams != nullptr)
        {
            for (int i = 0; i < FLAGS_nStreams; i++)
            {
                    cudaStreamDestroy(m_streams[i]);
            }
            delete[] m_streams;
        }
    }
    void setGraph(Graph<SSSP_TVALUE>* g)
    {
        m_graph = g;
    }

    void init_cuda_streams()
    {
        m_streams = new cudaStream_t[FLAGS_nStreams];
        for (int i = 0; i < FLAGS_nStreams; i++)
        {
            CHECK_CUDA_ERROR(cudaStreamCreate(&m_streams[i]));
        }
    }

    bool initial_Framework()
    {
        SSSP::kernel_InitGraph<<<m_graph->get_num_nodes() / 512 + 1, 512>>>(m_graph->get_num_nodes(),
                                                                     m_graph->get_device_value_ptr(),
                                                                     m_graph->get_device_buffer_ptr(),
                                                                     FLAGS_source_node);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        uint32_t streamIdx = 0;
        for (int i = 0; i < m_graph->get_num_partitions(); i++)
        {
            streamIdx++;
            if (streamIdx == FLAGS_nStreams)
            {
                streamIdx = 1;
            }
            uint32_t startNodeIdx = m_graph->get_partition_start_node(i);
            uint32_t numNodesInPartition = m_graph->get_partition_start_node(i + 1) - m_graph->get_partition_start_node(i);
            SSSP::kernel_RebuildWorklist_perPartition<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(startNodeIdx,
                                                                                                                       numNodesInPartition,
                                                                                                                       m_graph->get_device_value_ptr(),
                                                                                                                       m_graph->get_device_buffer_ptr(),
                                                                                                                       m_graph->get_vector_device_worklist_ptr()[i],
                                                                                                                       m_graph->get_vector_device_worklist_counter_ptr()[i]);
        }

        bool isConverged = true;
        streamIdx = 0;
        for (int i = 0; i < m_graph->get_num_partitions(); i++)
        {
            streamIdx++;
            if (streamIdx == FLAGS_nStreams)
            {
                streamIdx = 1;
            }
            uint32_t numItemInWorkList = m_graph->get_worklist_counter_value(i, m_streams[streamIdx]);
            m_graph->set_partition_numActiveNodes(i, numItemInWorkList);
            isConverged &= (numItemInWorkList == 0);
            // printf("partition: %d numItemInWorkList: %u\n", i, numItemInWorkList);
        }
        CHECK_LAST_CUDA_ERROR();
        // m_vec_dataTransferType_perPartition = std::vector<dataTransferType>(m_graph->get_num_partitions(), dataTransferType::Explicit_Filter);
        // printf("all EF\n");
        m_vec_dataTransferType_perPartition = std::vector<dataTransferType>(m_graph->get_num_partitions(), dataTransferType::Zero_Copy);
        printf("all ZC\n");
        // m_vec_dataTransferType_perPartition = std::vector<dataTransferType>(m_graph->get_num_partitions(), dataTransferType::Unified_Memory);
        // printf("all UM\n");

        return isConverged;
    }

    bool rebuild_workList_check_converge()
    {
        uint32_t streamIdx = 0;
        for (int i = 0; i < m_graph->get_num_partitions(); i++)
        {
            streamIdx++;
            if (streamIdx == FLAGS_nStreams)
            {
                streamIdx = 1;
            }
            uint32_t startNodeIdx = m_graph->get_partition_start_node(i);
            uint32_t numNodesInPartition = m_graph->get_partition_start_node(i + 1) - m_graph->get_partition_start_node(i);
            cudaMemsetAsync(m_graph->get_vector_device_worklist_counter_ptr()[i], 0, sizeof(uint32_t), m_streams[streamIdx]);
            SSSP::kernel_RebuildWorklist_perPartition<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(startNodeIdx,
                                                                                                                     numNodesInPartition,
                                                                                                                     m_graph->get_device_value_ptr(),
                                                                                                                     m_graph->get_device_buffer_ptr(),
                                                                                                                     m_graph->get_vector_device_worklist_ptr()[i],
                                                                                                                     m_graph->get_vector_device_worklist_counter_ptr()[i]);
        }

        bool isConverged = true;
        streamIdx = 0;
        for (int i = 0; i < m_graph->get_num_partitions(); i++)
        {
            streamIdx++;
            if (streamIdx == FLAGS_nStreams)
            {
                streamIdx = 1;
            }

            uint32_t numItemInWorkList = m_graph->get_worklist_counter_value(i, m_streams[streamIdx]);
std::cout<<"numActiveNode: ";
std::vector<uint32_t> vec_numActiveNode;
for(int j = 0; j < numItemInWorkList; j++)
{
    vec_numActiveNode.push_back(m_graph->get_vector_device_worklist_ptr()[i][j]);
}
std::sort(vec_numActiveNode.begin(), vec_numActiveNode.end());
for(int j = 0; j < numItemInWorkList; j++)
{
    std::cout<<vec_numActiveNode[j]<<" ";
}
std::cout<<std::endl;

            m_graph->set_partition_numActiveNodes(i, numItemInWorkList);
            isConverged &= (numItemInWorkList == 0);
        }
        return isConverged;
    }

    void EF_process_partition(uint32_t partitionIdx, uint32_t streamIdx)
    {
        uint32_t startNodeIdx = m_graph->get_partition_start_node(partitionIdx);
        uint32_t numNodesInPartition = m_graph->get_partition_start_node(partitionIdx + 1) - m_graph->get_partition_start_node(partitionIdx);
        uint32_t numEdgesInPartition = m_graph->get_partition_start_edge(partitionIdx + 1) - m_graph->get_partition_start_edge(partitionIdx);
        uint32_t *dev_edgeList_curPartiton = m_graph->get_device_edgeList_ptr(streamIdx);
        cudaMemcpyAsync(dev_edgeList_curPartiton,
                        m_graph->get_host_array_edgeList_ptr() + m_graph->get_partition_start_edge(partitionIdx),
                        numEdgesInPartition * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        m_streams[streamIdx]);

        SSSP::kernel_SSSP_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
                                                                                                        m_graph->get_device_value_ptr(),
                                                                                                        m_graph->get_device_buffer_ptr(),
                                                                                                        m_graph->get_vector_device_worklist_ptr()[partitionIdx],
                                                                                                        dev_edgeList_curPartiton,
                                                                                                        m_graph->get_partition_start_edge(partitionIdx),
                                                                                                        m_graph->get_device_node_edgeStartIndex_CSR_ptr());
    }
    void ZC_process_partition(uint32_t partitionIdx, uint32_t streamIdx)
    {
        uint32_t startNodeIdx = m_graph->get_partition_start_node(partitionIdx);
        uint32_t numNodesInPartition = m_graph->get_partition_start_node(partitionIdx + 1) - m_graph->get_partition_start_node(partitionIdx);
        uint32_t numEdgesInPartition = m_graph->get_partition_start_edge(partitionIdx + 1) - m_graph->get_partition_start_edge(partitionIdx);
        uint32_t *dev_edgeList_zeroCopy = m_graph->get_device_zeroCopy_edgeList_ptr();

        SSSP::kernel_SSSP_zeroCopy_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
                                                                                                        m_graph->get_device_value_ptr(),
                                                                                                        m_graph->get_device_buffer_ptr(),
                                                                                                        m_graph->get_vector_device_worklist_ptr()[partitionIdx],
                                                                                                        dev_edgeList_zeroCopy,
                                                                                                        m_graph->get_device_node_edgeStartIndex_CSR_ptr());
    }

    void UM_process_partition(uint32_t partitionIdx, uint32_t streamIdx)
    {
        uint32_t startNodeIdx = m_graph->get_partition_start_node(partitionIdx);
        uint32_t numNodesInPartition = m_graph->get_partition_start_node(partitionIdx + 1) - m_graph->get_partition_start_node(partitionIdx);
        uint32_t numEdgesInPartition = m_graph->get_partition_start_edge(partitionIdx + 1) - m_graph->get_partition_start_edge(partitionIdx);
        uint32_t *dev_edgeList_curPartiton = m_graph->get_unifiedMem_array_edgeList_ptr()+ m_graph->get_partition_start_edge(partitionIdx);
        SSSP::kernel_SSSP_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
                                                                                                        m_graph->get_device_value_ptr(),
                                                                                                        m_graph->get_device_buffer_ptr(),
                                                                                                        m_graph->get_vector_device_worklist_ptr()[partitionIdx],
                                                                                                        dev_edgeList_curPartiton,
                                                                                                        m_graph->get_partition_start_edge(partitionIdx),
                                                                                                        m_graph->get_device_node_edgeStartIndex_CSR_ptr());


        // uint32_t *dev_edgeList_zeroCopy = m_graph->get_unifiedMem_array_edgeList_ptr();
        // SSSP::kernel_SSSP_zeroCopy_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
        //                                                                                                 m_graph->get_device_value_ptr(),
        //                                                                                                 m_graph->get_device_buffer_ptr(),
        //                                                                                                 m_graph->get_vector_device_worklist_ptr()[partitionIdx],
        //                                                                                                 dev_edgeList_zeroCopy,
        //                                                                                                 m_graph->get_device_node_edgeStartIndex_CSR_ptr());


    }

    void start()
    {
        bool isConverged = false;
        isConverged = initial_Framework();

        uint32_t iterationCount = 0;
        while (!isConverged)
        {
            Stopwatch sw_iteration_time(true);
            uint32_t numPartitionToProcess = 0;
            uint32_t numActiveNodes = 0;
            uint32_t streamIdx = 0;
            for (int i = 0; i < m_graph->get_num_partitions(); i++)
            {
                if (m_graph->get_partition_numActiveNodes(i) == 0)
                {
                    continue;
                }
                else
                {
                    numPartitionToProcess++;
                    numActiveNodes += m_graph->get_partition_numActiveNodes(i);
                }
                streamIdx++;
                if (streamIdx == FLAGS_nStreams)
                {
                    streamIdx = 1;
                }
                if(m_vec_dataTransferType_perPartition[i] == dataTransferType::Explicit_Filter)
                {
                    // printf("iteration: %d partition: %d Explicit_Filter", iterationCount, i);
                    EF_process_partition(i, streamIdx);
                }
                else if(m_vec_dataTransferType_perPartition[i] == dataTransferType::Unified_Memory)
                {
                    UM_process_partition(i, streamIdx);
                }
                else if(m_vec_dataTransferType_perPartition[i] == dataTransferType::Zero_Copy)
                {
                    ZC_process_partition(i, streamIdx);
                }
                else
                {
                    printf("Error: data transfer type not supported\n");
                }
            }

            for (int i = 0; i < FLAGS_nStreams; i++)
            {
                cudaStreamSynchronize(m_streams[i]);
            }
            CHECK_LAST_CUDA_ERROR();

            isConverged = rebuild_workList_check_converge();
            sw_iteration_time.stop();
            printf("iteration: %d iteration_time: %f ms, numPartitionProcessed: %u Cur_numActiveNodes: %u\n", iterationCount, sw_iteration_time.ms(), numPartitionToProcess, numActiveNodes);
            iterationCount++;
            if (iterationCount == 1000)
            {
                break;
            }
        }

        std::cout << "total iteration: " << iterationCount << std::endl;
    }
};



int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    Stopwatch sw_overall_time(true);
    Graph<SSSP_TVALUE> g;
    g.loadGraph(FLAGS_graphInputfile);
    printf("source node: %u\n", FLAGS_source_node);

    Engine_SSSP engine;
    engine.setGraph(&g);

    for (int i = 0; i < FLAGS_num_run; i++)
    {
        Stopwatch sw_run_time(true);
        engine.start();
        sw_run_time.stop();
        printf("run_id %d Exec_time: %f ms\n", i, sw_run_time.ms());
        g.reset_UM_edgeList();
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
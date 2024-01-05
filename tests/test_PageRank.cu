#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include <gflags/gflags.h>
#include "graph.cuh"
#include "cudaErrorCheck.cuh"
#include <queue>

#define TVALUE float
#define ERROR_THRESHOLD 0.2
#define PR_ERROR 0.01f
#define PR_ALPHA 0.85f

DECLARE_string(graphInputfile);
// DEFINE_uint32(partition_size_MB, 32, "partition size in MB");
DECLARE_uint32(partition_size_MB);
DEFINE_uint32(source_node, 0, "source node");
DECLARE_uint32(nStreams);
DEFINE_bool(check, false, "check result");
DEFINE_uint32(num_run, 10, "number of run");
DEFINE_double(pr_error, 0.01, "error threshold of PageRank");
DEFINE_double(pr_alpha, 0.85, "alpha of PageRank");
DEFINE_bool(pr_norm, false, "Normalize PR output ranks (L1)");


namespace PageRank
{
    __forceinline__ __device__ bool kernel_isActiveNode(uint32_t node, TVALUE buffer, TVALUE value)
    {
        return buffer > PR_ERROR;
    }

    __global__ 
    void kernel_InitGraph(uint32_t work_size,
                          TVALUE *dev_node_value_datum,
                          TVALUE *dev_node_buffer_datum,
                          uint32_t source_node)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = i;
            dev_node_buffer_datum[node] = 1-PR_ALPHA;
            dev_node_value_datum[node] = 0.0f;
        }
    }

    __global__ 
    void kernel_RebuildWorklist(uint32_t work_size,
                                TVALUE *dev_node_value_datum,
                                TVALUE *dev_node_buffer_datum,
                                uint32_t *dev_worklist,
                                uint32_t *dev_worklist_counter)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = i;
            if(PageRank::kernel_isActiveNode(node, dev_node_buffer_datum[node], dev_node_value_datum[node]))
            {
                uint32_t location = atomicAdd(dev_worklist_counter, 1);
                dev_worklist[location] = node;
            }
        }
    }

    __global__ 
    void kernel_RebuildWorklist_perPartition(uint32_t startNodeIdx,
                                            uint32_t work_size,
                                            TVALUE *dev_node_value_datum,
                                            TVALUE *dev_node_buffer_datum,
                                            uint32_t *dev_worklist, 
                                            uint32_t *dev_worklist_counter)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = i + startNodeIdx;
            if (PageRank::kernel_isActiveNode(node, dev_node_buffer_datum[node], dev_node_value_datum[node]))
            {
                uint32_t location = atomicAdd(dev_worklist_counter, 1);
                dev_worklist[location] = node;
            }
        }
    }
    __global__ void kernel_PageRank_sync_push_dd(uint32_t work_size,
                                             TVALUE *dev_node_value_datum,
                                             TVALUE *dev_node_buffer_datum,
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
            TVALUE curDelta = atomicExch(&dev_node_buffer_datum[node], 0.0f);
            if(curDelta > PR_ERROR)
            {
                uint32_t edge_start = dev_arr_node_edgeStartIndex_CSR[node] - partition_start_edge_offset;
                uint32_t edge_end = dev_arr_node_edgeStartIndex_CSR[node + 1] - partition_start_edge_offset;
                TVALUE push_delta = curDelta * PR_ALPHA / (edge_end - edge_start);
                for (uint32_t edge = edge_start; edge < edge_end; edge++)
                {
                    uint32_t dst_node = dev_edgeList[edge];
                    atomicAdd(&dev_node_buffer_datum[dst_node], push_delta);
                }
            }


        }
    }
    __global__ void kernel_PageRank_zeroCopy_sync_push_dd(uint32_t work_size,
                                             TVALUE *dev_node_value_datum,
                                             TVALUE *dev_node_buffer_datum,
                                             uint32_t *dev_worklist,
                                             uint32_t *dev_edgeList,
                                             uint32_t *dev_arr_node_edgeStartIndex_CSR)
    {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t nthreads = blockDim.x * gridDim.x;
        for (uint32_t i = tid; i < work_size; i += nthreads)
        {
            uint32_t node = dev_worklist[i];
            TVALUE curDelta = atomicExch(&dev_node_buffer_datum[node], 0.0f);
            if(curDelta > PR_ERROR)
            {
                uint32_t edge_start = dev_arr_node_edgeStartIndex_CSR[node] ;
                uint32_t edge_end = dev_arr_node_edgeStartIndex_CSR[node + 1] ;
                TVALUE push_delta = curDelta * PR_ALPHA / (edge_end - edge_start);
                for (uint32_t edge = edge_start; edge < edge_end; edge++)
                {
                    uint32_t dst_node = dev_edgeList[edge];
                    atomicAdd(&dev_node_buffer_datum[dst_node], push_delta);
                }
            }
        }
    }

    std::vector<TVALUE> host_pr(Graph<TVALUE> &graph)
    {
        std::vector<TVALUE> residual(graph.get_num_nodes(), 0.0);
        std::vector<TVALUE> ranks(graph.get_num_nodes(), 1-PR_ALPHA);

        for (uint32_t node = 0; node < graph.get_num_nodes(); ++node)
        {
            uint32_t edge_start = graph.get_edgeStartIndex(node);
            uint32_t edge_end = graph.get_edgeStartIndex(node + 1);
            uint32_t out_degree = edge_end - edge_start;

            if (out_degree == 0)
                continue;

            TVALUE update = ((1.0 - PR_ALPHA) * PR_ALPHA) / out_degree;

            for (uint32_t edge = edge_start; edge < edge_end; ++edge)
            {
                uint32_t dest = graph.get_host_array_edgeList_ptr()[edge];
                residual[dest] += update;
            }
        }
        std::queue<uint32_t> wl1, wl2;
        std::queue<uint32_t> *in_wl = &wl1, *out_wl = &wl2;
        for (uint32_t node = 0; node < graph.get_num_nodes(); ++node)
        {
            in_wl->push(node);
        }
        int iteration = 0;
        while (!in_wl->empty())
        {
            while (!in_wl->empty())
            {
                uint32_t node = in_wl->front();
                in_wl->pop();

                TVALUE res = residual[node];
                ranks[node] += res;
                residual[node] = 0;

                uint32_t edge_start = graph.get_edgeStartIndex(node);
                uint32_t edge_end = graph.get_edgeStartIndex(node + 1);
                uint32_t out_degree = edge_end - edge_start;

                if (out_degree == 0)
                    continue;

                TVALUE update = res * PR_ALPHA / out_degree;

                for (uint32_t edge = edge_start; edge < edge_end; ++edge)
                {
                    uint32_t dest = graph.get_host_array_edgeList_ptr()[edge];
                    TVALUE prev = residual[dest];
                    residual[dest] += update;

                    if (prev + update > PR_ERROR && prev < PR_ERROR)
                    {
                        out_wl->push(dest);
                    }
                }
            }

            ++iteration;
            std::swap(in_wl, out_wl);
        }

        return ranks;

    }

    int PageRank_CheckErrors(const std::vector<TVALUE> &ranks, const std::vector<TVALUE> &regression)
    {
        std::cout<<"GPU result: "<<std::endl;
        for(int i = 0; i < 30; i++)
        {
            std::cout<<i<<":"<<ranks[i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"CPU result: "<<std::endl;
        for(int i = 0; i < 30; i++)
        {
            std::cout<<i<<":"<<regression[i]<<" ";
        }
        std::cout<<std::endl;
        
        if (ranks.size() != regression.size())
        {
            return std::abs((int64_t)ranks.size() - (int64_t)regression.size());
        }

        // if (FLAGS_pr_norm) // L1 normalization
        // {
        //     TVALUE ranks_sum = 0.0, regression_sum = 0.0;
        //     for (auto val : ranks) ranks_sum += val;
        //     for (auto val : regression) regression_sum += val;
        //     for (auto &val : ranks) val /= ranks_sum;
        //     for (auto &val : regression) val /= regression_sum;
        // }

        int num_diffs = 0;

        for (int node = 0; node < ranks.size(); node++)
        {

            bool is_right = true;
            if (fabs(ranks[node]) < 0.01f && fabs(regression[node] - 1) < 0.01f)
                continue;
            if (fabs(ranks[node] - 0.0) < 0.01f)
            {
                if (fabs(ranks[node] - regression[node]) > ERROR_THRESHOLD)
                    is_right = false;
            }
            else
            {
                if (fabs((ranks[node] - regression[node]) / regression[node]) > ERROR_THRESHOLD)
                    is_right = false;
            }

            if (!is_right)
                num_diffs++;
        }
        return num_diffs;
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

class Engine_PageRank
{

private:
    cudaStream_t *m_streams;
    Graph<TVALUE> *m_graph;
    std::vector<dataTransferType> m_vec_dataTransferType_perPartition;
public:
    Engine_PageRank() : m_graph(nullptr), m_streams(nullptr)
    {
        init_cuda_streams();
    }

    ~Engine_PageRank()
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
    void setGraph(Graph<TVALUE>* g)
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
        PageRank::kernel_InitGraph<<<m_graph->get_num_nodes() / 512 + 1, 512>>>(m_graph->get_num_nodes(),
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
            PageRank::kernel_RebuildWorklist_perPartition<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(startNodeIdx,
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
        m_vec_dataTransferType_perPartition = std::vector<dataTransferType>(m_graph->get_num_partitions(), dataTransferType::Explicit_Filter);
        printf("all EF\n");
        // m_vec_dataTransferType_perPartition = std::vector<dataTransferType>(m_graph->get_num_partitions(), dataTransferType::Zero_Copy);
        // printf("all ZC\n");
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
            PageRank::kernel_RebuildWorklist_perPartition<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(startNodeIdx,
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

        PageRank::kernel_PageRank_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
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

        PageRank::kernel_PageRank_zeroCopy_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
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
        PageRank::kernel_PageRank_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
                                                                                                        m_graph->get_device_value_ptr(),
                                                                                                        m_graph->get_device_buffer_ptr(),
                                                                                                        m_graph->get_vector_device_worklist_ptr()[partitionIdx],
                                                                                                        dev_edgeList_curPartiton,
                                                                                                        m_graph->get_partition_start_edge(partitionIdx),
                                                                                                        m_graph->get_device_node_edgeStartIndex_CSR_ptr());


        // uint32_t *dev_edgeList_zeroCopy = m_graph->get_unifiedMem_array_edgeList_ptr();
        // PageRank::kernel_SSSP_zeroCopy_sync_push_dd<<<numNodesInPartition / 512 + 1, 512, 0, m_streams[streamIdx]>>>(numNodesInPartition,
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
            printf("iteration: %d time: %f ms, numPartitionProcessed: %u numActiveNodes: %u\n", iterationCount, sw_iteration_time.ms(), numPartitionToProcess, numActiveNodes);
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
    Graph<TVALUE> g;
    g.loadGraph(FLAGS_graphInputfile);
    // printf("source node: %u\n", FLAGS_source_node);

    Engine_PageRank engine;
    engine.setGraph(&g);

    for (int i = 0; i < FLAGS_num_run; i++)
    {
        Stopwatch sw_run_time(true);
        engine.start();
        sw_run_time.stop();
        printf("run %d time: %f ms\n", i, sw_run_time.ms());
    }


    if(FLAGS_check)
    {   
        Stopwatch sw_check_time(true);
        auto regression = PageRank::host_pr(g);
        int errors = PageRank::PageRank_CheckErrors(g.getherValues(), regression);
        printf("total errors: %d\n", errors);
        sw_check_time.stop();
        std::cout << "Check_Time: " << sw_check_time.ms() << " ms" << std::endl;
    }
    else
    {
        printf("Warning: Result not checked\n");
    }
    sw_overall_time.stop();
    std::cout << "Over_All_Time: " << sw_overall_time.ms() << " ms" << std::endl;

    return 0;
}
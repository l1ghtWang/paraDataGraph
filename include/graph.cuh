// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "cudaErrorCheck.cuh"
#include "gflags/gflags.h"
#include <assert.h>
#include <map>
#include <algorithm>

DEFINE_uint32(partition_size_MB, 32, "partition size in MB");
DEFINE_string(graphInputfile, "/home/share/graph_data/raw/datasets/Google/web-Google.el", "input graph file");
DEFINE_uint32(nStreams, 32, "number of cuda streams");

struct Edge
{
    uint32_t source;
    uint32_t end;
};

// template <typename T>
// struct DEV_GraphDatum
// {
//     T *data_ptr;
//     uint32_t size;

//     DEV_GraphDatum() : data_ptr(nullptr), size(0)
//     {
//     }

//     DEV_GraphDatum(T *data_ptr, uint32_t size) : data_ptr(data_ptr), size(size)
//     {
//     }

//     __device__ __forceinline__ T get_item(uint32_t idx) const
//     {
//         assert(idx < size);
//         return data_ptr[idx];
//     }

//     __device__ __forceinline__ T &operator[](uint32_t idx)
//     {
//         return data_ptr[idx];
//     }

//     __device__ __forceinline__ T *get_item_ptr(uint32_t idx) const
//     {
//         assert(idx < size);
//         return data_ptr + (idx);
//     }

//     __device__ __forceinline__ void set_item(uint32_t idx, const T &item) const
//     {
//         assert(idx < size);
//         data_ptr[idx] = item;
//     }
// };
            
template <typename TValue>
class Graph
{
private:
    uint32_t m_num_nodes;
    uint32_t m_num_edges;
    bool m_isWeighted;
    uint32_t *m_host_arr_node_edgeStartIndex_CSR;
    uint32_t *m_host_arr_edgeList;
    uint32_t *m_dev_zeroCopy_arr_edgeList;
    uint32_t *m_unifiedMem_arr_edgeList;
    uint8_t *m_managed_arr_activeNodesLabeling;
    uint32_t *m_managed_arr_prefixLabeling;
    uint32_t *m_managed_activeNodesDegree;
    uint32_t *m_managed_prefixSumDegree;
    uint32_t *m_managed_arr_subGraph_edgeStartIndex_CSR;

    std::vector<TValue> m_vec_host_node_value_datum;
    std::vector<uint32_t> m_vec_partition_start_node;
    std::vector<uint32_t> m_vec_partition_start_edge;
    std::vector<uint32_t *> m_vec_dev_edgeList_perStream;
    uint32_t *m_dev_arr_node_edgeStartIndex_CSR;
    TValue *m_dev_node_value_datum;
    TValue *m_dev_node_buffer_datum;

    uint32_t m_num_partitions;

    std::vector<uint32_t *> m_vec_managed_worklist_perPartition;
    std::vector<uint32_t *> m_vec_dev_worklist_counter_perPartition;
    std::vector<uint32_t *> m_vec_dev_numActiveEdge_counter_perPartition;
    std::vector<uint32_t> m_vec_partition_numActiveNodes;


public:
    Graph() : m_num_nodes(0), m_num_edges(0), m_isWeighted(false),
              m_host_arr_node_edgeStartIndex_CSR(nullptr), m_host_arr_edgeList(nullptr), m_dev_zeroCopy_arr_edgeList(nullptr),m_unifiedMem_arr_edgeList(nullptr),
               m_num_partitions(0),m_managed_arr_activeNodesLabeling(nullptr),m_managed_activeNodesDegree(nullptr),m_managed_arr_prefixLabeling(nullptr),
               m_managed_prefixSumDegree(nullptr),m_managed_arr_subGraph_edgeStartIndex_CSR(nullptr),
              m_dev_arr_node_edgeStartIndex_CSR(nullptr),
              m_dev_node_buffer_datum(nullptr), m_dev_node_value_datum(nullptr)
              {};
    ~Graph(){
        if(m_host_arr_node_edgeStartIndex_CSR != nullptr)
            delete[] m_host_arr_node_edgeStartIndex_CSR;
        if(m_host_arr_edgeList != nullptr)
            CHECK_CUDA_ERROR(cudaFreeHost(m_host_arr_edgeList));
        if(m_dev_arr_node_edgeStartIndex_CSR != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_dev_arr_node_edgeStartIndex_CSR));
        if(m_dev_node_value_datum != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_dev_node_value_datum));
        if(m_dev_node_buffer_datum != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_dev_node_buffer_datum));
        if(m_managed_arr_activeNodesLabeling != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_managed_arr_activeNodesLabeling));
        if(m_managed_arr_prefixLabeling != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_managed_arr_prefixLabeling));
        if(m_managed_activeNodesDegree != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_managed_activeNodesDegree));
        if(m_managed_arr_subGraph_edgeStartIndex_CSR != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_managed_arr_subGraph_edgeStartIndex_CSR));
        if(m_managed_prefixSumDegree != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_managed_prefixSumDegree));
        for(int i = 0; i < m_vec_managed_worklist_perPartition.size(); i++)
        {
            if(m_vec_managed_worklist_perPartition[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_managed_worklist_perPartition[i]));
        }
        for(int i = 0; i < m_vec_dev_worklist_counter_perPartition.size(); i++)
        {
            if(m_vec_dev_worklist_counter_perPartition[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_dev_worklist_counter_perPartition[i]));
        }
        for(int i = 0; i < m_vec_dev_numActiveEdge_counter_perPartition.size(); i++)
        {
            if(m_vec_dev_numActiveEdge_counter_perPartition[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_dev_numActiveEdge_counter_perPartition[i]));
        }
        for(int i = 0; i < m_vec_dev_edgeList_perStream.size(); i++)
        {
            if(m_vec_dev_edgeList_perStream[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_dev_edgeList_perStream[i]));
        }

        if(m_unifiedMem_arr_edgeList != nullptr)
            CHECK_CUDA_ERROR(cudaFree(m_unifiedMem_arr_edgeList));
    };

    std::vector<TValue> getherValues()
    {

        TValue* host_node_value_datum = new TValue[m_num_nodes];
        CHECK_CUDA_ERROR(cudaMemcpy(host_node_value_datum, m_dev_node_value_datum, m_num_nodes * sizeof(TValue), cudaMemcpyDeviceToHost));
        m_vec_host_node_value_datum.resize(m_num_nodes);
        for(int i = 0; i < m_num_nodes; i++)
        {
            m_vec_host_node_value_datum[i] = host_node_value_datum[i];
        }
        delete[] host_node_value_datum;
        return m_vec_host_node_value_datum;

    }
    uint32_t get_num_nodes()
    {
        return m_num_nodes;
    }
    uint32_t get_num_edges()
    {
        return m_num_edges;
    }

    uint32_t get_num_partitions()
    {
        return m_num_partitions;
    }
    uint8_t* get_managed_activeNodesLabeling_ptr()
    {
        return m_managed_arr_activeNodesLabeling;
    }
    uint32_t* get_managed_prefixLabeling_ptr()
    {
        return m_managed_arr_prefixLabeling;
    }
    uint32_t* get_managed_activeNodesDegree_ptr()
    {
        return m_managed_activeNodesDegree;
    }
    uint32_t* get_managed_prefixSumDegree_ptr()
    {
        return m_managed_prefixSumDegree;
    }
    uint32_t* get_managed_subGraph_edgeStartIndex_CSR_ptr()
    {
        return m_managed_arr_subGraph_edgeStartIndex_CSR;
    }
    uint32_t get_partition_numActiveNodes(uint32_t partition_id)
    {
        return m_vec_partition_numActiveNodes[partition_id];
    }
    uint32_t get_partition_numActiveEdges(uint32_t partition_id)
    {
        return 0;
    }

    void set_partition_numActiveNodes(uint32_t partition_id, uint32_t numActiveNodes)
    {
        m_vec_partition_numActiveNodes[partition_id] = numActiveNodes;
    }

    uint32_t get_edgeStartIndex(uint32_t node_id)
    {
        return m_host_arr_node_edgeStartIndex_CSR[node_id];
    }
    uint32_t* get_host_array_node_edgeStartIndex_CSR_ptr()
    {
        return m_host_arr_node_edgeStartIndex_CSR;
    }

    uint32_t* get_device_node_edgeStartIndex_CSR_ptr()
    {
        return m_dev_arr_node_edgeStartIndex_CSR;
    }

    uint32_t* get_host_array_edgeList_ptr()
    {
        return m_host_arr_edgeList;
    }

    uint32_t* get_unifiedMem_array_edgeList_ptr()
    {
        return m_unifiedMem_arr_edgeList;
    }
    uint32_t* get_device_zeroCopy_edgeList_ptr()
    {
        return m_dev_zeroCopy_arr_edgeList;
    }
    uint32_t get_partition_start_node(uint32_t partition_id)
    {
        return m_vec_partition_start_node[partition_id];
    }


    uint32_t get_partition_start_edge(uint32_t partition_id)
    {
        return m_vec_partition_start_edge[partition_id];
    }

    TValue * get_device_value_ptr()
    {
        return m_dev_node_value_datum;
    }
    TValue *get_device_buffer_ptr()
    {
        return m_dev_node_buffer_datum;
    }

    uint32_t *get_device_edgeList_ptr(uint32_t streamIdx)
    {
        return m_vec_dev_edgeList_perStream[streamIdx];
    }
    // uint32_t *get_device_worklist_ptr()
    // {
    //     return m_dev_worklist;
    // }

    // uint32_t *get_device_worklist_counter_ptr()
    // {
    //     return m_dev_worklist_counter;
    // }

    std::vector<uint32_t *> get_vector_device_worklist_ptr()
    {
        return m_vec_managed_worklist_perPartition;
    }

    std::vector<uint32_t *> get_vector_device_worklist_counter_ptr()
    {
        return m_vec_dev_worklist_counter_perPartition;
    }
    std::vector<uint32_t *> get_vector_device_numActiveEdge_counter_ptr()
    {
        return m_vec_dev_numActiveEdge_counter_perPartition;
    }

    uint32_t get_worklist_counter_value(uint32_t partition_id, cudaStream_t streamIdx)
    {
        uint32_t worklist_counter;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&worklist_counter, m_vec_dev_worklist_counter_perPartition[partition_id], sizeof(uint32_t), cudaMemcpyDeviceToHost, streamIdx));
        return worklist_counter;
    }
    uint32_t get_numActiveEdge_counter_value(uint32_t partition_id, cudaStream_t streamIdx)
    {
        uint32_t numActiveEdge_counter;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&numActiveEdge_counter, m_vec_dev_numActiveEdge_counter_perPartition[partition_id], sizeof(uint32_t), cudaMemcpyDeviceToHost, streamIdx));
        return numActiveEdge_counter;
    }

    std::string get_File_Extension(std::string fileName)
    {
        if (fileName.find_last_of(".") != std::string::npos)
            return fileName.substr(fileName.find_last_of(".") + 1);
        return "";
    }

    void print_partitioned_graph_info()
    {
        for (uint32_t i = 0; i < m_vec_partition_start_node.size() - 1; i++)
        {
            std::cout << "Partition " << i << ": ";
            std::cout << "start node: " << m_vec_partition_start_node[i] << " end node: " << m_vec_partition_start_node[i + 1] - 1 << "";
            std::cout << " node_size: " << m_vec_partition_start_node[i + 1] - m_vec_partition_start_node[i];
            std::cout << " start edge: " << m_vec_partition_start_edge[i] << " end edge: " << m_vec_partition_start_edge[i + 1] - 1 << "";
            std::cout << " edgeList_size: " << m_vec_partition_start_edge[i + 1] - m_vec_partition_start_edge[i];
            std::cout << std::endl;
        }
    }

    void loadGraph(std::string filename)
    {

        Stopwatch sw(true);
        std::cout << "Loading graph from " << filename << std::endl;
        std::string fileExtension = get_File_Extension(filename);
        if (fileExtension == "el")
        {
            m_isWeighted = false;
            std::ifstream infile;
            infile.open(filename);
            std::stringstream ss;
            std::string line;
            uint32_t edgeCounter = 0;

            std::map<uint32_t, std::vector<uint32_t>> edges_map;
            // Edge newEdge;
            uint32_t max = 0;
            while (getline(infile, line))
            {
                uint32_t source, end;
                ss.str("");
                ss.clear();
                ss << line;

                ss >> source;
                ss >> end;

                // edges.push_back(newEdge);
                edges_map[source].push_back(end);
                edgeCounter++;

                if (max < source)
                    max = source;
                if (max < end)
                    max = end;
            }
            infile.close();

            m_num_nodes = max + 1;
            m_num_edges = edgeCounter;

            CHECK_CUDA_ERROR(cudaMallocHost(&m_host_arr_edgeList, (m_num_edges) * sizeof(uint32_t)));
            m_host_arr_node_edgeStartIndex_CSR = new uint32_t[m_num_nodes + 1];

            uint32_t *degree = new uint32_t[m_num_nodes];  
            uint32_t counter = 0;
            for (uint32_t i = 0; i < m_num_nodes; i++)
            {
                degree[i] = edges_map[i].size();
                std::sort(edges_map[i].begin(), edges_map[i].end());
                for(int j = 0; j < edges_map[i].size(); j++)
                {
                    m_host_arr_edgeList[counter + j] = edges_map[i][j];
                }
                counter = counter + edges_map[i].size();
            }

            counter = 0;
            for (uint32_t i = 0; i < m_num_nodes; i++)
            {
                m_host_arr_node_edgeStartIndex_CSR[i] = counter;
                counter = counter + degree[i];
            }

            m_host_arr_node_edgeStartIndex_CSR[m_num_nodes] = m_num_edges;

            edges_map.clear();
            delete[] degree;

        }
        else if (fileExtension == "bcsr")
        {
            std::ifstream infile (filename, std::ios::in | std::ios::binary);
        
            infile.read ((char*)&m_num_nodes, sizeof(uint32_t));
            infile.read ((char*)&m_num_edges, sizeof(uint32_t));
            
            m_host_arr_node_edgeStartIndex_CSR = new uint32_t[m_num_nodes + 1];
            CHECK_CUDA_ERROR(cudaMallocHost(&m_host_arr_edgeList, (m_num_edges) * sizeof(uint32_t)));
            
            infile.read ((char*)m_host_arr_node_edgeStartIndex_CSR, sizeof(uint32_t)*m_num_nodes);
            infile.read ((char*)m_host_arr_edgeList, (m_num_edges) * sizeof(uint32_t));
            m_host_arr_node_edgeStartIndex_CSR[m_num_nodes] = m_num_edges;
            infile.close();
        }
        else if (fileExtension == "wel")
            m_isWeighted = true;
        else
        {
            std::cout << "File extension not supported!" << std::endl;
            exit(1);
        }

        
        if (m_isWeighted)
        {
        }
        else
        {


            
            CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_dev_zeroCopy_arr_edgeList, m_host_arr_edgeList, 0));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_unifiedMem_arr_edgeList, (m_num_edges) * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_arr_activeNodesLabeling, (m_num_nodes)*sizeof(uint8_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_arr_prefixLabeling, (m_num_nodes)*sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_activeNodesDegree, (m_num_nodes)*sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_prefixSumDegree, (m_num_nodes)*sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_arr_subGraph_edgeStartIndex_CSR, (m_num_nodes + 1)*sizeof(uint32_t)));

            memcpy(m_unifiedMem_arr_edgeList, m_host_arr_edgeList, m_num_edges*sizeof(uint32_t));

            // for(uint32_t i = 0; i < m_num_edges; i++)
            // {
            //     m_unifiedMem_arr_edgeList[i] = m_host_arr_edgeList[i];
            // }


            printf("FLAGS_partition_size_MB: %d\n", FLAGS_partition_size_MB);
            uint32_t m_num_edgesInOnePartition = (uint64_t)FLAGS_partition_size_MB * (uint64_t)1024 * (uint64_t)1024 / (uint64_t)sizeof(uint32_t);
            std::cout<<"m_num_edgesInOnePartition: "<<m_num_edgesInOnePartition<<std::endl;
            uint32_t counter_edgesInOnePartition = 0;

            m_vec_partition_start_node.clear();
            m_vec_partition_start_edge.clear();
            m_vec_partition_start_node.push_back(0);
            m_vec_partition_start_edge.push_back(0);
            m_num_partitions = 0;
            uint32_t maximum_edgeListPerPartition = 0;
            uint32_t max_degree = 0;
            for (uint32_t i = 0; i < m_num_nodes; i++)
            {
                uint32_t current_degree = m_host_arr_node_edgeStartIndex_CSR[i + 1] - m_host_arr_node_edgeStartIndex_CSR[i];
                if (max_degree < current_degree)
                    max_degree = current_degree;


                counter_edgesInOnePartition = counter_edgesInOnePartition + current_degree;
                if (counter_edgesInOnePartition >= m_num_edgesInOnePartition)
                {
                    m_num_partitions++;
                    m_vec_partition_start_node.push_back(i + 1);
                    m_vec_partition_start_edge.push_back(m_host_arr_node_edgeStartIndex_CSR[i + 1]);
                    if(maximum_edgeListPerPartition < counter_edgesInOnePartition)
                        maximum_edgeListPerPartition = counter_edgesInOnePartition;
                    counter_edgesInOnePartition = 0;
                }
            }
            if (counter_edgesInOnePartition > 0)
            {
                m_num_partitions++;
                m_vec_partition_start_node.push_back(m_num_nodes);
                m_vec_partition_start_edge.push_back(m_num_edges);
                if(maximum_edgeListPerPartition < counter_edgesInOnePartition)
                    maximum_edgeListPerPartition = counter_edgesInOnePartition;
            }
            
            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_arr_node_edgeStartIndex_CSR, (m_num_nodes + 1) * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMemcpy(m_dev_arr_node_edgeStartIndex_CSR, m_host_arr_node_edgeStartIndex_CSR, (m_num_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
            m_vec_partition_numActiveNodes.clear();
            // initial m_vec_partition_numActiveNodes to zeros
            m_vec_partition_numActiveNodes.resize(m_num_partitions);


            std::cout<<"The graph has "<<m_num_nodes<<" vertices, and "<<m_num_edges<<" edges (average_degree: "<<(float)m_num_edges / m_num_nodes<<", max_degree: "<<max_degree<<")"<<std::endl;
            print_partitioned_graph_info();
            std::cout<<"maximum_edgeListPerPartition: "<<maximum_edgeListPerPartition<<std::endl;

            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_node_value_datum, m_num_nodes * sizeof(TValue)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_node_buffer_datum, m_num_nodes * sizeof(TValue)));
            // CHECK_CUDA_ERROR(cudaMalloc(&m_dev_worklist, m_num_nodes * sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMalloc(&m_dev_worklist_counter, sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMemset(m_dev_worklist_counter, 0, sizeof(uint32_t)));
            printf("m_num_partitions: %d\n", m_num_partitions);
            for(int i = 0; i < m_num_partitions; i++)
            {
                uint32_t *dev_worklist;
                uint32_t node_partionSize = m_vec_partition_start_node[i + 1] - m_vec_partition_start_node[i];
                // printf("partition %d size: %d\n", i, node_partionSize);
                CHECK_CUDA_ERROR(cudaMallocManaged(&dev_worklist, node_partionSize * sizeof(uint32_t)));
                m_vec_managed_worklist_perPartition.push_back(dev_worklist);
                uint32_t *dev_worklist_counter;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_worklist_counter, sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemset(dev_worklist_counter, 0, sizeof(uint32_t)));
                m_vec_dev_worklist_counter_perPartition.push_back(dev_worklist_counter);

                uint32_t *dev_numActiveEdge_counter;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_numActiveEdge_counter, sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemset(dev_numActiveEdge_counter, 0, sizeof(uint32_t)));
                m_vec_dev_numActiveEdge_counter_perPartition.push_back(dev_numActiveEdge_counter);

            }
            printf("m_vec_managed_worklist_perPartition size: %d\n", m_vec_managed_worklist_perPartition.size());
            printf("m_vec_dev_worklist_counter_perPartition size: %d\n", m_vec_dev_worklist_counter_perPartition.size());


            if(FLAGS_nStreams == 1)
                FLAGS_nStreams = 2; //we dont want to assign task to stream 0
            for (int i = 1; i < FLAGS_nStreams; i++)
            {
                uint32_t *dev_edgeList;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_edgeList, (maximum_edgeListPerPartition) * sizeof(uint32_t)));
                m_vec_dev_edgeList_perStream.push_back(dev_edgeList);

            }
            // edges.clear();
            
            // delete[] outDegreeCounter;
        }
        sw.stop();
        std::cout << "Loading graph Time: " << sw.ms() << " ms" << std::endl;
    };
    void loadGraph0(std::string filename)
    {

        Stopwatch sw(true);
        std::cout << "Loading graph from " << filename << std::endl;
        std::string fileExtension = get_File_Extension(filename);
        if (fileExtension == "el")
            m_isWeighted = false;
        else if (fileExtension == "wel")
            m_isWeighted = true;
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
        if (m_isWeighted)
        {
        }
        else
        {
            // std::vector<Edge> edges;
            std::map<uint32_t, std::vector<uint32_t>> edges_map;
            // Edge newEdge;
            uint32_t max = 0;
            while (getline(infile, line))
            {
                uint32_t source, end;
                ss.str("");
                ss.clear();
                ss << line;

                ss >> source;
                ss >> end;

                // edges.push_back(newEdge);
                edges_map[source].push_back(end);
                edgeCounter++;

                if (max < source)
                    max = source;
                if (max < end)
                    max = end;
            }
            infile.close();
            m_num_nodes = max + 1;
            m_num_edges = edgeCounter;
            m_host_arr_node_edgeStartIndex_CSR = new uint32_t[m_num_nodes + 1];
            CHECK_CUDA_ERROR(cudaMallocHost(&m_host_arr_edgeList, (m_num_edges) * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_dev_zeroCopy_arr_edgeList, m_host_arr_edgeList, 0));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_unifiedMem_arr_edgeList, (m_num_edges) * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_arr_activeNodesLabeling, (m_num_nodes)*sizeof(uint8_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_arr_prefixLabeling, (m_num_nodes)*sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_activeNodesDegree, (m_num_nodes)*sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_prefixSumDegree, (m_num_nodes)*sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&m_managed_arr_subGraph_edgeStartIndex_CSR, (m_num_nodes + 1)*sizeof(uint32_t)));

            uint32_t *degree = new uint32_t[m_num_nodes];
            uint32_t counter = 0;
            for (uint32_t i = 0; i < m_num_nodes; i++)
            {
                degree[i] = edges_map[i].size();
                std::sort(edges_map[i].begin(), edges_map[i].end());
                for(int j = 0; j < edges_map[i].size(); j++)
                {
                    m_host_arr_edgeList[counter + j] = edges_map[i][j];
                    m_unifiedMem_arr_edgeList[counter + j] = edges_map[i][j];
                    // if(i == 0)
                    // {
                    //     printf("debug_info: node %d , edge %d , to: %d\n", i, j, edges_map[i][j]);
                    
                    // }
                }
                counter = counter + edges_map[i].size();
            }
            // for (uint32_t i = 0; i < m_num_edges; i++)
            // {
            //     degree[edges[i].source]++;
            // }

            counter = 0;
            uint32_t max_degree = 0;

            printf("FLAGS_partition_size_MB: %d\n", FLAGS_partition_size_MB);
            uint32_t m_num_edgesInOnePartition = FLAGS_partition_size_MB * 1024 * 1024 / sizeof(uint32_t);
            uint32_t counter_edgesInOnePartition = 0;

            m_vec_partition_start_node.clear();
            m_vec_partition_start_edge.clear();
            m_vec_partition_start_node.push_back(0);
            m_vec_partition_start_edge.push_back(0);
            m_num_partitions = 0;
            uint32_t maximum_edgeListPerPartition = 0;
            for (uint32_t i = 0; i < m_num_nodes; i++)
            {
                if (max_degree < degree[i])
                    max_degree = degree[i];
                m_host_arr_node_edgeStartIndex_CSR[i] = counter;
                counter = counter + degree[i];
                counter_edgesInOnePartition = counter_edgesInOnePartition + degree[i];
                if (counter_edgesInOnePartition >= m_num_edgesInOnePartition)
                {
                    m_num_partitions++;
                    m_vec_partition_start_node.push_back(i + 1);
                    m_vec_partition_start_edge.push_back(counter);
                    if(maximum_edgeListPerPartition < counter_edgesInOnePartition)
                        maximum_edgeListPerPartition = counter_edgesInOnePartition;
                    counter_edgesInOnePartition = 0;
                }
            }
            if (counter_edgesInOnePartition > 0)
            {
                m_num_partitions++;
                m_vec_partition_start_node.push_back(m_num_nodes);
                m_vec_partition_start_edge.push_back(m_num_edges);
                if(maximum_edgeListPerPartition < counter_edgesInOnePartition)
                    maximum_edgeListPerPartition = counter_edgesInOnePartition;
            }
            m_host_arr_node_edgeStartIndex_CSR[m_num_nodes] = m_num_edges;
            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_arr_node_edgeStartIndex_CSR, (m_num_nodes + 1) * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMemcpy(m_dev_arr_node_edgeStartIndex_CSR, m_host_arr_node_edgeStartIndex_CSR, (m_num_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
            m_vec_partition_numActiveNodes.clear();
            // initial m_vec_partition_numActiveNodes to zeros
            m_vec_partition_numActiveNodes.resize(m_num_partitions);

            // uint32_t *outDegreeCounter = new uint32_t[m_num_nodes];
            // uint32_t location;
            // for (uint32_t i = 0; i < m_num_edges; i++)
            // {
            //     location = m_host_arr_node_edgeStartIndex_CSR[edges[i].source] + degree[edges[i].source];
            //     m_host_arr_edgeList[location] = edges[i].end;
            // }

            printf("The graph has %d vertices, and %ld edges (average_degree: %f, max_degree: %d)\n", m_num_nodes, m_num_edges, (float)m_num_edges / m_num_nodes, max_degree);
            print_partitioned_graph_info();
            printf("maximum_edgeListPerPartition: %d\n", maximum_edgeListPerPartition);

            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_node_value_datum, m_num_nodes * sizeof(TValue)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_node_buffer_datum, m_num_nodes * sizeof(TValue)));
            // CHECK_CUDA_ERROR(cudaMalloc(&m_dev_worklist, m_num_nodes * sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMalloc(&m_dev_worklist_counter, sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMemset(m_dev_worklist_counter, 0, sizeof(uint32_t)));
            printf("m_num_partitions: %d\n", m_num_partitions);
            for(int i = 0; i < m_num_partitions; i++)
            {
                uint32_t *dev_worklist;
                uint32_t node_partionSize = m_vec_partition_start_node[i + 1] - m_vec_partition_start_node[i];
                // printf("partition %d size: %d\n", i, node_partionSize);
                CHECK_CUDA_ERROR(cudaMallocManaged(&dev_worklist, node_partionSize * sizeof(uint32_t)));
                m_vec_managed_worklist_perPartition.push_back(dev_worklist);
                uint32_t *dev_worklist_counter;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_worklist_counter, sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemset(dev_worklist_counter, 0, sizeof(uint32_t)));
                m_vec_dev_worklist_counter_perPartition.push_back(dev_worklist_counter);

                uint32_t *dev_numActiveEdge_counter;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_numActiveEdge_counter, sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemset(dev_numActiveEdge_counter, 0, sizeof(uint32_t)));
                m_vec_dev_numActiveEdge_counter_perPartition.push_back(dev_numActiveEdge_counter);

            }
            printf("m_vec_managed_worklist_perPartition size: %d\n", m_vec_managed_worklist_perPartition.size());
            printf("m_vec_dev_worklist_counter_perPartition size: %d\n", m_vec_dev_worklist_counter_perPartition.size());

            for (int i = 0; i < FLAGS_nStreams; i++)
            {
                uint32_t *dev_edgeList;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_edgeList, (maximum_edgeListPerPartition) * sizeof(uint32_t)));
                m_vec_dev_edgeList_perStream.push_back(dev_edgeList);

            }
            // edges.clear();
            edges_map.clear();
            delete[] degree;
            // delete[] outDegreeCounter;
        }
        sw.stop();
        std::cout << "Loading graph Time: " << sw.ms() << " ms" << std::endl;
    };


    void reset_UM_edgeList()
    {
        //new a array temp to store m_unifiedMem_arr_edgeList content
        uint32_t *temp = new uint32_t[m_num_edges];
        //copy m_unifiedMem_arr_edgeList to temp
        memcpy(temp, m_unifiedMem_arr_edgeList, m_num_edges*sizeof(uint32_t));
        //memset m_unifiedMem_arr_edgeList to 0
        memset(m_unifiedMem_arr_edgeList, 0, m_num_edges*sizeof(uint32_t));
        printf("%u %u %u\n", m_unifiedMem_arr_edgeList[0], m_unifiedMem_arr_edgeList[m_num_edges/2], m_unifiedMem_arr_edgeList[m_num_edges-1]);
        //copy temp to m_unifiedMem_arr_edgeList
        memcpy(m_unifiedMem_arr_edgeList, temp, m_num_edges*sizeof(uint32_t));
        //free temp
        delete[] temp;
        printf("UM edgeList reset done\n");

    }

    void rankNodeByDegree(std::string filename)
    {

        Stopwatch sw(true);
        std::cout << "Loading graph from " << filename << std::endl;
        std::string fileExtension = get_File_Extension(filename);
        if (fileExtension == "el")
            m_isWeighted = false;
        else if (fileExtension == "wel")
            m_isWeighted = true;
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
        if (m_isWeighted)
        {
        }
        else
        {
            // std::vector<Edge> edges;
            std::map<uint32_t, std::vector<uint32_t>> edges_map;
            Edge newEdge;
            uint32_t max = 0;
            while (getline(infile, line))
            {
                ss.str("");
                ss.clear();
                ss << line;

                ss >> newEdge.source;
                ss >> newEdge.end;

                // edges.push_back(newEdge);
                // std::cout<<"newEdge.source: "<<newEdge.source<<" newEdge.end: "<<newEdge.end<<std::endl;
                edges_map[newEdge.source].push_back(newEdge.end);
                edgeCounter++;

                if (max < newEdge.source)
                    max = newEdge.source;
                if (max < newEdge.end)
                    max = newEdge.end;
            }
            infile.close();
            m_num_nodes = max + 1;
            m_num_edges = edgeCounter;
            std::vector<std::pair<uint32_t, uint32_t>> vec_node_degree;
            for(int i = 0; i < m_num_nodes; i++)
            {
                vec_node_degree.push_back(std::make_pair(i, edges_map[i].size()));
            }
            std::sort(vec_node_degree.begin(), vec_node_degree.end(), 
                [&](std::pair<uint32_t, uint32_t>& a, 
                    std::pair<uint32_t, uint32_t>& b) 
                { 
                    return a.second > b.second; 
                } 
            );
            

            std::cout<<"output node rank by degree"<<std::endl;
            uint32_t numNode_10segment = m_num_nodes / 10;
            for(int i = 0; i < 10; i++)
            {
                std::cout<<"\nsegment "<<i<<std::endl<<std::endl;
                int numRandNode = 10;
                if(i == 0)
                    numRandNode = 100;
                //randomly select a few nodes from each segment
                for(int j = 0; j < numRandNode; j++)
                {
                    int random_index = rand() % numNode_10segment + i * numNode_10segment;
                    std::cout<<vec_node_degree[random_index].first<<" "<<vec_node_degree[random_index].second<<std::endl;
                }
            }

            
        }
        sw.stop();
        std::cout << "Loading graph Time: " << sw.ms() << " ms" << std::endl;
    };

};



// template <typename T>
// class NodeOutputDatum
// {
// private:
//     std::vector<T> m_host_data;
//     DEV_GraphDatum<T> m_dev_datum;

// public:
//     NodeOutputDatum() : m_dev_datum(nullptr, 0) {}

//     ~NodeOutputDatum()
//     {
//         Deallocate();
//     }

//     void Deallocate()
//     {
//         if (m_dev_datum.data_ptr != nullptr)
//         {
//             CHECK_CUDA_ERROR(cudaFree(m_dev_datum.data_ptr));
//             m_dev_datum.data_ptr = nullptr;
//             m_dev_datum.size = 0;
//         }
//     }

//     void Allocate(Graph &graph)
//     {
//         Deallocate();
//         m_host_data.resize(graph.m_num_nodes);
//         CHECK_CUDA_ERROR(cudaMalloc(&m_dev_datum.data_ptr, graph.m_num_nodes * sizeof(T)));
//         m_dev_datum.size = graph.m_num_nodes;
//     }
// };
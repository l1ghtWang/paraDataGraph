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
            

class Graph
{
private:
    uint32_t m_num_nodes;
    uint32_t m_num_edges;
    bool m_isWeighted;
    uint32_t *m_host_arr_node_edgeStartIndex_CSR;
    uint32_t *m_host_arr_edgeList;
    uint32_t *m_dev_zeroCopy_arr_edgeList;
    std::vector<uint32_t> m_vec_host_node_value_datum;
    std::vector<uint32_t> m_vec_partition_start_node;
    std::vector<uint32_t> m_vec_partition_start_edge;
    std::vector<uint32_t *> m_vec_dev_edgeList_perPartition;
    uint32_t *m_dev_arr_node_edgeStartIndex_CSR;
    uint32_t *m_dev_node_value_datum;
    uint32_t *m_dev_node_buffer_datum;

    uint32_t m_num_partitions;

    std::vector<uint32_t *> m_vec_dev_worklist_perPartition;
    std::vector<uint32_t *> m_vec_dev_worklist_counter_perPartition;
    std::vector<uint32_t> m_vec_partition_numActiveNodes;



public:
    Graph() : m_num_nodes(0), m_num_edges(0), m_isWeighted(false),
              m_host_arr_node_edgeStartIndex_CSR(nullptr), m_host_arr_edgeList(nullptr), m_dev_zeroCopy_arr_edgeList(nullptr),
               m_num_partitions(0),
              m_dev_arr_node_edgeStartIndex_CSR(nullptr),
              m_dev_node_buffer_datum(nullptr), m_dev_node_value_datum(nullptr){};
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

        for(int i = 0; i < m_vec_dev_worklist_perPartition.size(); i++)
        {
            if(m_vec_dev_worklist_perPartition[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_dev_worklist_perPartition[i]));
        }
        for(int i = 0; i < m_vec_dev_worklist_counter_perPartition.size(); i++)
        {
            if(m_vec_dev_worklist_counter_perPartition[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_dev_worklist_counter_perPartition[i]));
        }
        for(int i = 0; i < m_vec_dev_edgeList_perPartition.size(); i++)
        {
            if(m_vec_dev_edgeList_perPartition[i] != nullptr)
                CHECK_CUDA_ERROR(cudaFree(m_vec_dev_edgeList_perPartition[i]));
        }
    };

    std::vector<uint32_t> getherValues()
    {

        uint32_t* host_node_value_datum = new uint32_t[m_num_nodes];
        CHECK_CUDA_ERROR(cudaMemcpy(host_node_value_datum, m_dev_node_value_datum, m_num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
    uint32_t get_num_partitions()
    {
        return m_num_partitions;
    }

    uint32_t get_partition_numActiveNodes(uint32_t partition_id)
    {
        return m_vec_partition_numActiveNodes[partition_id];
    }
    void set_partition_numActiveNodes(uint32_t partition_id, uint32_t numActiveNodes)
    {
        m_vec_partition_numActiveNodes[partition_id] = numActiveNodes;
    }

    uint32_t get_host_array_node_edgeStartIndex_CSR(uint32_t node_id)
    {
        return m_host_arr_node_edgeStartIndex_CSR[node_id];
    }


    uint32_t* get_device_node_edgeStartIndex_CSR_ptr()
    {
        return m_dev_arr_node_edgeStartIndex_CSR;
    }

    uint32_t* get_host_array_edgeList_ptr()
    {
        return m_host_arr_edgeList;
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

    uint32_t * get_device_value_ptr()
    {
        return m_dev_node_value_datum;
    }
    uint32_t *get_device_buffer_ptr()
    {
        return m_dev_node_buffer_datum;
    }

    uint32_t *get_device_edgeList_ptr(uint32_t streamIdx)
    {
        return m_vec_dev_edgeList_perPartition[streamIdx];
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
        return m_vec_dev_worklist_perPartition;
    }

    std::vector<uint32_t *> get_vector_device_worklist_counter_ptr()
    {
        return m_vec_dev_worklist_counter_perPartition;
    }

    uint32_t get_worklist_counter_value(uint32_t partition_id, cudaStream_t streamIdx)
    {
        uint32_t worklist_counter;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&worklist_counter, m_vec_dev_worklist_counter_perPartition[partition_id], sizeof(uint32_t), cudaMemcpyDeviceToHost, streamIdx));
        return worklist_counter;
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
            m_host_arr_node_edgeStartIndex_CSR = new uint32_t[m_num_nodes + 1];
            CHECK_CUDA_ERROR(cudaMallocHost(&m_host_arr_edgeList, (m_num_edges) * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_dev_zeroCopy_arr_edgeList, m_host_arr_edgeList, 0));
            
            uint32_t *degree = new uint32_t[m_num_nodes];
            uint32_t counter = 0;
            for (uint32_t i = 0; i < m_num_nodes; i++)
            {
                degree[i] = edges_map[i].size();
                std::sort(edges_map[i].begin(), edges_map[i].end());
                for(int j = 0; j < edges_map[i].size(); j++)
                {
                    m_host_arr_edgeList[counter + j] = edges_map[i][j];
                    if(i == 0)
                    {
                        printf("node %d , edge %d , to: %d\n", i, j, edges_map[i][j]);
                    
                    }
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

            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_node_value_datum, m_num_nodes * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dev_node_buffer_datum, m_num_nodes * sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMalloc(&m_dev_worklist, m_num_nodes * sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMalloc(&m_dev_worklist_counter, sizeof(uint32_t)));
            // CHECK_CUDA_ERROR(cudaMemset(m_dev_worklist_counter, 0, sizeof(uint32_t)));
            printf("m_num_partitions: %d\n", m_num_partitions);
            for(int i = 0; i < m_num_partitions; i++)
            {
                uint32_t *dev_worklist;
                uint32_t node_partionSize = m_vec_partition_start_node[i + 1] - m_vec_partition_start_node[i];
                // printf("partition %d size: %d\n", i, node_partionSize);
                CHECK_CUDA_ERROR(cudaMalloc(&dev_worklist, node_partionSize * sizeof(uint32_t)));
                m_vec_dev_worklist_perPartition.push_back(dev_worklist);
                uint32_t *dev_worklist_counter;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_worklist_counter, sizeof(uint32_t)));
                CHECK_CUDA_ERROR(cudaMemset(dev_worklist_counter, 0, sizeof(uint32_t)));
                m_vec_dev_worklist_counter_perPartition.push_back(dev_worklist_counter);
            }
            printf("m_vec_dev_worklist_perPartition size: %d\n", m_vec_dev_worklist_perPartition.size());
            printf("m_vec_dev_worklist_counter_perPartition size: %d\n", m_vec_dev_worklist_counter_perPartition.size());

            for (int i = 0; i < FLAGS_nStreams; i++)
            {
                uint32_t *dev_edgeList;
                CHECK_CUDA_ERROR(cudaMalloc(&dev_edgeList, (maximum_edgeListPerPartition) * sizeof(uint32_t)));
                m_vec_dev_edgeList_perPartition.push_back(dev_edgeList);

            }
            // edges.clear();
            edges_map.clear();
            delete[] degree;
            // delete[] outDegreeCounter;
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
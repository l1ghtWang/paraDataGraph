
# include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories("${PROJECT_BINARY_DIR}")


cuda_add_executable(testGraphLoad testGraphLoad.cu)
target_link_libraries(testGraphLoad ${EXTRA_LIBS})


cuda_add_executable(test_SSSP test_SSSP.cu)
target_link_libraries(test_SSSP ${EXTRA_LIBS})

cuda_add_executable(test_SSSP_EFonly test_SSSP_EFonly.cu)
target_link_libraries(test_SSSP_EFonly ${EXTRA_LIBS})

cuda_add_executable(test_SSSP_UMonly test_SSSP_UMonly.cu)
target_link_libraries(test_SSSP_UMonly ${EXTRA_LIBS})

cuda_add_executable(test_SSSP_ZConly test_SSSP_ZConly.cu)
target_link_libraries(test_SSSP_ZConly ${EXTRA_LIBS})

cuda_add_executable(test_SSSP_ZConly_partitionInfo_activeNode test_SSSP_ZConly_partitionInfo_activeNode.cu)
target_link_libraries(test_SSSP_ZConly_partitionInfo_activeNode ${EXTRA_LIBS})

cuda_add_executable(test_rankNodeByDegree test_rankNodeByDegree.cu)
target_link_libraries(test_rankNodeByDegree ${EXTRA_LIBS})



cuda_add_executable(test_PageRank test_PageRank.cu)
target_link_libraries(test_PageRank ${EXTRA_LIBS})
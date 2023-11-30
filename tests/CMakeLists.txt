
# include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories("${PROJECT_BINARY_DIR}")
cuda_add_executable(testGraphLoad testGraphLoad.cu)
target_link_libraries(testGraphLoad ${EXTRA_LIBS})
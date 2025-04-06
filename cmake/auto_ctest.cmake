set(test ${CMAKE_CURRENT_LIST_DIR}/auto_ctest_impl.py)

execute_process(
    COMMAND python3 ${test} ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

message("${output}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "init test env failed: ${error}")
endif()

enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/CMakeTestAutoGeneration.cmake)
    
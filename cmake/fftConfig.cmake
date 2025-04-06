# fftConfig.cmake - Configuration file for the fft package.

# Ensure that this script is included only once.
if(TARGET fft)
    message(WARNING "NO fft found")
    return()
endif()
message(STATUS "Found fft")

# Get the directory where this file is located.
get_filename_component(_current_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)

# # Include the exported targets file.
include("${_current_dir}/fftTargets.cmake")
# Set the package version variables.
set(fft_VERSION_MAJOR 1) # Replace with your major version
set(fft_VERSION_MINOR 0) # Replace with your minor version
set(fft_VERSION_PATCH 0) # Replace with your patch version
set(fft_VERSION "${fft_VERSION_MAJOR}.${fft_VERSION_MINOR}.${fft_VERSION_PATCH}")

# Check if the requested version is compatible.
if(NOT "${fft_FIND_VERSION}" STREQUAL "")
    if(NOT "${fft_FIND_VERSION}" VERSION_LESS "${fft_VERSION}")
        set(fft_VERSION_COMPATIBLE TRUE)
    endif()

    if("${fft_FIND_VERSION}" VERSION_EQUAL "${fft_VERSION}")
        set(fft_VERSION_EXACT TRUE)
    endif()
endif()

find_package(MKL CONFIG REQUIRED PATHS /opt/intel/oneapi/mkl/latest/)
# link_libraries(MKL::MKL)
link_libraries( ${MKL_IMPORTED_TARGETS})

message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

# Mark the package as found.
set(fft_FOUND TRUE)
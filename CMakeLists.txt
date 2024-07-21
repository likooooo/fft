cmake_minimum_required(VERSION 3.22)
project(fftw3)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# fftw
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW )
link_libraries     (PkgConfig::FFTW)
pkg_search_module(FFTWF REQUIRED fftw3f IMPORTED_TARGET)
include_directories(PkgConfig::FFTWF )
link_libraries     (PkgConfig::FFTWF)

# eigen header-only
pkg_search_module(EIGEN REQUIRED eigen3 IMPORTED_TARGET)
include_directories(PkgConfig::EIGEN )

# mkl
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
link_libraries(MKL::MKL)
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

# add executable
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB list ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
message(STATUS "add executable: ${list}")
foreach(cpp IN LISTS list)
  get_filename_component(base_name ${cpp} NAME_WLE)
  message(${base_name})
  add_executable(${base_name} ${cpp})
  target_link_libraries(${base_name}  )
endforeach()
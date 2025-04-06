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
find_package(mekil)
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

include(${PROJECT_SOURCE_DIR}/cmake/auto_ctest.cmake)


file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")
set(PACKAGE_INCLUDE_INSTALL_DIR include/fft)
install(FILES ${HEADERS} DESTINATION ${PACKAGE_INCLUDE_INSTALL_DIR})

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftConfig.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftConfig-version.cmake"
    DESTINATION cmake
)
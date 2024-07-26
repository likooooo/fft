#pragma once
#include <eigen3/Eigen/Dense>
#include "fftw_plan.hpp"


template<class T>
inline auto mapping_to_matrix(T* p, int xsize, int ysize, int stride = 0)
{
    using namespace Eigen;
    if(0 == stride) stride = xsize;
    return Map<MatrixX<T>, 0, Stride<Dynamic, 1>>(
        p, xsize, ysize, 
        Stride<Dynamic, 1>(stride, 1)
    );
}


template<class T1, class T2>
inline auto cal_eps(const T1&a, const  T2& b)
{
    Eigen::Matrix<typename T1::Scalar, Eigen::Dynamic, Eigen::Dynamic> diff = a - b;
    for(auto& x : diff.reshaped()) x = std::abs(x);
    auto max_error = std::real(
        *std::max_element(
            diff.reshaped().begin(), 
            diff.reshaped().end(), 
            [](auto a, auto b){ 
                return std::abs(a) < std::abs(b);
            }
        )
    );
    return std::make_tuple(diff, max_error);
}
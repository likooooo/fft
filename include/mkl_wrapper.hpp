#pragma once
#include "mkl.h"
#include <complex>

enum class matrix_major
{
    row_major, col_major
};
char major = 'r';
template<class T>
void transpose(T* p, int sizey, int sizex)
{
    if constexpr(std::is_same_v<float, std::remove_cv_t<T>>)
    {    
        // mkl_simatcopy(major, 'T', sizex, sizey, T(1), p, sizey, sizex);
        mkl_simatcopy(major, 'T', sizey, sizex, T(1), p, sizex, sizey);

    }
    else if constexpr(std::is_same_v<double, std::remove_cv_t<T>>)
    {
        mkl_dimatcopy(major, 'T', sizey, sizex, T(1), p, sizex, sizey);
    }
    else if constexpr(std::is_same_v<std::complex<float>, std::remove_cv_t<T>>)
    {
        mkl_cimatcopy(major, 'T', sizey, sizex, MKL_Complex8(1), (MKL_Complex8*)p, sizex, sizey);
    }
    else if constexpr(std::is_same_v<std::complex<double>, std::remove_cv_t<T>>)
    {
        mkl_zimatcopy(major, 'T', sizey, sizex, MKL_Complex16(1), (MKL_Complex16*)p, sizex, sizey);
    }
}
#pragma once
#include "mkl.h"
#include <complex>

enum class matrix_major
{
    row_major,
    col_major
};
template <class T>
constexpr static bool is_c = std::is_same_v<std::complex<float>, std::remove_cv_t<T>>;
template <class T>
constexpr static bool is_s = std::is_same_v<float, std::remove_cv_t<T>>;
template <class T>
constexpr static bool is_d = std::is_same_v<double, std::remove_cv_t<T>>;
template <class T>
constexpr static bool is_z = std::is_same_v<std::complex<double>, std::remove_cv_t<T>>;

template <class T>
inline void CenterCornerFlip(T *image, int widht, int height)
{
    const int sizeX = widht;
    const int sizeY = height;
    const int halfSizeX = sizeX / 2;
    const int halfSizeY = sizeY / 2;
    
    T *pA = image;
    T *pB = image + halfSizeX + sizeX % 2;
    T *pC = image + (halfSizeY  + sizeY % 2) * sizeX;
    T *pD = image + (halfSizeY  + sizeY % 2) * sizeX + halfSizeX + sizeX % 2;

    for (int i = 0; i < halfSizeY; i++)
    {
        if constexpr (is_s<T>)
        {
            cblas_sswap(halfSizeX, pA, 1, pD, 1);
            cblas_sswap(halfSizeX, pB, 1, pC, 1);
        }
        else if constexpr (is_c<T>)
        {
            cblas_cswap(halfSizeX, pA, 1, pD, 1);
            cblas_cswap(halfSizeX, pB, 1, pC, 1);
        }
        else if constexpr (is_d<T>)
        {
            cblas_dswap(halfSizeX, pA, 1, pD, 1);
            cblas_dswap(halfSizeX, pB, 1, pC, 1);
        }
        else if constexpr (is_z<T>)
        {
            cblas_zswap(halfSizeX, pA, 1, pD, 1);
            cblas_zswap(halfSizeX, pB, 1, pC, 1);
        }
        pA += sizeX;
        pD += sizeX;
        pB += sizeX;
        pC += sizeX;
    }
}
template<class T>inline void CropImage(T* pOut, const int ostride, const T* pIn, const int istride, const int sizex, const int sizey)
{
    if constexpr (is_s<T>) cblas_scopy_batch_strided(sizex, pIn, 1, istride, pOut, 1, ostride, sizey);
    else if constexpr (is_d<T>) cblas_dcopy_batch_strided(sizex, pIn, 1, istride, pOut, 1, ostride, sizey);
    else if constexpr (is_c<T>) cblas_ccopy_batch_strided(sizex, pIn, 1, istride, pOut, 1, ostride, sizey);
    else if constexpr (is_z<T>) cblas_zcopy_batch_strided(sizex, pIn, 1, istride, pOut, 1, ostride, sizey);
}
template<class T> inline void VecDiv(int n, T*a, T*b, T* y)
{
    if constexpr(is_s<T>)
    {
        vsDiv(n, a, b, y);
    }
    else if constexpr(is_d<T>)
    {
        vdDiv(n, a, b, y);
    }
    else if constexpr(is_c<T>)
    {
        vcDiv(n, (MKL_Complex8*)a, (MKL_Complex8*)b, (MKL_Complex8*)y);
    }
    else if constexpr(is_z<T>)
    {
        vzDiv(n, (MKL_Complex16*)a, (MKL_Complex16*)b, (MKL_Complex16*)y);
    }
}
template<class T> inline void VecMul(int n, T*a, T*b, T* y)
{
    if constexpr(is_s<T>)
    {
        vsMul(n, a, b, y);
    }
    else if constexpr(is_d<T>)
    {
        vdMul(n, a, b, y);
    }
    else if constexpr(is_c<T>)
    {
        vcMul(n, (MKL_Complex8*)a, (MKL_Complex8*)b, (MKL_Complex8*)y);
    }
    else if constexpr(is_z<T>)
    {
        vzMul(n, (MKL_Complex16*)a, (MKL_Complex16*)b, (MKL_Complex16*)y);
    }
}
template<class T> inline void VecScala(int n, const T a, T* x, int inc = 1)
{
    if constexpr(is_s<T>)
    {
        cblas_sscal(n, a, x, inc);
    }
    else if constexpr(is_d<T>)
    {
        cblas_dscal(n, a, x, inc);
    }
    else if constexpr(is_c<T>)
    {
        cblas_cscal(n, (MKL_Complex8*)&a, (MKL_Complex8*)x, inc);
    }
    else if constexpr(is_z<T>)
    {
        cblas_zscal(n, (MKL_Complex16*)&a, (MKL_Complex16*)x, inc);
    }
}

#ifdef UNFINISHED_CODE
char major = 'r';
template <class T>
void transpose(T *p, int sizey, int sizex)
{
    if constexpr (std::is_same_v<float, std::remove_cv_t<T>>)
    {
        // mkl_simatcopy(major, 'T', sizex, sizey, T(1), p, sizey, sizex);
        mkl_simatcopy(major, 'T', sizey, sizex, T(1), p, sizex, sizey);
    }
    else if constexpr (std::is_same_v<double, std::remove_cv_t<T>>)
    {
        mkl_dimatcopy(major, 'T', sizey, sizex, T(1), p, sizex, sizey);
    }
    else if constexpr (std::is_same_v<std::complex<float>, std::remove_cv_t<T>>)
    {
        mkl_cimatcopy(major, 'T', sizey, sizex, MKL_Complex8(1), (MKL_Complex8 *)p, sizex, sizey);
    }
    else if constexpr (std::is_same_v<std::complex<double>, std::remove_cv_t<T>>)
    {
        mkl_zimatcopy(major, 'T', sizey, sizex, MKL_Complex16(1), (MKL_Complex16 *)p, sizex, sizey);
    }
}
#endif
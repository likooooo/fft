#pragma once
#include "fftw_plan.hpp"
#include <mekil/mkl_wrapper.hpp>

template<class T, class TTo>
void conv(T* p, T* k,const int xsize, const int ysize)
{
    using plan = plan_holder<T, TTo>;
    auto pFFT  = plan::make_plan(p, (TTo*)p, xsize, ysize);
    auto pIFFT = plan::make_inv_plan(pFFT);
    auto pFFTKernel  = plan::make_plan(k, (TTo*)k, xsize, ysize);

    fft_operator::transform(pFFT);
    fft_operator::transform(pFFTKernel);
    if constexpr(plan::is_same_type)
        VecMul<TTo>(xsize * ysize, p, k, p);
    else
        VecMul<TTo>((xsize / 2 + 1) * ysize, (TTo*)p, (TTo*)k, (TTo*)p);
    fft_operator::transform(pIFFT);
    VecScala(xsize * ysize, T(1) / T(xsize * ysize), p);
}
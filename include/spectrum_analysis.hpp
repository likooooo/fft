#pragma once
#include "fftw_allocator.hpp"
#include "fftw_plan.hpp"
#include <cmath>
#include <mekil/mkl_wrapper.hpp>

template<class T, class TReal = typename fft_scalar_type<T>::value_type>
void phase_modulate(T* p, const int NX, const int xsize, const int ysize, const TReal dx, const TReal dy)
{
    static_assert(fft_scalar_type<T>::is_complex_type);
    using complex = std::complex<TReal>;
    const int NY = ysize;
    auto phase = [](int ix, int NX, TReal dx){
        TReal k = (ix - NX * int(ix >= NX/2)) * (-dx / NX);
        return std::exp(complex(0, 1) * TReal(k * M_PI * 2));
    };
    dynamic_vec<complex> xPhase(xsize);
    for(int x = 0; x < xsize; x++) xPhase[x] = phase(x, NX, dx);
    
    for(int y = 0; y < ysize; y++, p += xsize)
    {
        auto yPhase = phase(y, NY, dy);
        VecMul(xsize, p, xPhase.data(), p);
        VecScala(xsize, yPhase, p);
    }
}

template<class T, class TTo, class TReal = typename fft_scalar_type<T>::value_type>
void shift(T* p, const int xsize, const int ysize, const TReal dx, const TReal dy)
{
    using plan = plan_holder<T, TTo>;
    auto pFFT  = plan::make_plan(p, (TTo*)p, xsize, ysize);
    auto pIFFT = plan::make_inv_plan(pFFT);
    fft_operator::transform(pFFT);
    if constexpr(plan::is_same_type)
        phase_modulate((TTo*)p, 
            xsize, xsize, ysize,
            dx, dy
        );
    else
       phase_modulate((TTo*)p, 
            xsize, (xsize / 2 + 1), ysize,
            dx, dy
        );
    fft_operator::transform(pIFFT);
    VecScala(xsize * ysize, T(1.0/(xsize * ysize)), p);
}
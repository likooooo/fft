#pragma once
#include "fftw_plan.hpp"
#include <cmath>
#include "mkl_wrapper.hpp"

void cross_spectrum()
{

}
template<class T> void center_corner_flip_with(T* pInFreq, int xsize, int ysize) requires (is_c<T> || is_z<T>)
{
    for(int y = 0; y < ysize; y++)
    {
        for(int x = 0; x < xsize; x++)
        {
            if((x%2 + y%2) == 1)
            {
                pInFreq[y*xsize + x] *= -1;
            }        
        }
    }
}
template<class T, class TReal = typename fft_scalar_type<T>::value_type>
void phase_modulate(T* p, const int NX, const int xsize, const int ysize, const TReal dx, const TReal dy)
    requires fft_scalar_type<T>::is_complex_type
{
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
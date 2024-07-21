#pragma once
#include <fftw3.h>
#include <type_traits>
#include <vector>
#include <span>
#include "fftw_plan.hpp"
template<class T>
struct fftw_allocator
{
    using floating_point_type = typename fft_scalar_type<T>::value_type;
    using value_type = T;

    static_assert(std::is_standard_layout_v<T>, "T of fftw_allocator should be standard layout");
    fftw_allocator() = default;
    ~fftw_allocator() = default;

    constexpr T*allocate(size_t N)
    { 
        T* p = nullptr;
        if constexpr(sizeof(floating_point_type) == 4 )
            p = (T*)fftwf_malloc(sizeof(T) * N);
        else if constexpr(sizeof(floating_point_type) == 8 )
            p = (T*)fftw_malloc(sizeof(T) * N);
        return p;
    }
    constexpr void deallocate(T* p, size_t N)
    {
        if constexpr(sizeof(floating_point_type) == 4 )
            fftwf_free(p);
        else if constexpr(sizeof(floating_point_type) == 8 )
            fftw_free(p);
    }
};

template<class T> using dynamic_vec = std::vector<T, fftw_allocator<T>>;

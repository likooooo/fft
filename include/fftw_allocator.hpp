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

template<class T, class TDim,class ...TDims> std::tuple<dynamic_vec<T>, size_t> make_vec(TDim d0, TDims ...rest)
{
    std::array<TDim, sizeof...(rest) + 1> n{d0, rest...};
    auto prod = std::accumulate(n.begin() + 1, n.end(), (size_t)1, [](auto a, auto b){return a*b;});
    auto withpadding = (std::is_floating_point_v<T> ? (n[0] / 2 + 1)* 2 : n[0]);
    return std::make_tuple(dynamic_vec<T>(prod * withpadding), withpadding - n[0]);
}
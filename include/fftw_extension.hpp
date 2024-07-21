#pragma once
#include <span>
#include <iostream>
#include <random>
#include "fftw_allocator.hpp"
#if __cplusplus < 201103L
    namespace std
    {
        template<class T>
        struct span;//TODO
    }
#endif


template<class T, size_t extent = std::dynamic_extent>
inline std::ostream &operator<<(std::ostream &out, std::span<T, extent> span)
{
    out << "[";
    auto write = [&](auto span){
        int i = 0;
        for(; i < span.size() - 1; i++) out <<span.data()[i]<<", ";
        out << span.data()[i];
    };
    constexpr int max_count = 10;
    if(span.size() < max_count)
    {
        write(span);
    }
    else
    {
        constexpr int size = max_count / 3;
        write(std::span<T, extent>(span.begin(), span.begin() + size));
        out << ", ...";
        write(std::span<T, extent>(span.end() - 1 - size, span.end()));
    }

    out<< "]";
    return out;
}
template<typename T>
concept container_with_fftw_alloc = requires (T t)
{
    requires std::is_same_v<
        fftw_allocator<typename T::value_type>, 
        typename std::remove_cvref_t<T>::allocator_type
    >;
};
template<class TContainer> requires container_with_fftw_alloc<TContainer>
inline std::ostream &operator<<(std::ostream &out, const TContainer& container)
{
    return out << std::span(container.begin(), container.end());
}

template<class T> inline bool operator==(const dynamic_vec<T>& x, const dynamic_vec<T>& y)
{
    if(x.size() == y.size())
    {
        for(int i = 0; i < x.size(); i++)
        {
            if(std::abs(x[i] - y[i]) > 1e-6) 
            {
                std::cout << "error at " << i << " "<< x[i] << " " << y[i]<< " " << std::abs(x[i] - y[i]) << "\n";
                return false;
            }
        }
        return true;
    }
    return false;
}
// TODO
enum class random_distribution
{
    uniform,
    bernoulli,
    poisson,
    normal,
    sampling,
};
template<class T, 
    random_distribution N = random_distribution::uniform
>
struct random_operator
{
    template<class ...TArgs>
      T operator()(TArgs... args)
    {  
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr(N == random_distribution::uniform)
        {
            if constexpr( std::is_integral_v<T>)
            {
                return std::uniform_int_distribution<T>(((T)args)...)(gen);
            }
            else if constexpr(std::is_floating_point_v<T>)
            {
                return std::uniform_real_distribution<T>(((T)args)...)(gen);
            }
        }
        else if constexpr(N == random_distribution::bernoulli)
        {
            // std::bernoulli_distribution d(0.25);
        }
        return T();
    }
};
template<class T> requires std::is_floating_point_v<T> 
T phi(T x1, T x2){
    return (std::erf(x2 / std::sqrt<T>(2)) - std::erf(x1 / std::sqrt<T>(2))) / 2;
}
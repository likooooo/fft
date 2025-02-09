#pragma once
#include <fftw3.h>
#include <array>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <memory>
#include <mekil/mkl_wrapper.hpp>
#include "eigen_wrapper.hpp"
#include <eigen3/Eigen/Dense>
template<class T>struct TemplateUnreachableAssert{
    TemplateUnreachableAssert(){
        static_assert(std::is_same<T, T>::value == false, "template unreachable");
    }
};

template<class T> struct fft_scalar_type{
    using value_type = T;
    static_assert(std::is_floating_point_v<value_type>);
    constexpr static bool is_complex_type = false;
};
template<class T> struct fft_scalar_type<std::complex<T>>{
    using value_type = typename fft_scalar_type<T>::value_type;  
    constexpr static bool is_complex_type = true;
};
enum class fft_domain: bool  
{
    space = 0, freq
};
template<matrix_major m, int ...N> 
struct fft_dim
{
    constexpr static bool is_row_major = m == matrix_major::row_major;
    template<class ...Args> constexpr static auto get(Args... args)
    {
        auto dims = get_impl(std::forward<Args>(args)...);
        update_dim_sequence(dims);
        return dims;
    }
    template<class TDim> static void update_dim_sequence(TDim&& dim)
    {
        if constexpr(!is_row_major) std::reverse(dim.begin(), dim.end());
    }
private:
    template<class ...Args> constexpr static auto get_impl(Args... args)
    {
        constexpr int input_rank = sizeof...(args);
        if constexpr(input_rank)
        {
            return std::array<int, input_rank>{args...};
        }
        else
        {
            return std::array{N...};
        }
    }
};
template<
    class T/*TFrom*/, class TTo = T,  
    unsigned flag = FFTW_ESTIMATE, 
    fft_domain current_domain = fft_domain::space,
    matrix_major major = matrix_major::col_major,
    class dims = fft_dim<major>
> struct plan_holder
{
    using floating_point_type = typename fft_scalar_type<T>::value_type;
    using plan_ptr_type = std::conditional_t<
        std::is_same_v<float, floating_point_type>,
        fftwf_plan, 
        fftw_plan
    >;
    using plan_type = std::remove_pointer_t<plan_ptr_type>;
    constexpr static bool is_complex_type = fft_scalar_type<T>::is_complex_type;
    constexpr static bool is_same_type = std::is_same_v<T, TTo>;
    constexpr static bool is_freq = current_domain == fft_domain::freq; 
    constexpr static bool is_row_major = dims::is_row_major;
    constexpr static int  direction = 1 - 2 * static_cast<int>(!is_freq);
    constexpr static bool template_valid_check()
    {
        constexpr bool invalid_if = is_freq && !is_complex_type;
        return !(invalid_if);
    }
    static_assert(template_valid_check(), "invalid template");

    static void destory_plan(plan_ptr_type p)
    {
        if constexpr(sizeof(floating_point_type) ==4 )
            fftwf_destroy_plan(p);
        else if constexpr(sizeof(floating_point_type) == 8 )
            fftw_destroy_plan(p);
        else
            TemplateUnreachableAssert<void>();
    }
    template<class TDim> struct plan_deleter_with_meta
    { 
        using plan_holder_type = plan_holder; 
        using dim_type = TDim;
        T* pFrom;
        TTo* pTo;
        TDim row_major_dim;
        void operator()(plan_ptr_type p){destory_plan(p);}
    };
    template<class ...Args> constexpr static 
        auto make_plan(T* pFrom, TTo* pTo, Args ...args)
    {
        const auto dim = dims().get(std::forward<Args>(args)...);
        using TDim = std::remove_cv_t<decltype(dim)>;
        using deleter = plan_deleter_with_meta<TDim>;
        static_assert(std::is_standard_layout_v<deleter>);
        using ptr = std::unique_ptr<plan_type, deleter>;

        ptr p;
        if constexpr(is_same_type && is_complex_type)
        {
            p = ptr(
                plan_c2c(pFrom, pTo, dim),
                deleter{pFrom, pTo, dim}
            );
        }
        else if constexpr(is_complex_type)
        {
            p = ptr(
                plan_c2r(pFrom, pTo, dim),
                deleter{pFrom, pTo, dim}
            );
        }
        else //r2c
        {
            p = ptr(
                plan_r2c(pFrom, pTo, dim),
                deleter{pFrom, pTo, dim}
            );
        }
        return p;
    }
    template<class smart_ptr_type> constexpr static void transform(const smart_ptr_type& pPlan)
    {
        static_assert(is_same_type, "fft with r2c & c2r ,please use fft_operator::transform");
        const auto meta = pPlan.get_deleter();
        if constexpr(sizeof(floating_point_type) == 4 )
            fftwf_execute(pPlan.get());
        else if constexpr(sizeof(floating_point_type) == 8 )
            fftw_execute(pPlan.get());
    }
    template<class fft_plan_type, class deleter> constexpr static auto make_inv_plan(const std::unique_ptr<fft_plan_type, deleter>& pPlan)
    {
        using inv_plan_holder_type = plan_holder<TTo, T, flag, change_domain(current_domain), major, dims>; 
        auto meta = pPlan.get_deleter();
        using Indx = std::make_index_sequence<meta.row_major_dim.size()>;
        constexpr auto _make_plan = []<std::size_t... I>(auto pFrom, auto pTo, auto dim, std::index_sequence<I...>) { 
            return inv_plan_holder_type::make_plan(pFrom, pTo, dim[I]...);
        };
        dims::update_dim_sequence(meta.row_major_dim);
        return _make_plan(meta.pTo, meta.pFrom, meta.row_major_dim, Indx{});
    }
    constexpr static auto sprint_plan(const plan_ptr_type p)
    {
        struct deleter{void operator()(char *s){free(s);}};
        char* s;
        if constexpr(sizeof(floating_point_type) == 4 )
            s = fftwf_sprint_plan(p);
        else if constexpr(sizeof(floating_point_type) == 8 )
            s = fftw_sprint_plan(p);
        return std::unique_ptr<char, deleter>(s, deleter());
    }
    
    template<class TDim> constexpr static plan_ptr_type plan_c2c(T* pFrom, TTo* pTo, const TDim& dim)
    {
        const int rank = dim.size();
        if constexpr(sizeof(floating_point_type) == 4 )
        {
            return fftwf_plan_dft(
                rank, dim.data(),
                reinterpret_cast<fftwf_complex*>(pFrom),
                reinterpret_cast<fftwf_complex*>(pTo), 
                direction, flag
            );
        }
        else if constexpr(sizeof(floating_point_type) == 8 ) // TODO
        {
            return fftw_plan_dft(
                rank, dim.data(),
                reinterpret_cast<fftw_complex*>(pFrom),
                reinterpret_cast<fftw_complex*>(pTo), 
                direction, flag
            );
        }
        else
        {
            TemplateUnreachableAssert<void>();
        }
    }
    template<class TDim> constexpr static plan_ptr_type plan_c2r(T* pFrom, TTo* pTo, TDim dim)
    {
        static_assert(direction == FFTW_BACKWARD);
        const int rank = dim.size();
        // dim[0] += 1;
        if constexpr(sizeof(floating_point_type) == 4 )
        {
            return fftwf_plan_dft_c2r(
                rank, dim.data(),
                reinterpret_cast<fftwf_complex*>(pFrom),
                pTo, flag
            );
        }
        else if constexpr(sizeof(floating_point_type) == 8 ) // TODO
        {
            return fftw_plan_dft_c2r(
                rank, dim.data(),
                reinterpret_cast<fftw_complex*>(pFrom),
                pTo, flag
            );
        }
        else
        {
            TemplateUnreachableAssert<void>();
        }
    }
    template<class TDim> constexpr static plan_ptr_type plan_r2c(T* pFrom, TTo* pTo, const TDim& dim)
    {
        const int rank = dim.size();
        if constexpr(sizeof(floating_point_type) == 4 )
        {
            static_assert(direction == FFTW_FORWARD);
            return fftwf_plan_dft_r2c(
                rank, dim.data(), pFrom, 
                reinterpret_cast<fftwf_complex*>(pTo), flag
            );
        }
        else if constexpr(sizeof(floating_point_type) == 8 ) // TODO
        {
            static_assert(direction == FFTW_FORWARD);
            return fftw_plan_dft_r2c(
                rank, dim.data(), pFrom, 
                reinterpret_cast<fftw_complex*>(pTo), flag
            );
        }
        else
        {
            TemplateUnreachableAssert<void>();
        }
                
    }
private:
    constexpr static fft_domain change_domain(fft_domain current){ return static_cast<fft_domain>(!static_cast<bool>(current)); } 
};

namespace fft_operator
{
    template<class plan_type, class meta_type> 
    auto get_meta(const std::unique_ptr<plan_type, meta_type>& pPlan)
    {
        using plan_holder = meta_type::plan_holder_type;
        auto& meta = pPlan.get_deleter();
        return std::make_tuple(
            meta.pFrom, meta.pTo, 
            // fft_domain::freq,
            (plan_holder::is_freq ? fft_domain::freq : fft_domain::space), 
            meta.row_major_dim
        );
    }
    template<class smart_ptr_type> using plan_type = typename smart_ptr_type::deleter_type::plan_holder_type;

    template<class plan_type, class meta_type>
    auto transform(const std::unique_ptr<plan_type, meta_type>& pPlan)
    {
        using plan_holder = meta_type::plan_holder_type;
        using floating_point_type = plan_holder::floating_point_type;
        constexpr int floating_byte_count = sizeof(floating_point_type);
        const auto meta = pPlan.get_deleter();
        if constexpr (!plan_holder::is_same_type && !plan_holder::is_freq)
        {
            if ((void*)meta.pFrom == (void*)meta.pTo)
            {
                const auto dim = meta.row_major_dim;
                auto xsize = dim.back();
                auto ysize = std::accumulate(dim.begin(), dim.end() - 1, 1, [](auto a, auto b) {return a * b; });
                auto mat = mapping_to_matrix(meta.pFrom, xsize, ysize, (xsize / 2 + 1) * 2);
                mat = Eigen::MatrixX<floating_point_type>(mapping_to_matrix(meta.pFrom, xsize, ysize));
            }
        }
        if constexpr (floating_byte_count == 4)
            fftwf_execute(pPlan.get());
        else if constexpr (floating_byte_count == 8)
            fftw_execute(pPlan.get());
        if constexpr (!plan_holder::is_same_type && plan_holder::is_freq)
        {
            if ((void*)meta.pFrom == (void*)meta.pTo)
            {
                const auto dim = meta.row_major_dim;
                auto xsize = dim.back();
                auto ysize = std::accumulate(dim.begin(), dim.end() - 1, 1, [](auto a, auto b) {return a * b; });
                auto mat = mapping_to_matrix(meta.pTo, xsize, ysize, (xsize / 2 + 1) * 2);
                mapping_to_matrix(meta.pTo, xsize, ysize) = Eigen::MatrixX<floating_point_type>(mat);
            }
        }
    }
};

#ifdef UNFINISHED_CODE
    #include <array>
    #include <memory>
    #include <string>
    #include <iostream>

    constexpr int max_dim = 5;
    constexpr int flexible_dim_flag = 0;
    using TDim = int;
    using TDims = std::array<TDim, max_dim>;
    template<class TAlloc> constexpr inline int max_dim_byte_count()
    {
        constexpr int byte_aligin = TAlloc::byte_aligin;
        return (sizeof(TDim) * max_dim +  byte_aligin - 1) / byte_aligin;
    }  


    template<class T>
    struct STDAlloc : public std::allocator<T>{
        
        constexpr static int byte_align = 1;

    };


    template<class T, class TShape,int dim_index = 0>
    struct Ref
    {
        using shape_type = TShape;
        using next_ref_type = Ref<T, TShape, dim_index + 1>;
        union
        {
            struct{
                T* p;
                const TShape& s;
        
            };
            T n;
        };
        constexpr Ref(T* _p, const TShape& _s = TShape()) noexcept : p(_p), s(_s){}

        T& operator*() { return *p; }
        constexpr T& operator*() const { return *p; }
        operator T&(){ return *(*this); }
        operator const T&() const{ return *(*this); }
        constexpr Ref& operator = (T&& t){  *p = t;return *this; }
        constexpr bool index_check(int i){ return i < s.shape()[dim_index]; }
        constexpr next_ref_type operator[](int i)  __attribute__((always_inline))
        {
            if(!index_check(i))
            {
                printf("out of range\n");
            }
            return next_ref_type(p + (i * s.aligin_size(dim_index + 1)), s);
        }
        T* begin(){return p;}
        const T* begin() const{return p;}

        T* end(){return p + s.aligin_size(dim_index + 1);}
        const T* end() const{return p + s.aligin_size(dim_index + 1);}
    };

    template<int ..._dims>
    struct Shape
    {
        static constexpr std::array<int, max_dim>dims = { _dims..., 1 };
        
        constexpr static int aligin_size(int dim_index = 1)
        {
            int size = dims[dim_index];
            for(int i = dim_index + 1; i < max_dim; i++)
            {
                if(dims[i]) size *= dims[i];
            }
            return size;
        }
        constexpr static int size(){return aligin_size(0);}
        constexpr static std::array<TDim, max_dim> shape() 
        {
            return dims;
        }
    };


    template<class T>
    void print(T&& t){
        std::cout << t << std::endl;
    }
    template<class T, class ...TRest>
    void print(T&& t, TRest&& ...args){
        std::cout << t << " ";
        if constexpr( sizeof...(TRest))
        {
            print(std::forward<TRest>(args)...);
        }
    }
    template<class T, class TShape>
    constexpr T test(const int n[])
    {
        Ref<const int,TShape> ref(n);
        return  ref[0][0] + ref[0][1] + ref[1] + ref[1][0];
    }
    constexpr int n[] = {1, 2, 3, 4};
    constexpr void a()
    {
        printf("%d", test<const int,  Shape<2, 2>>(n));
    }
    template<class T, int x, int y>
    struct matrix : Ref<T, Shape<x, y>>
    {
        using base_type = Ref<T, Shape<x, y>>;
        std::array<T, base_type::shape_type::size()> p;

        // using base_type::base_type;

        matrix():base_type(p.data()){}
    };
#endif
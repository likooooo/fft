#include <fftw_extension.hpp>
#include <fftw_plan.hpp>
#include <eigen_wrapper.hpp>
#include <vector>
#include <typeinfo>

using plan_holder_f = plan_holder<float>;
using plan_holder_d = plan_holder<double>;
using vec_f = dynamic_vec<float>; 
using vec_d = dynamic_vec<double>; 
using vec_zf = dynamic_vec<std::complex<float>>; 
using vec_zd = dynamic_vec<std::complex<double>>; 
using e_vec_f = Eigen::VectorXf;
using e_vec_d = Eigen::VectorXd;
using e_vec_zf = Eigen::VectorXcf;
using e_vec_zd = Eigen::VectorXcd;

int mkl_test()
{
    MKLVersion mkl_version;
    mkl_get_version(&mkl_version);
    printf("\nYou are using oneMKL %d.%d\n", mkl_version.MajorVersion, mkl_version.UpdateVersion);
    return 0;
}
template<matrix_major m>
int dims_test(bool enable_print = true)
{
    printf("\n * dim test (%s)\n", m == matrix_major::row_major ? "row_major" : "col_major");
    std::cout.setstate(std::ios_base::goodbit);
    std::cout.setstate(enable_print ? std::ios_base::goodbit : std::ios_base::failbit);
    std::cout << "    input dim [1, 2, 3]\n";
    auto dim = fft_dim<m>::get(1, 2, 3);
    std::cout << "    dim args in arguments : " << std::span{dim} << "\n";
    
    auto dim1 = fft_dim<m, 1, 2, 3>::get();
    std::cout << "    dim args in template  : " << std::span{dim1} << "\n";
    return 0;
}
template<class vec, class ...T> int plan_test(T ...xsize)
{
    bool enable = true; if(!enable) return 0;
    using scalar = typename vec::value_type;
    using plan = plan_holder<scalar>;
    auto dims = fft_dim<matrix_major::col_major>::get(xsize...);
    std::stringstream dimstr;
    dimstr << std::span{dims};
    printf("\n * plan test (%s) dim : %s\n", typeid(scalar).name(), dimstr.str().c_str());
    
    auto n = std::accumulate(dims.begin() + 1, dims.end(), ((dims.front()/2 + 1)*2), [](auto a, auto b){return a * b;});
    vec in(n), out(n);
    auto p = plan::make_plan(in.data(), out.data(), xsize...);
    if(nullptr == p) throw std::invalid_argument("create plan failed");
    auto str = plan::sprint_plan(p.get());
    printf("    %s, %zd\n", str.get(), sizeof(p));
    return 0;
}
template<class scalar, class scalarTo = scalar>
int fft_operator_inplace_test()
{
    using namespace Eigen;
    int xsize = random_operator<int>()(1, 128);
    int ysize = random_operator<int>()(1, 128);
    printf("\n * [inplace] fft operator test (%s -> %s), shape : (%d, %d)\n", typeid(scalar).name(), typeid(scalarTo).name(), xsize, ysize);
    using plan = plan_holder<scalar, scalarTo>;
    using vec = dynamic_vec<scalar>;
    using matrix = MatrixX<scalar>;
    /**
     *  alloc memory
    */
    vec in;
    in.resize(xsize * ysize);
    auto stride = std::is_floating_point_v<scalar> ? (xsize / 2 + 1)*2 : xsize;
    in.reserve( ysize * stride );
    dynamic_vec<scalarTo>* out = reinterpret_cast<dynamic_vec<scalarTo>*>(&in);
    fft_domain current_domain;
    /**
     * make plan
    */
    auto pFFT = plan::make_plan(in.data(), out->data(), xsize, ysize); 
    {
        auto [pin, pout, domain, dim] = fft_operator::get_meta(pFFT);
        current_domain = domain;
        if(in.data() != pin || out->data() != pout || ysize != dim[0] || xsize != dim[1])
        {
            printf("ptr : (%p, %p), shape : (%d, %d), domain : %s, pPlan : %p\n", pin, pout, dim[0], dim[1], (current_domain == fft_domain::freq ? "freq" : "space"), pFFT.get());   
            printf("ptr : (%p, %p), shape : (%d, %d)\n", in.data(), out->data(), ysize, xsize);
            throw std::runtime_error("unexcept meta data");
        }
    }
    auto pIFFT = plan::make_inv_plan(pFFT);
    {
        auto [pin, pout, domain, dim] = fft_operator::get_meta(pIFFT);
        if(domain == current_domain)
        {
            printf("ptr : (%p, %p), shape : (%d, %d), domain : %s, pPlan : %p\n", pin, pout, dim[0], dim[1], (domain == fft_domain::freq ? "freq" : "space"), pIFFT.get());   
            throw std::runtime_error("unexcept meta data after fft");
        }
    }
    /**
     * update matrix
    */
    //Map<matrix, 0, Stride<Dynamic, 1>> matrix_in_out(in.data(), xsize, ysize, Stride<Dynamic, 1>(stride, 1));
    Map<matrix> matrix_in_out(in.data(), xsize, ysize);

    matrix_in_out.setRandom();
    matrix origin(matrix_in_out);
    /**
     * 1/N * IFFT(FFT(X))
    */
    fft_operator::transform(pFFT);
    fft_operator::transform(pIFFT);
    matrix_in_out /= (xsize * ysize);

    auto [diff, max_error] = cal_eps(origin, matrix_in_out);
    std::cout << "max error : " << max_error <<"\n";
    if( max_error > 9e-6 )
    {   
        std::cout << "* X            : \n" << origin <<"\n";
        std::cout <<"------------------\n";
        std::cout << "* IFFT(FFT(X)) : \n" << matrix_in_out <<"\n";
        std::cout <<"------------------\n";
        std::cout << "* abs(diff)     :\n" << diff <<"\n";
        throw std::runtime_error("epsion out of range");
    }
    return 0;
}

template<class scalar, class scalarTo = scalar>
int fft_operator_outplace_test()
{
    using namespace Eigen;
    int xsize = random_operator<int>()(1, 128);
    int ysize = random_operator<int>()(1, 128);
    printf("\n * [outplace] fft operator test (%s -> %s), shape : (%d, %d)\n", typeid(scalar).name(), typeid(scalarTo).name(), xsize, ysize);
    using plan = plan_holder<scalar, scalarTo>;
    using vec = dynamic_vec<scalar>;
    using matrix = MatrixX<scalar>;
    /**
     *  alloc memory
    */
    vec in;
    in.resize(xsize * ysize);
    auto stride = std::is_floating_point_v<scalar> ? (xsize / 2 + 1)*2 : xsize;
    dynamic_vec<scalarTo> out;
    out.resize( ysize * stride * sizeof(scalar) / sizeof(scalarTo));
    fft_domain current_domain;
    /**
     * make plan
    */
    auto pFFT = plan::make_plan(in.data(), out.data(), xsize, ysize); 
    {
        auto [pin, pout, domain, dim] = fft_operator::get_meta(pFFT);
        current_domain = domain;
        if(in.data() != pin || out.data() != pout || ysize != dim[0] || xsize != dim[1])
        {
            printf("ptr : (%p, %p), shape : (%d, %d), domain : %s, pPlan : %p\n", pin, pout, dim[0], dim[1], (current_domain == fft_domain::freq ? "freq" : "space"), pFFT.get());   
            printf("ptr : (%p, %p), shape : (%d, %d)\n", in.data(), out.data(), ysize, xsize);
            throw std::runtime_error("unexcept meta data");
        }
    }
    auto pIFFT = plan::make_inv_plan(pFFT);
    {
        auto [pin, pout, domain, dim] = fft_operator::get_meta(pIFFT);
        if(domain == current_domain)
        {
            printf("ptr : (%p, %p), shape : (%d, %d), domain : %s, pPlan : %p\n", pin, pout, dim[0], dim[1], (domain == fft_domain::freq ? "freq" : "space"), pIFFT.get());   
            throw std::runtime_error("unexcept meta data after fft");
        }
    }
    /**
     * update matrix
    */
    auto matrix_in_out = mapping_to_matrix(in.data(), xsize, ysize);
    matrix_in_out.setRandom();
    matrix origin(matrix_in_out);
    /**
     * 1/N * IFFT(FFT(X))
    */
    fft_operator::transform(pFFT);
    fft_operator::transform(pIFFT);
    matrix_in_out /= (xsize * ysize);

    matrix diff = origin - matrix_in_out;
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
    std::cout << "max error : " << max_error <<"\n";
    if( max_error > 9e-6 )
    {   
        std::cout << "* X            : \n" << origin <<"\n";
        std::cout <<"------------------\n";
        std::cout << "* IFFT(FFT(X)) : \n" << matrix_in_out <<"\n";
        std::cout <<"------------------\n";
        std::cout << "* abs(diff)     :\n" << diff <<"\n";
        throw std::runtime_error("epsion out of range");
    }
    return 0;
}

auto run_test()
{
    return std::array{
        dims_test<matrix_major::row_major>(),
        dims_test<matrix_major::col_major>(),

        plan_test<e_vec_f>(128),
        plan_test<vec_f>(128),
        plan_test<e_vec_d>(128),
        plan_test<vec_d>(128),
        plan_test<e_vec_zf>(128),
        plan_test<vec_zf>(128),
        plan_test<e_vec_zd>(128),
        plan_test<vec_zd>(128),

        fft_operator_inplace_test<float, std::complex<float>>(),
        fft_operator_inplace_test<std::complex<float>>(),
        fft_operator_inplace_test<double, std::complex<double>>(),
        fft_operator_inplace_test<std::complex<double>>(),

        fft_operator_outplace_test<float, std::complex<float>>(),
        fft_operator_outplace_test<std::complex<float>>(),
        fft_operator_outplace_test<double, std::complex<double>>(),
        fft_operator_outplace_test<std::complex<double>>(),

        0
    };
}

int main()
{
    run_test();
    int repeat_count = 100;
    for(int i = 0; i < repeat_count; i++)
    {
       auto n = {
            fft_operator_inplace_test<float, std::complex<float>>(),
            fft_operator_inplace_test<std::complex<float>>(),
            fft_operator_inplace_test<double, std::complex<double>>(),
            fft_operator_inplace_test<std::complex<double>>(),

            fft_operator_outplace_test<float, std::complex<float>>(),
            fft_operator_outplace_test<std::complex<float>>(),
            fft_operator_outplace_test<double, std::complex<double>>(),
            fft_operator_outplace_test<std::complex<double>>()
        };
    }
    mkl_test();
    std::cout <<"\n---------------------------\n   test end\n";
}


#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <fftw3.h>
#include <fftw_extension.hpp>
#include <vector>
#include <fftw_allocator.hpp>
#include <complex>
#include <fftw_plan.hpp>
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
    auto pFFT = plan::make_plan(in.data(), out->data(), ysize, xsize); 
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
    Map<matrix, 0, Stride<Dynamic, 1>> matrix_in_out(in.data(), xsize, ysize, Stride<Dynamic, 1>(stride, 1));
    matrix origin(matrix_in_out);
    origin.setRandom();
    matrix_in_out = origin; 
    /**
     * 1/N * IFFT(FFT(X))
    */
    plan::transform(pFFT);
    plan::transform(pIFFT);
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
    auto pFFT = plan::make_plan(in.data(), out.data(), ysize, xsize); 
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
    Map<matrix> matrix_in_out(in.data(), xsize, ysize);
    matrix origin(matrix_in_out);
    origin.setRandom();
    matrix_in_out = origin; 
    /**
     * 1/N * IFFT(FFT(X))
    */
    plan::transform(pFFT);
    plan::transform(pIFFT);
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

        fft_operator_inplace_test<std::complex<float>>(),
        fft_operator_inplace_test<float, std::complex<float>>(),
        fft_operator_inplace_test<std::complex<double>>(),
        fft_operator_inplace_test<double, std::complex<double>>(),

        fft_operator_outplace_test<std::complex<float>>(),
        fft_operator_outplace_test<float, std::complex<float>>(),
        fft_operator_outplace_test<std::complex<double>>(),
        fft_operator_outplace_test<double, std::complex<double>>(),

        0
    };
}
auto test_entries = run_test();


int main()
{
    int repeat_count = 100;
    for(int i = 0; i < repeat_count; i++)
    {
       auto n = {
            fft_operator_inplace_test<std::complex<float>>(),
            fft_operator_inplace_test<float, std::complex<float>>(),
            fft_operator_inplace_test<std::complex<double>>(),
            fft_operator_inplace_test<double, std::complex<double>>(),

            fft_operator_outplace_test<std::complex<float>>(),
            fft_operator_outplace_test<float, std::complex<float>>(),
            fft_operator_outplace_test<std::complex<double>>(),
            fft_operator_outplace_test<double, std::complex<double>>()
        };
    }
    mkl_test();
    std::cout <<"\n---------------------------\n   test end\n";
}











int example() {
    int N0 = 4, N1 = 8; // 例子中的维度
    double *in; // 输入数据（实数）
    fftw_complex *out; // 输出数据（复数）
    fftw_complex *in_complex; // 用于 c2r 变换的输入数据（复数）
    double *out_real; // 输出数据（实数）
    fftw_plan p_r2c, p_c2r;

    in = (double*) fftw_malloc(sizeof(double) * (N0 * N1));
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((N0/2+1) * N1));
    // in = (double*)out;
    // in_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((N0/2+1) * N1));
    in_complex = out;
    out_real = (double*) fftw_malloc(sizeof(double) * (N0 * N1));

    // 创建 r2c 计划
    p_r2c = fftw_plan_dft_r2c_2d(N0, N1, in, out, FFTW_ESTIMATE);

    // 创建 c2r 计划
    p_c2r = fftw_plan_dft_c2r_2d(N0, N1, in_complex, out_real, FFTW_ESTIMATE);

    // 填充输入数据（列主序）
    // 假设你的列主序数据数组是 `data`
    for (int i = 0; i < N0; ++i) {
        for (int j = 0; j < N1; ++j) {
            in[i * N1 + j] = random_operator< double>()(0.0+ 1e6, 1.0+ 1e6);
        }
    }
    // transpose<double>(in, N1, N0 + 2);

    // 执行 r2c 变换
    fftw_execute(p_r2c);
    // transpose<double>(out_real, N0 + 2, N1);

    // // 转换复数数组为用于 c2r 变换的复数数组
    // for (int i = 0; i < (N0/2+1) * N1; ++i) {
    //     in_complex[i][0] = out[i][0];
    //     in_complex[i][1] = out[i][1];
    // }

    // 执行 c2r 变换
    fftw_execute(p_c2r);

    Eigen::Map<Eigen::MatrixX<double>> a(in, N0, N1);
    Eigen::Map<Eigen::MatrixX<double>> b(out_real, N0, N1);
    b /= (N0 * N1);
    std::cout << (a - b) <<"\n";
    // 处理输出数据
    // 这里你需要根据实际情况来处理变换后的数据

    // 释放资源
    fftw_destroy_plan(p_r2c);
    fftw_destroy_plan(p_c2r);
    // fftw_free(in);
    fftw_free(out);
    // fftw_free(in_complex);
    fftw_free(out_real);

    return 0;
}
int example1() {
    // 假设x方向的大小为M，y方向的大小为N
    int M = 5; // x方向的大小
    int N = 3; // y方向的大小

    // 输入/输出缓冲区
    fftw_complex *in_out;
    fftw_plan plan_r2c, plan_c2r;

    // 分配内存（原地变换，因此只需要为复数数组分配空间）
    // 对于二维变换，复数数组的大小是 (M/2+1) * N
    in_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((M / 2 + 1) * N));

    // 创建二维R2C变换计划
    plan_r2c = fftw_plan_dft_r2c_2d(N, M, (double*)in_out, in_out, FFTW_ESTIMATE);

    // 创建二维C2R变换计划
    plan_c2r = fftw_plan_dft_c2r_2d(N, M, in_out, (double*)in_out, FFTW_ESTIMATE);

    memset(in_out, 0, sizeof(fftw_complex) * ((M / 2 + 1) * N));
    // in_out[0][0] = 1;
    in_out[1][0] = 1;


    // 执行R2C变换
    fftw_execute(plan_r2c);

    // 注意：此时in_out数组包含变换后的复数结果

    // 执行C2R变换
    fftw_execute(plan_c2r);

    // 输出结果，由于C2R变换的结果是实数，我们需要除以(M * N)来得到正确的结果
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            printf("%f ", ((double*)in_out)[j * M + i] / (M * N));
        }
        printf("\n");
    }

    // 释放内存和计划
    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(in_out);

    return 0;
}
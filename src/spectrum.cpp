#include <fftw_extension.hpp>
#include <fftw_plan.hpp>
#include <eigen_wrapper.hpp>
#include <vector>
#include <typeinfo>
#include <spectrum_analysis.hpp>
using namespace Eigen;
template <class T>
void rect_image(T *p, int width, int height, int rec_size)
{
    std::memset(p, 0, width * height * sizeof(T));
    for (int y = (height - rec_size) / 2; y < (height + rec_size) / 2; y++)
    {
        for (int x = (width - rec_size) / 2; x < (width + rec_size) / 2; x++)
        {
            p[width * y + x] = T(1);
        }
    }
}
template <class T, class Ttag>
void print_matrix(const T &matrix, const Ttag &tag)
{
    std::cout << tag << " :\n"
              << matrix << "\n-----------------------\n";
}

template <class T>
void flip_test()
{
    int min_rec_size = 2;
    int rec_size = random_operator<int>()(min_rec_size, min_rec_size * 2);

    int xsize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    int ysize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    xsize = xsize / 2 * 2;
    ysize = ysize / 2 * 2;
    dynamic_vec<T> vec(xsize * ysize);
    rect_image(vec.data(), xsize, ysize, rec_size);
    auto matrix = mapping_to_matrix(vec.data(), xsize, ysize);
    MatrixX<T> origin(matrix);
    CenterCornerFlip(vec.data(), xsize, ysize);
    CenterCornerFlip(vec.data(), xsize, ysize);
    if (origin != matrix)
    {
        std::cout << "xsize : " << xsize << ", ysize : " << ysize << ", rec_size : " << rec_size << "\n";
        print_matrix(origin, "origin");
        print_matrix(matrix, "flip twice");
    }
}
template <class T> void crop_test()
{
    int min_rec_size = 2;
    int rec_size = random_operator<int>()(min_rec_size, min_rec_size * 2);
    int xsize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    int ysize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    xsize = xsize / 2 * 2;
    ysize = ysize / 2 * 2;
    dynamic_vec<T> vec(xsize * ysize);
    rect_image(vec.data(), xsize, ysize, rec_size);
    auto origin = mapping_to_matrix(vec.data(), xsize, ysize);
    auto[vec_padded, padding] = make_vec<T>(xsize, ysize);
    int stride = xsize + padding;
    auto crop = mapping_to_matrix(vec_padded.data(), xsize, ysize, stride);
    int offset = (ysize - rec_size)/2 * xsize + (xsize - rec_size) / 2;
    CropImage<T>(
        vec_padded.data() + (ysize - rec_size) / 2 * stride + (xsize - rec_size) / 2, stride,
        vec.data()        + (ysize - rec_size) / 2 * xsize + (xsize - rec_size) / 2, xsize, rec_size, rec_size);
    auto [diff, max_error] = cal_eps(MatrixX<T>(origin), crop);
    printf("* crop test (%s)\n    max error %f\n", typeid(T).name(), (float)std::real(max_error));
    if(max_error != 0)
    {
        print_matrix(origin,"origin");
        print_matrix(crop,  "croped");
        print_matrix(diff,  "error_image");
    }
}
template <class T, class TTo = T>
void compare_filp_spectrum_with_origin()
{
    using plan = plan_holder<T, TTo>;
    printf("* compare_filp_spectrum_with_origin (%s)\n", typeid(T).name());
    int min_rec_size = random_operator<int>()(2, 6);
    int rec_size = random_operator<int>()(min_rec_size, min_rec_size * 2);

    int xsize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    int ysize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    xsize = xsize / 2 * 2;
    ysize = ysize / 2 * 2;
    /**
     * create input data
     */
    auto [vec, padding] = make_vec<T>(xsize, ysize);
    rect_image(vec.data(), xsize, ysize, rec_size);
    auto origin(vec);
    CenterCornerFlip(vec.data(), xsize, ysize);
    /**
     * X  = FFT(flip(X))
     * X* = FFT(X) * M (in center_corner_flip_with)
     */
    auto pFFT = plan::make_plan(vec.data(), (TTo *)vec.data(), xsize, ysize);
    auto pOriginFFT = plan::make_plan(origin.data(), (TTo*)origin.data(), xsize, ysize);
    fft_operator::transform(pFFT);
    fft_operator::transform(pOriginFFT);
    auto stride = (padding ? (xsize / 2 + 1) : xsize);
    auto a = mapping_to_matrix((TTo*)origin.data(), stride, ysize);
    auto b = mapping_to_matrix((TTo*)vec.data(), stride, ysize);
    center_corner_flip_with((TTo*)origin.data(), a.rows(), a.cols());
    /**
     * compare X with X*
     */
    auto [eps, max_error] = cal_eps(a, b);
    printf("    max error : %f\n", (float)max_error);
    if (max_error > 9e-6)
    {
        print_matrix(a, "origin ");
        print_matrix(b, "fliped ");
        print_matrix(eps, "eps ");
        std::array<char, 255> buf = {0};
        sprintf(buf.data(), "epsion out of range %f, matrix size :(%d, %d), rec_size : (%d, %d)", 
            (float)max_error, xsize, ysize, rec_size, rec_size
        );
        throw std::runtime_error(std::string(buf.data()));
    }
}

template<class T, class TTo = T>
void phase_modulate_test()
{
    using plan = plan_holder<T, TTo>;
    using float_type = typename plan::floating_point_type; 
    printf("* phase_modulate_test (%s)\n", typeid(T).name());
    int min_rec_size = 2;
    int rec_size = random_operator<int>()(min_rec_size, min_rec_size * 2);
    int xsize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    int ysize = random_operator<int>()(min_rec_size * 2, min_rec_size * 4);
    const float_type dx = std::min((xsize - rec_size + 1) / 2, 1);
    const float_type dy = std::min((ysize - rec_size + 1) / 2, 1);
    auto [vec, padding] = make_vec<T>(xsize, ysize);
    rect_image(vec.data(), xsize, ysize, rec_size);
    auto image = mapping_to_matrix(vec.data(), xsize, ysize);
    dynamic_vec<T> standard(vec.size());
    int offset = (ysize - rec_size) / 2 * xsize + (xsize - rec_size) / 2;
    CropImage(standard.data() + (offset + (int)dx + (int)dy * xsize), xsize,
        vec.data() + offset, xsize,
        rec_size, rec_size
    );
    auto pFFT  = plan::make_plan(vec.data(), (TTo*)vec.data(), xsize, ysize);
    auto pIFFT = plan::make_inv_plan(pFFT);
    fft_operator::transform(pFFT);
    if constexpr(plan::is_same_type)
        phase_modulate((TTo*)vec.data(), 
            xsize, xsize, ysize,
            dx, dy
        );
    else
       phase_modulate((TTo*)vec.data(), 
            xsize, (xsize / 2 + 1), ysize,
            dx, dy
        );
    fft_operator::transform(pIFFT);
    VecScala(vec.size(), T(1.0/(xsize * ysize)), vec.data());

    auto [error_image, max_error] = cal_eps(
        MatrixX<T>(image),
        mapping_to_matrix(standard.data(), xsize, ysize)
    );
    printf("    max error : %f\n", (float)max_error);
    if (max_error > 9e-6)
    {
        print_matrix(image, "phase shift ");
        print_matrix(mapping_to_matrix(standard.data(), xsize, ysize), "crop shift ");
        print_matrix(error_image, "error_image ");
        throw std::runtime_error("max error out of range");
    }
}

int main()
{
    for (int i = 0; i < 100; i++)
    {
        phase_modulate_test<float, std::complex<float>>();
        phase_modulate_test<double, std::complex<double>>();
        phase_modulate_test<std::complex<float>>();
        phase_modulate_test<std::complex<double>>();

        compare_filp_spectrum_with_origin< std::complex<float>>();
        compare_filp_spectrum_with_origin< std::complex<double>>();
#ifdef UNFINISHED_CODE
        compare_filp_spectrum_with_origin<float, std::complex<float>>();
        compare_filp_spectrum_with_origin<double, std::complex<double>>();
#endif
        flip_test<float>();
        flip_test<double>();
        flip_test<std::complex<double>>();
        flip_test<std::complex<float>>();
        crop_test<float>();
        crop_test<std::complex<float>>();
        crop_test<double>();
        crop_test<std::complex<double>>();
    }
    printf("test pass\n");
}
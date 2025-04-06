#include <conv.hpp>

template<class T>
void test()
{
    for(size_t i = 0; i < 10; i++)
    {
        std::vector<T> x(10); x.reserve(x.size() + 4);
        std::iota(x.begin(), x.end(), 1);
        std::vector<T> k(10); k.reserve(k.size() + 4);
        k.at(i) = 1;
        conv<T, complex_t<T>>(x.data(), k.data(), x.size() / 2, 2);
        std::cout << x << std::endl;
    }
}
int main()
{
    // test<std::complex<double>>();
    test<double>();
}
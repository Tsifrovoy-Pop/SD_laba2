#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace chrono;

using Complex = complex<double>;

void multiply_naive_ijk(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C, int n) {
    fill(C.begin(), C.end(), Complex(0, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Complex sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiply_optimized_ikj(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C, int n) {
    fill(C.begin(), C.end(), Complex(0, 0));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            Complex aik = A[i * n + k];
            const Complex* b_row = &B[k * n];
            Complex* c_row = &C[i * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

void multiply_blocked_omp(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C, int n) {
    fill(C.begin(), C.end(), Complex(0, 0));
    const int block_size = 64;

#pragma omp parallel for collapse(2) schedule(guided)
    for (int si = 0; si < n; si += block_size) {
        for (int sk = 0; sk < n; sk += block_size) {
            for (int sj = 0; sj < n; sj += block_size) {

                int i_end = min(si + block_size, n);
                int k_end = min(sk + block_size, n);
                int j_end = min(sj + block_size, n);

                for (int i = si; i < i_end; ++i) {
                    for (int k = sk; k < k_end; ++k) {
                        Complex aik = A[i * n + k];
                        int c_base = i * n;
                        int b_base = k * n;
                        for (int j = sj; j < j_end; ++j) {
                            C[c_base + j] += aik * B[b_base + j];
                        }
                    }
                }

            }
        }
    }
}

int main() {
    system("chcp 65001 > nul");
    setlocale(LC_ALL, "Russian");

    const int n = 4096;
    const long long ops = 2LL * n * n * n;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    auto gen_mat = [&]() {
        vector<Complex> mat(n * n);
        for (auto& val : mat) val = Complex(dis(gen), dis(gen));
        return mat;
        };

    auto A = gen_mat();
    auto B = gen_mat();
    vector<Complex> C1(n * n), C2(n * n), C3(n * n);

    cout << "Запуск Варианта 1... " << endl;
    auto start = high_resolution_clock::now();
    multiply_naive_ijk(A, B, C1, n);
    auto end = high_resolution_clock::now();
    double t1 = duration<double>(end - start).count();
    cout << "Время: " << t1 << " сек | " << (ops / t1) * 1e-6 << " MFLOPS\n" << endl;

    cout << "Запуск Варианта 2... " << endl;
    start = high_resolution_clock::now();
    multiply_optimized_ikj(A, B, C2, n);
    end = high_resolution_clock::now();
    double t2 = duration<double>(end - start).count();
    cout << "Время: " << t2 << " сек | " << (ops / t2) * 1e-6 << " MFLOPS\n" << endl;

    cout << "Запуск Варианта 3... " << endl;
    start = high_resolution_clock::now();
    multiply_blocked_omp(A, B, C3, n);
    end = high_resolution_clock::now();
    double t3 = duration<double>(end - start).count();
    cout << "Время: " << t3 << " сек | " << (ops / t3) * 1e-6 << " MFLOPS\n" << endl;

    cout << "=== Результаты ===" << endl;
    cout << "Вариант 1: " << (ops / t1) * 1e-6 << " MFLOPS" << endl;
    cout << "Вариант 2: " << (ops / t2) * 1e-6 << " MFLOPS" << endl;
    cout << "Вариант 3: " << (ops / t3) * 1e-6 << " MFLOPS" << endl;

    return 0;
}
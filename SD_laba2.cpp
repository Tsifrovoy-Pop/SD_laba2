extern "C" {
#include <cblas.h>
}

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

double matrix_diff_norm(const vector<Complex>& A, const vector<Complex>& B) {
    double sum = 0;
    for (size_t i = 0; i < A.size(); i++) {
        sum += norm(A[i] - B[i]);
    }
    return sqrt(sum);
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
    vector<Complex> C1(n * n), C2(n * n), C3(n * n), C4(n * n);

    cout << "Запуск Варианта 1 (naive ijk)...\n";
    auto start = high_resolution_clock::now();
    multiply_naive_ijk(A, B, C1, n);
    auto end = high_resolution_clock::now();
    double t1 = duration<double>(end - start).count();
    cout << "Время: " << t1 << " сек | " << (ops / t1) * 1e-6 << " MFLOPS\n\n";

    cout << "Запуск Варианта 2 (optimized ikj)...\n";
    start = high_resolution_clock::now();
    multiply_optimized_ikj(A, B, C2, n);
    end = high_resolution_clock::now();
    double t2 = duration<double>(end - start).count();
    cout << "Время: " << t2 << " сек | " << (ops / t2) * 1e-6 << " MFLOPS\n\n";

    cout << "Запуск Варианта 3 (blocked OMP)...\n";
    start = high_resolution_clock::now();
    multiply_blocked_omp(A, B, C3, n);
    end = high_resolution_clock::now();
    double t3 = duration<double>(end - start).count();
    cout << "Время: " << t3 << " сек | " << (ops / t3) * 1e-6 << " MFLOPS\n\n";
    cout << "Запуск OpenBLAS (cblas_zgemm)...\n";
    Complex alpha = 1.0;
    Complex beta = 0.0;

    start = high_resolution_clock::now();
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n,
        &alpha,
        A.data(), n,
        B.data(), n,
        &beta,
        C4.data(), n
    );
    end = high_resolution_clock::now();
    double t4 = duration<double>(end - start).count();
    cout << "Время: " << t4 << " сек | " << (ops / t4) * 1e-6 << " MFLOPS\n\n";
    cout << "=== Сравнение ускорений ===\n";
    auto speedup = [&](double t) { return (t / t4) * 100.0; };
    cout << "Наивный ijk: " << speedup(t1) << " % от скорости OpenBLAS\n";
    cout << "Оптимизированный ikj: " << speedup(t2) << " % от скорости OpenBLAS\n";
    cout << "Блочный OMP: " << speedup(t3) << " % от скорости OpenBLAS\n\n";
    cout << "=== Проверка корректности ===\n";
    cout << "||C1 - C4|| = " << matrix_diff_norm(C1, C4) << "\n";
    cout << "||C2 - C4|| = " << matrix_diff_norm(C2, C4) << "\n";
    cout << "||C3 - C4|| = " << matrix_diff_norm(C3, C4) << "\n";
    cout << "Эффективность Варианта 3 относительно Варианта 2: " << (t2 / t3) * 100.0 << "%" << endl;

    return 0;
}

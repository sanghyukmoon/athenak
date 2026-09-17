#ifndef UTILS_BENCHMARK_TIMER_HPP_
#define UTILS_BENCHMARK_TIMER_HPP_

#include <sys/resource.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// Opt-in benchmark sampling only. Synchronization and diagnostics are outside samples.
class BenchmarkTimer {
 public:
  struct Sample { std::string kind; double seconds; };
  BenchmarkTimer(int repeats, double warmup_seconds)
      : repeats_(repeats), warmup_seconds_(warmup_seconds) {
    if (repeats < 1 || warmup_seconds < 0) Kokkos::abort("invalid benchmark sampling");
    Kokkos::fence();
#ifdef KOKKOS_ENABLE_CUDA
    if (cudaMemGetInfo(&free_before_, &total_before_) != cudaSuccess) {
      Kokkos::abort("cannot query initialized device memory");
    }
#endif
  }
  void Start() {
    Kokkos::fence();
#if MPI_PARALLEL_ENABLED
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    clock_.reset();
  }
  void Stop() {
    Kokkos::fence();
    const double seconds = clock_.seconds();
    double global_seconds = seconds;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&seconds, &global_seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    const std::string kind = samples.empty() ? "first" :
        (warmup_elapsed_ < warmup_seconds_ ? "warmup" : "measured");
    samples.push_back({kind, seconds});
    if (kind == "warmup") warmup_elapsed_ += global_seconds;
    if (kind == "measured") ++measured_;
  }
  bool Done() const { return measured_ == repeats_; }
  void Write(const std::string &prefix, std::size_t result_bytes=0) const {
    const auto rank = std::to_string(global_variable::my_rank);
    std::ofstream out(prefix+"_rank"+rank+".csv");
    out << std::setprecision(17) << "sample,kind,wall_total\n";
    for (std::size_t i=0; i<samples.size(); ++i) {
      out << i << ',' << samples[i].kind << ',' << samples[i].seconds << '\n';
    }
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::ofstream memory("memory_rank"+rank+".txt");
    memory << "peak_rss_kib " << usage.ru_maxrss << '\n'
           << "result_array_bytes " << result_bytes << '\n';
#ifdef KOKKOS_ENABLE_CUDA
    std::size_t free_after, total_after;
    if (cudaMemGetInfo(&free_after, &total_after) != cudaSuccess) {
      Kokkos::abort("cannot query device memory after benchmark");
    }
    memory << "device_free_before_bytes " << free_before_ << '\n'
           << "device_free_after_bytes " << free_after << '\n'
           << "device_used_before_bytes " << total_before_-free_before_ << '\n'
           << "device_used_after_bytes " << total_after-free_after << '\n'
           << "device_total_bytes " << total_after << '\n';
#endif
    if (!out || !memory) Kokkos::abort("cannot write benchmark samples");
  }
  std::vector<Sample> samples;

 private:
  int repeats_, measured_ = 0;
  double warmup_seconds_, warmup_elapsed_ = 0;
  Kokkos::Timer clock_;
#ifdef KOKKOS_ENABLE_CUDA
  std::size_t free_before_, total_before_;
#endif
};
#endif  // UTILS_BENCHMARK_TIMER_HPP_

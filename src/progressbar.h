#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

class Timer
{
private:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::duration<double, std::ratio<1>> second;

    std::chrono::time_point<clock> start_time;
    double accumulated_time;
    bool running;

public:
    Timer()
    {
        accumulated_time = 0;
        running = false;
    }

    void start()
    {
        if (running)
            throw std::runtime_error("Timer was already started!");
        running = true;
        start_time = clock::now();
    }

    double stop()
    {
        if (!running)
            throw std::runtime_error("Timer was already stopped!");

        accumulated_time += lap();
        running = false;

        return accumulated_time;
    }

    double accumulated()
    {
        if (running)
            throw std::runtime_error("Timer is still running!");
        return accumulated_time;
    }

    double lap()
    {
        if (!running)
            throw std::runtime_error("Timer was not started!");
        return std::chrono::duration_cast<second>(clock::now() - start_time).count();
    }

    void reset()
    {
        accumulated_time = 0;
        running = false;
    }
};

class ProgressBar
{
private:
    uint32_t total_work;
    uint32_t next_update;
    uint32_t call_diff;
    uint32_t work_done;
    uint16_t old_percent;
    Timer timer;
    std::string prefix;

    void clearConsoleLine() const
    {
        std::cerr << "\r\033[2K" << std::flush;
    }

public:
    void start(uint32_t total_work, std::string prefix = "")
    {
        timer = Timer();
        timer.start();
        this->total_work = total_work;
        next_update = 0;
        call_diff = total_work / 200;
        old_percent = 0;
        work_done = 0;
        clearConsoleLine();
        this->prefix = prefix;
    }

    void update(uint32_t work_done0)
    {

#ifdef NOPROGRESS
        return;
#endif

        if (omp_get_thread_num() != 0)
            return;

        work_done = work_done0;

        if (work_done < next_update)
            return;

        next_update += call_diff;

        uint16_t percent = (uint8_t)(work_done * omp_get_num_threads() * 100 / total_work);

        if (percent > 100)
            percent = 100;

        if (percent == old_percent)
            return;

        old_percent = percent;

        // Print an update string which looks like this:
        //   [================================================  ] (96% - 1.0s - 4 threads)
        std::cerr << "\r\033[2K" << prefix << "["
                  << std::string(percent / 2, '=') << std::string(50 - percent / 2, ' ')
                  << "] ("
                  << percent << "% - ETA: "
                  << std::fixed << std::setprecision(1) << timer.lap() / percent * (100 - percent)
                  << "s - "
                  << omp_get_num_threads() << " threads)" << std::flush;
    }

    /// Increment by one the work done and update the progress bar
    ProgressBar &operator++()
    {
        if (omp_get_thread_num() != 0)
            return *this;

        work_done++;
        update(work_done);
        return *this;
    }

    double stop()
    {
        clearConsoleLine();

        timer.stop();
        return timer.accumulated();
    }

    double time_it_took()
    {
        return timer.accumulated();
    }
};
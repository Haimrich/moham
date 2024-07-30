#ifndef MOHAM_NSGA_H_
#define MOHAM_NSGA_H_

#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <cmath>

#define ENERGY_OBJ 0
#define LATENCY_OBJ 1
#define AREA_OBJ 2

namespace moham 
{
    enum Dominance
    {
        DOMINATING,
        DOMINATED,
        FRONTIER
    };

    class NSGA 
    {
    protected:

        template <std::size_t N>
        struct Individual {
            unsigned rank;
            double crowding_distance;
            std::array<double, N> objectives; // energy, latency, area
            bool valid;
        };

        // https://hal.inria.fr/hal-01909120/document
        std::vector<double> gen_stability_measures_;
        double stability_;

        bool single_obj_ = false, prod_obj_;
        std::vector<double> weights_obj_; // energy, latency, area

    protected:

        template <typename P>
        void Merging(P& merged_pop, const P& parent_pop, const P& offspring_pop, const P& immigrant_pop)
        {
            merged_pop.resize(parent_pop.size() + immigrant_pop.size());
            merged_pop.reserve(parent_pop.size() + immigrant_pop.size() + offspring_pop.size());
            
            std::copy(
                parent_pop.begin(),
                parent_pop.end(),
                merged_pop.begin());
            std::copy(
                immigrant_pop.begin(),
                immigrant_pop.end(),
                merged_pop.begin() + parent_pop.size());
            std::copy_if(
                offspring_pop.begin(),
                offspring_pop.end(),
                std::back_inserter(merged_pop),
                [](auto& ind) { return ind.valid; });
        }

        template <typename P, typename URNG>
        void Survival(P& new_pop, const P& source_pop, const bool shuffle, URNG& rng)
        {
            // Sort by rank and crowding distance and select population_size_
            std::partial_sort_copy(
                source_pop.begin(), source_pop.end(),
                new_pop.begin(), new_pop.end(),
                [&](const auto &a, const auto &b) -> bool
                {
                    return a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance);
                });

            // Shuffle
            if (shuffle) std::shuffle(std::begin(new_pop), std::end(new_pop), rng);
        }

        template <std::size_t N>
        double CalculateSingleObj(const Individual<N> &ind) {
            double tot_obj = prod_obj_ ? 1 : 0;
            for (unsigned i = 0; i < std::tuple_size<decltype(ind.objectives)>::value; i++)
            {
                double weight = i < weights_obj_.size() ? weights_obj_[i] : 1;
                if (prod_obj_) {
                    tot_obj *= std::pow(weight, ind.objectives[i]);
                } else {
                    tot_obj += weight * ind.objectives[i];
                }
            }
            return tot_obj;
        }

        template <std::size_t N>
        Dominance CheckDominance(const Individual<N> &a, const Individual<N> &b)
        {
            if (!single_obj_) {
                bool all_a_less_or_equal_than_b = true;
                bool any_a_less_than_b = false;
                bool all_b_less_or_equal_than_a = true;
                bool any_b_less_than_a = false;

                for (unsigned i = 0; i < N; i++)
                {
                    if (a.objectives[i] > b.objectives[i])
                    {
                        all_a_less_or_equal_than_b = false;
                        any_b_less_than_a = true;
                    }
                    else if (b.objectives[i] > a.objectives[i])
                    {
                        any_a_less_than_b = true;
                        all_b_less_or_equal_than_a = false;
                    }
                }

                if (all_a_less_or_equal_than_b && any_a_less_than_b)
                    return Dominance::DOMINATING;
                if (all_b_less_or_equal_than_a && any_b_less_than_a)
                    return Dominance::DOMINATED;
            } 
            else 
            {
                double obj_a = CalculateSingleObj(a);
                double obj_b = CalculateSingleObj(b);

                if (obj_a < obj_b) return Dominance::DOMINATING;
                if (obj_a > obj_b) return Dominance::DOMINATED; 
            }
            return Dominance::FRONTIER;
        }

        template <typename P>
        void AssignCrowdingDistance(P &population, std::vector<uint64_t> &pareto_front)
        {
            for (auto p : pareto_front)
                // population[p].crowding_distance = -std::accumulate(population[p].objectives.begin(), population[p].objectives.end(), 1, std::multiplies<double>());
                population[p].crowding_distance = 0.0;

            if (!single_obj_) {
                for (unsigned i = 0; i < std::tuple_size<decltype(population[0].objectives)>::value; i++)
                {
                    std::sort(pareto_front.begin(), pareto_front.end(), 
                    [&](const uint64_t a, const uint64_t b) -> bool
                        { return population[a].objectives[i] < population[b].objectives[i]; });

                    population[pareto_front.front()].crowding_distance = 10e14;
                    population[pareto_front.back()].crowding_distance = 10e14;

                    double range = population[pareto_front.back()].objectives[i] - population[pareto_front.front()].objectives[i];
                    assert(range >= 0);
                    range = (range == 0.0) ? 1.0 : range;

                    for (uint64_t j = 1; j < pareto_front.size() - 1; j++)
                    {
                        uint64_t r_prev = pareto_front[j - 1];
                        uint64_t r_next = pareto_front[j + 1];
                        uint64_t r_this = pareto_front[j];
                        population[r_this].crowding_distance += std::abs(population[r_next].objectives[i] - population[r_prev].objectives[i]) / range;
                    }
                }
            } else {
                std::sort(pareto_front.begin(), pareto_front.end(), 
                    [&](const uint64_t a, const uint64_t b) -> bool
                        { return CalculateSingleObj(population[a]) < CalculateSingleObj(population[b]); });

                    population[pareto_front.front()].crowding_distance = 10e14;
                    population[pareto_front.back()].crowding_distance = 10e14;

                    double range = CalculateSingleObj(population[pareto_front.back()]) - CalculateSingleObj(population[pareto_front.front()]);
                    assert(range >= 0);
                    range = (range == 0.0) ? 1.0 : range;

                    for (uint64_t j = 1; j < pareto_front.size() - 1; j++)
                    {
                        uint64_t r_prev = pareto_front[j - 1];
                        uint64_t r_next = pareto_front[j + 1];
                        uint64_t r_this = pareto_front[j];
                        population[r_this].crowding_distance += std::abs(CalculateSingleObj(population[r_next]) - CalculateSingleObj(population[r_prev])) / range;
                    }
            }
        }
        
        template <typename P>
        void AssignRankAndCrowdingDistance(P &population)
        {
            std::vector<uint64_t> pareto_front;
            std::vector<std::vector<uint64_t>> dominated_by(population.size(), std::vector<uint64_t>());
            std::vector<uint64_t> num_dominating(population.size(), 0);

            for (uint64_t i = 0; i < population.size(); i++)
            {
                for (uint64_t j = i + 1; j < population.size(); j++)
                {
                    switch (CheckDominance(population[i], population[j]))
                    {
                    case Dominance::DOMINATING:
                        dominated_by[i].push_back(j);
                        num_dominating[j]++;
                        break;
                    case Dominance::DOMINATED:
                        dominated_by[j].push_back(i);
                        num_dominating[i]++;
                        break;
                    case Dominance::FRONTIER:
                        break;
                    }
                }

                if (num_dominating[i] == 0)
                {
                    population[i].rank = 0;
                    pareto_front.push_back(i);
                }
            }

            uint64_t total_debug = 0;
            for (uint64_t f = 0; !pareto_front.empty(); f++)
            {
                total_debug += pareto_front.size();

                AssignCrowdingDistance(population, pareto_front);

                std::vector<uint64_t> new_pareto_front;

                for (uint64_t p : pareto_front)
                {

                    for (uint64_t q : dominated_by[p])
                    {

                        num_dominating[q]--;

                        if (num_dominating[q] == 0)
                        {
                            population[q].rank = f + 1;
                            new_pareto_front.push_back(q);
                        }
                    }
                }
                pareto_front = new_pareto_front;
            }

            assert(total_debug == population.size());
        }

        template <typename P>
        std::string RankString(P& pop) {
            std::stringstream ss;

            unsigned ranked =  0;
            for (unsigned i = 0; i < 4; i++)
            {
                unsigned i_ranked = std::count_if(pop.begin(), pop.end(), [i](const auto& ind){ return ind.rank == i; });
                ss << i << ":" << std::left << std::setw(4) << i_ranked;
                ranked += i_ranked;
            }

            ss << "+:" << std::left << std::setw(4) << pop.size() - ranked;
            
            /*
            for (auto &ind : pop)
            {
                if (ind.rank < 10)
                    ss << ind.rank;
                else if (ind.rank < 36)
                    ss << (char)(ind.rank + 55);
                else if (ind.rank < 61)
                    ss << (char)(ind.rank + 61);
                else
                    ss << "+";
            }
            */
            
            return ss.str();
        }


        template <typename P>
        void UpdateStabilityMeasure(P &pop, unsigned l, std::size_t stability_window) {
            double dl = -1.0;

            for (auto &ind : pop)
                if (ind.crowding_distance > dl)
                    dl = ind.crowding_distance;
            
            gen_stability_measures_[l % stability_window] = dl;
        }


        double CalculateGenerationStability(std::size_t L)
        {
            double dmean = std::accumulate(gen_stability_measures_.begin(), gen_stability_measures_.end(), 0.0) / L;

            return std::sqrt(std::accumulate(
                    gen_stability_measures_.begin(), 
                    gen_stability_measures_.end(),
                    0.0, 
                    [dmean](double psum, double d) {
                        return psum + std::pow(d - dmean, 2);
                    }
                ) / L);
        } 
    };
}

#endif // MOHAM_NSGA_H_
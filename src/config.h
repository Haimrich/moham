#ifndef MOHAM_CONFIG_H_
#define MOHAM_CONFIG_H_

namespace moham {

    struct Config 
    {
        struct Medea 
        {
            unsigned num_generations            = 1;
            unsigned population_size            = 100;
            unsigned immigrant_population_size  = 0;
            unsigned num_threads                = 0;

            double fill_mutation_prob           = 0.5;
            double parallel_mutation_prob       = 0.5;
            double random_mutation_prob         = 0.5;
            double crossover_prob               = 0.5;

            bool use_tournament                 = false;
            bool update_ert                     = false;
            bool random_when_illegal            = true;
            unsigned buffer_update_granularity  = 16;

            bool stop_on_convergence            = true;
            unsigned stability_window           = 5;
            double stability_threshold          = 0.04;

        } medea;

        struct Moham 
        {
            unsigned num_generations            = 1;
            unsigned population_size            = 100;
            unsigned immigrant_population_size  = 0;
            unsigned num_threads                = 0;

            double priority_crossover_prob      = 0.3;
            double subacc_crossover_prob        = 0.3;
            double mapping_crossover_prob       = 0.3;

            double splitting_mutation_prob      = 0.15;
            double merging_mutation_prob        = 0.15;
            double priority_mutation_prob       = 0.15;
            double mapping_mutation_prob        = 0.15;
            double template_mutation_prob       = 0.10;
            double assignment_mutation_prob     = 0.15;
            double position_mutation_prob       = 0.15;

            bool xu_priority                    = true;
            bool use_tournament                 = false;
            bool random_when_illegal            = true;
            
            bool stop_on_convergence            = true;
            unsigned stability_window           = 5;
            double stability_threshold          = 0.04;

            unsigned max_per_workload_mappings  = 0;

            unsigned max_subaccelerators        = 0;
            double system_bandwidth             = -1.0;
            double nip_link_bandwidth           = -1.0;
            double nip_hop_energy               = -1.0;

            std::vector<std::pair<unsigned, unsigned>> memory_interfaces_amount;
            unsigned max_memory_interfaces_amount = 4;
            std::string memory_interfaces_position = "corner";

            bool explore_mapping                    = true;
            bool negotiate_arch                     = true;

            bool random_search                      = false;

            bool single_obj                         = false;
            bool prod_obj                           = false;
            std::vector<double> weights_obj;
        } moham;

        std::string out_dir                     = ".";
        unsigned long seed                      = 0;
    };

}


#endif // MOHAM_CONFIG_H_
#ifndef MOHAM_MAPPING_H_
#define MOHAM_MAPPING_H_

#include <vector>

#include "mapping/mapping.hpp"

namespace moham {

    class Mapping : public ::Mapping
    {
    public:
        typedef std::size_t ID;

        Mapping &operator=(const ::Mapping &m)
        {
            id = m.id;
            loop_nest = m.loop_nest;
            complete_loop_nest = m.complete_loop_nest;
            datatype_bypass_nest = m.datatype_bypass_nest;
            confidence_thresholds = m.confidence_thresholds;
            fanoutX_map = m.fanoutX_map;
            fanoutY_map = m.fanoutY_map;

            return *this;
        }

    };

}


#endif // MOHAM_MAPPING_H_
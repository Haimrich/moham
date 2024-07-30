# Copyright (c) 2019 Yannan Wu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from accelergy.raw_inputs_2_dicts import RawInputs2Dicts
from accelergy.system_state import SystemState
from accelergy.component_class import ComponentClass
from accelergy.arch_dict_2_obj import arch_dict_2_obj
from accelergy.plug_in_path_to_obj import plug_in_path_to_obj
from accelergy.primitive_component import PrimitiveComponent
from accelergy.compound_component import CompoundComponent
from accelergy.ART_generator import AreaReferenceTableGenerator
from accelergy.ERT_generator import EnergyReferenceTableGenerator
from accelergy.utils import *

import argparse, sys
from collections import OrderedDict
from yaml import dump

def update_depth(arch, buffer_name, new_depth):
    if isinstance(arch, list):
        for k,v in enumerate(arch):
            if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, list):
                arch[k] = update_depth(v, buffer_name, new_depth)
        return arch
            
    for k, v in arch.items():
        if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, list):
            arch[k] = update_depth(v, buffer_name, new_depth)
        elif k == 'name' and buffer_name in v:
            arch['attributes']['depth'] = new_depth
            return arch
        
    return arch

def buffer_depth_list(string):
    sp = string.split(",")
    return (sp[0], int(sp[1]))

def main():
    sys.stdout = open(os.devnull, 'w')
    accelergy_version = 0.3

    # ----- Interpret Commandline Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--oprefix', type =str, default = '')
    parser.add_argument('--updates', nargs='*', default=[], type=buffer_depth_list)
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()
    output_prefix = args.oprefix
    path_arglist = args.files
    precision = 5
    
    compute_ART = 1

    # ----- Global Storage of System Info
    system_state = SystemState()
    system_state.set_accelergy_version(accelergy_version)

    # ----- Load Raw Inputs to Parse into Dicts
    raw_input_info = {'path_arglist': path_arglist, 'parser_version': accelergy_version}
    raw_dicts = RawInputs2Dicts(raw_input_info)
    
    for buffer, depth in args.updates:
        raw_dicts.hier_arch_spec_dict = update_depth(raw_dicts.hier_arch_spec_dict, buffer, depth)

    # ----- Determine what operations should be performed
    available_inputs = raw_dicts.get_available_inputs()

    # ---- Detecting config only cases and gracefully exiting
    if len(available_inputs) == 0:
        INFO("no input is provided, exiting...")
        sys.exit(0)

    if compute_ART not in available_inputs:
        # ----- Interpret the input architecture description using only the input information (w/o class definitions)
        system_state.set_hier_arch_spec(raw_dicts.get_hier_arch_spec_dict())

    # ----- Add the Component Classes
    for pc_name, pc_info in raw_dicts.get_pc_classses().items():
        system_state.add_pc_class(ComponentClass(pc_info))
    for cc_name, cc_info in raw_dicts.get_cc_classses().items():
        system_state.add_cc_class(ComponentClass(cc_info))

    # ----- Set Architecture Spec (all attributes defined)
    arch_obj = arch_dict_2_obj(raw_dicts.get_flatten_arch_spec_dict(), system_state.cc_classes, system_state.pc_classes)
    system_state.set_arch_spec(arch_obj)

    # ERT/ERT_summary/energy estimates/ART/ART summary need to be generated without provided ERT
    #        ----> all components need to be defined
    # ----- Add the Fully Defined Components (all flattened out)

    for arch_component in system_state.arch_spec:
        if arch_component.get_class_name() in system_state.pc_classes:
            class_name = arch_component.get_class_name()
            pc = PrimitiveComponent({'component': arch_component, 'pc_class': system_state.pc_classes[class_name]})
            system_state.add_pc(pc)
        elif arch_component.get_class_name() in system_state.cc_classes:
            cc = CompoundComponent({'component': arch_component, 'pc_classes':system_state.pc_classes, 'cc_classes':system_state.cc_classes})
            system_state.add_cc(cc)
        else:
            ERROR_CLEAN_EXIT('Cannot find class name %s specified in architecture'%arch_component.get_class())

    # ----- Add all available plug-ins
    system_state.add_plug_ins(plug_in_path_to_obj(raw_dicts.get_estimation_plug_in_paths(), output_prefix))


    # ----- Generate Energy Reference Table
    ert_gen = EnergyReferenceTableGenerator({'parser_version': accelergy_version,
                                                'pcs': system_state.pcs,
                                                'ccs': system_state.ccs,
                                                'plug_ins': system_state.plug_ins,
                                                'precision': precision})

    # ----- Generate Area Reference Table
    art_gen = AreaReferenceTableGenerator({'parser_version': accelergy_version,
                                            'pcs': system_state.pcs,
                                            'ccs': system_state.ccs,
                                            'plug_ins': system_state.plug_ins,
                                            'precision': precision})

    # ----- Output Reference Tables to stdout
    sys.stdout = sys.__stdout__
    print(dump(art_gen.get_ART().get_ART(), default_flow_style= False, Dumper= accelergy_dumper))
    print(dump(ert_gen.get_ERT().get_ERT(), default_flow_style= False, Dumper= accelergy_dumper))

if __name__ == "__main__":
    main()

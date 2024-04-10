"""
End Effector Posiitons
"""
def gen_end_effector_positions_inner_temp_mem_size(self):
    num_ees = self.robot.get_total_leaf_nodes()
    return 2*16*num_ees

def gen_end_effector_positions_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_Xhom_name = "s_XmatsHom", \
        s_eePos_name = "s_eePos", \
        s_q_name = "s_q", \
        s_topology_helpers_name = "s_topology_helpers", \
        s_temp_name = "s_temp", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    code_start = "end_effector_positions_inner<T>(" + var_names["s_eePos_name"] + ", " + var_names["s_q_name"] + ", "
    code_middle = var_names["s_Xhom_name"] + ", "
    code_end =  var_names["s_temp_name"] + ");"
    # account for thread group and serial chains
    if use_thread_group:
        code_start = code_start.replace("(","(tgrp, ")
    if not self.robot.is_serial_chain():
        code_middle += var_names["s_topology_helpers_name"] + ", "
    self.gen_add_code_line(code_start + code_middle + code_end)

def gen_end_effector_positions_inner(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
    all_ees = self.robot.get_leaf_nodes()
    num_ees = len(all_ees)
    # construct the boilerplate and function definition
    func_params = ["s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = " + str(num_ees), \
                   "s_q is the vector of joint positions", \
                   "s_Xhom is the pointer to the homogenous transformation matricies ", \
                   "s_temp is a pointer to helper shared memory of size " + \
                            str(self.gen_end_effector_positions_inner_temp_mem_size())]
    func_notes = ["Assumes the Xhom matricies have already been updated for the given q"]
    func_def_start = "void end_effector_positions_inner("
    func_def_middle = "T *s_eePos, const T *s_q, const T *s_Xhom, "
    func_def_end = "T *s_temp) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -1, NO_XI_FLAG = True)
    func_def = func_def_start + func_def_middle + func_def_end
    # now generate the code
    self.gen_add_func_doc("Computes the End Effector Position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    #
    # Initial Debug Prints if Requested
    #
    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,4,4>(&s_Xhom[16*i],4);}")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # For each chain we need to (in parallel) multiply the Xmats
    # 
    self.gen_add_code_line("//")
    self.gen_add_code_line("// For each branch in parallel chain up the transform")
    self.gen_add_code_line("// Keep chaining until reaching the root (starting from the leaves)")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels): # at most bfs levels of parents to chain
        # if serial chain manipulator then this is easy
        if self.robot.is_serial_chain():
            self.gen_add_code_line("// Serial chain manipulator so optimize as parent is jid-1")
            if bfs_level == 0:
                self.gen_add_code_line("// First set to leaf transform")
                self.gen_add_parallel_loop("ind",str(16),use_thread_group)
                self.gen_add_code_line("s_temp[ind] = s_Xhom[16*" + str(all_ees[0]) + " + ind];")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
            else:
                self.gen_add_code_line("// Update with parent transform until you reach the base [level " + str(bfs_level) + "/" + str(n_bfs_levels-1) + "]")
                self.gen_add_parallel_loop("ind",str(16),use_thread_group)
                self.gen_add_code_line("int row = ind % 4; int col = ind / 4;")
                # get the parents we need at this level working backwards from all_ees
                parent = all_ees[0]
                for i in range(bfs_level):
                    parent = self.robot.get_parent_id(parent)
                # need to swap dst and start each time
                even = bfs_level % 2
                tempDstOffset = 16*(even)
                tempSrcOffset = 16*(not even)
                self.gen_add_code_line("s_temp[ind + " + str(tempDstOffset) + "] = dot_prod<T,4,4,1>" + \
                                       "(&s_Xhom[16*" + str(parent) + " + row], &s_temp[" + str(tempSrcOffset) + " + 4*col]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
        else:
            # if first loop then just set to transform at the leaf
            if bfs_level == 0:
                self.gen_add_code_line("// First set to leaf transform")
                self.gen_add_parallel_loop("ind",str(16*num_ees),use_thread_group)
                self.gen_add_code_line("int rc = ind % 16;")
                select_var_vals = [("int", "eeInd", [str(jid) for jid in all_ees])]
                self.gen_add_multi_threaded_select("ind", "<", [str(16*(i+1)) for i in range(num_ees)], select_var_vals)
                self.gen_add_code_line("s_temp[ind] = s_Xhom[16*eeInd + rc];")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
            else:
                self.gen_add_code_line("// Update with parent transform until you reach the base [level " + str(bfs_level) + "/" + str(n_bfs_levels-1) + "]")
                self.gen_add_parallel_loop("ind",str(16*num_ees),use_thread_group)
                self.gen_add_code_line("int row = ind % 4; int col = (ind / 4) % 4; int eeOffset = ind - (ind % 16);")
                # get the parents we need at this level working backwards from all_ees
                curr_parents = all_ees
                for i in range(bfs_level):
                    curr_parents = [(-1 if jid == -1 else self.robot.get_parent_id(jid)) for jid in curr_parents]
                # need to swap dst and start each time
                even = bfs_level % 2
                tempDstOffset = 16*num_ees*(even)
                tempSrcOffset = 16*num_ees*(not even)
                # get parents for this level
                select_var_vals = [("int", "parent_jid", [str(jid) for jid in curr_parents])]
                self.gen_add_multi_threaded_select("ind", "<", [str(16*(i+1)) for i in range(num_ees)], select_var_vals)
                if (-1 in curr_parents):
                    self.gen_add_code_line("if(parent_jid == -1){continue;}")
                self.gen_add_code_line("s_temp[ind + " + str(tempDstOffset) + "] = dot_prod<T,4,4,1>" + \
                                       "(&s_Xhom[16*parent_jid + row], &s_temp[" + str(tempSrcOffset) + " + eeOffset + 4*col]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
    
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Now extract the eePos from the Tansforms")
    self.gen_add_code_line("// TODO: ADD OFFSETS")
    self.gen_add_code_line("//")
    tempOffset = 16*num_ees*(bfs_level % 2)
    # xyz position is easy (eePos_xyz1 = Xmat_hom * offset) where offset = [x,y,z,1]
    self.gen_add_parallel_loop("ind",str(3*num_ees),use_thread_group)
    self.gen_add_code_line("// xyz is easy")
    self.gen_add_code_line("int xyzInd = ind % 3; int eeInd = ind / 3; T *s_Xmat_hom = &s_temp[" + str(tempOffset) + " + 16*eeInd];")
    self.gen_add_code_line("s_eePos[6*eeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];")
    # roll pitch yaw is a bit more difficult
    self.gen_add_code_line("// roll pitch yaw is a bit more difficult")
    self.gen_add_code_line("if(xyzInd > 0){continue;}")
    self.gen_add_code_line("s_eePos[6*eeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);")
    self.gen_add_code_line("s_eePos[6*eeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));")
    self.gen_add_code_line("s_eePos[6*eeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    self.gen_add_end_function()

def gen_end_effector_positions_device_temp_mem_size(self):
    n = self.robot.get_num_pos()
    wrapper_size = self.gen_topology_helpers_size() + self.gen_get_Xhom_size() # for Xhom
    return self.gen_end_effector_positions_inner_temp_mem_size() + wrapper_size

def gen_end_effector_positions_device(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    num_ees = self.robot.get_total_leaf_nodes()
    # construct the boilerplate and function definition
    func_params = ["s_eePos is a pointer to shared memory of size 6*NUM_EE where NUM_EE = " + str(num_ees), \
                   "s_q is the vector of joint positions", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)"]
    func_notes = []
    func_def_start = "void end_effector_positions_device("
    func_def_middle = "T *s_eePos, const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def = func_def_start + func_def_middle + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes the End Effector Position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    shared_mem_size = self.gen_end_effector_positions_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XmatsHom_helpers_temp_shared_memory_code(shared_mem_size)
    # then load/update XI and run the algo
    self.gen_load_update_XmatsHom_helpers_function_call(use_thread_group)
    self.gen_end_effector_positions_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_end_effector_positions_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    num_ees = self.robot.get_total_leaf_nodes()
    # define function def and params
    func_params = ["d_eePos is the vector of end effector positions", \
                   "d_q is the vector of joint positions", \
                   "stride_q is the stide between each q", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void end_effector_positions_kernel(T *d_eePos, const T *d_q, const int stride_q, "
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Compute the End Effector Position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q[" + str(n) + "];", \
                       "__shared__ T s_eePos[" + str(6*num_ees) + "];"]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_end_effector_positions_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XmatsHom_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q","stride_q",str(n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        # then load/update X and run the algo
        self.gen_load_update_XmatsHom_helpers_function_call(use_thread_group)
        self.gen_end_effector_positions_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("eePos",str(6*num_ees),str(6*num_ees),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q",str(n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        # then load/update X and run the algo
        self.gen_load_update_XmatsHom_helpers_function_call(use_thread_group)
        self.gen_end_effector_positions_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("eePos",str(6*num_ees),use_thread_group)
    self.gen_add_end_function()

def gen_end_effector_positions_host(self, mode = 0):
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void end_effector_positions(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end =   "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the End Effector Positions",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "end_effector_positions_kernel<T><<<block_dimms,thread_dimms,EE_POS_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_eePos,hd_data->d_q,stride_q,"
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "int stride_q;", \
                                 "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "else {stride_q = 3*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    # then compute but adjust for compressed mem and qdd usage
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    # add in compressed mem adjusts
    func_call_mem_adjust = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "else                    {" + func_call.replace("hd_data->d_q","hd_data->d_q_qd_u") + "}"
    # compule into a set of code
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_eePos,hd_data->d_eePos,6*NUM_EES*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call EEPOS %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_end_effector_positions_gradient_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    num_ees = self.robot.get_total_leaf_nodes()
    return 2*16*num_ees*n

def gen_end_effector_positions_gradient_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_Xhom_name = "s_XmatsHom", \
        s_dXhom_name = "s_dXmatsHom", \
        s_deePos_name = "s_deePos", \
        s_q_name = "s_q", \
        s_topology_helpers_name = "s_topology_helpers", \
        s_temp_name = "s_temp", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    code_start = "end_effector_positions_gradient_inner<T>(" + var_names["s_deePos_name"] + ", " + var_names["s_q_name"] + ", "
    code_middle = var_names["s_Xhom_name"] + ", " + var_names["s_dXhom_name"] + ", "
    code_end =  var_names["s_temp_name"] + ");"
    # account for thread group
    if use_thread_group:
        code_start = code_start.replace("(","(tgrp, ")
    if not self.robot.is_serial_chain():
        code_middle += var_names["s_topology_helpers_name"] + ", "
    self.gen_add_code_line(code_start + code_middle + code_end)

def gen_end_effector_positions_gradient_inner(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
    all_ees = self.robot.get_leaf_nodes()
    num_ees = len(all_ees)
    # construct the boilerplate and function definition
    func_params = ["s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = " + str(n) + " and NUM_EE = " + str(num_ees), \
                   "s_q is the vector of joint positions", \
                   "s_Xhom is the pointer to the homogenous transformation matricies ", \
                   "s_dXhom is the pointer to the gradient of the homogenous transformation matricies ", \
                   "s_temp is a pointer to helper shared memory of size " + \
                            str(self.gen_end_effector_positions_gradient_inner_temp_mem_size())]
    func_notes = ["Assumes the Xhom and dXhom matricies have already been updated for the given q"]
    func_def_start = "void end_effector_positions_gradient_inner("
    func_def_middle = "T *s_deePos, const T *s_q, const T *s_Xhom, const T *s_dXhom, "
    func_def_end = "T *s_temp) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -1, NO_XI_FLAG = True)
    func_def = func_def_start + func_def_middle + func_def_end
    # now generate the code
    self.gen_add_func_doc("Computes the Gradient of the End Effector Position with respect to joint position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    #
    # Initial Debug Prints if Requested
    #
    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,4,4>(&s_Xhom[16*i],4);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"dX[%d]\\n\",i); printMat<T,4,4>(&s_dXhom[16*i],4);}")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # For each chain we need to (in parallel) multiply the (d)Xmats
    # 
    self.gen_add_code_line("//")
    self.gen_add_code_line("// For each branch/gradient in parallel chain up the transform")
    self.gen_add_code_line("// Keep chaining until reaching the root (starting from the leaves)")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels): # at most bfs levels of parents to chain
        # if serial chain manipulator then this is easy
        if self.robot.is_serial_chain():
            self.gen_add_code_line("// Serial chain manipulator so optimize as parent is jid-1")
            if bfs_level == 0:
                self.gen_add_code_line("// First set to leaf transform")
                self.gen_add_parallel_loop("ind",str(16*n),use_thread_group)
                self.gen_add_code_line("int djid = ind / 16; int rc = ind % 16; int eeIndStart = 16*" + str(all_ees[0]) + ";")
                self.gen_add_code_line("s_temp[ind] = (djid == " + str(all_ees[0]) + ") ? s_dXhom[eeIndStart + rc] : s_Xhom[eeIndStart + rc];")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
                if self.DEBUG_MODE:
                    self.gen_add_sync(use_thread_group)
                    self.gen_add_serial_ops(use_thread_group)
                    self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"dX_chain0[%d]\\n\",i); printMat<T,4,4>(&s_temp[16*i],4);}")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync(use_thread_group)
            else:
                self.gen_add_code_line("// Update with parent transform until you reach the base [level " + str(bfs_level) + "/" + str(n_bfs_levels-1) + "]")
                self.gen_add_parallel_loop("ind",str(16*n),use_thread_group)
                self.gen_add_code_line("int djid = ind / 16; int rc = ind % 16; int row = rc % 4; int colInd = ind - row;")
                # get the parents we need at this level working backwards from all_ees
                parent = all_ees[0]
                for i in range(bfs_level):
                    parent = self.robot.get_parent_id(parent)
                # need to swap dst and start each time
                even = bfs_level % 2
                tempDstOffset = 16*n*(even)
                tempSrcOffset = 16*n*(not even)
                self.gen_add_code_line("const T *s_Xhom_dXhom = ((djid == " + str(parent) + ") ? s_dXhom : s_Xhom);")
                self.gen_add_code_line("s_temp[ind + " + str(tempDstOffset) + "] = dot_prod<T,4,4,1>" + \
                                       "(&s_Xhom_dXhom[16*" + str(parent) + " + row], &s_temp[" + str(tempSrcOffset) + " + colInd]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
                if self.DEBUG_MODE:
                    self.gen_add_sync(use_thread_group)
                    self.gen_add_serial_ops(use_thread_group)
                    self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"dX_chain[%d]\\n\",i); printMat<T,4,4>(&s_temp[16*i + " + str(tempDstOffset) + "],4);}")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync(use_thread_group)
        else:
            # if first loop then just set to transform at the leaf
            if bfs_level == 0:
                self.gen_add_code_line("// First set to leaf transform")
                self.gen_add_parallel_loop("ind",str(16*n*num_ees),use_thread_group)
                self.gen_add_code_line("int rc = ind % 16; int djid = (ind / 16) % " + str(n) + ";")
                select_var_vals = [("int", "eeInd", [str(jid) for jid in all_ees])]
                # make sure to zero out all things not in the chain
                jidChainCode = []
                for eejid in all_ees:
                    jidChain = sorted(self.robot.get_ancestors_by_id(eejid))
                    jidChain.append(eejid)
                    code = self.gen_var_in_list("djid", [str(jid) for jid in jidChain])
                    jidChainCode.append(code)
                select_var_vals.append(("bool", "inChain", jidChainCode))
                self.gen_add_multi_threaded_select("ind", "<", [str(16*n*(i+1)) for i in range(num_ees)], select_var_vals)
                self.gen_add_code_line("s_temp[ind] = inChain * ((djid == eeInd) ? s_dXhom[16*eeInd + rc] : s_Xhom[16*eeInd + rc]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
                if self.DEBUG_MODE:
                    self.gen_add_sync(use_thread_group)
                    self.gen_add_serial_ops(use_thread_group)
                    self.gen_add_code_line("for (int i = 0; i < " + str(n*num_ees) + "; i++){printf(\"dX_chain level[%d] with dj_ee_id [%d]\\n\"," + str(bfs_level) + ",i); printMat<T,4,4>(&s_temp[16*i],4);}")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync(use_thread_group)
            else:
                self.gen_add_code_line("// Update with parent transform until you reach the base [level " + str(bfs_level) + "/" + str(n_bfs_levels-1) + "]")
                self.gen_add_parallel_loop("ind",str(16*n*num_ees),use_thread_group)
                self.gen_add_code_line("int rc = ind % 16; int djid = (ind / 16) % " + str(n) + ";")
                self.gen_add_code_line("int row = rc % 4; int colInd = ind - row;")
                # get the parents we need at this level working backwards from all_ees
                curr_parents = all_ees
                for i in range(bfs_level):
                    curr_parents = [(-1 if jid == -1 else self.robot.get_parent_id(jid)) for jid in curr_parents]
                # need to swap dst and start each time
                even = bfs_level % 2
                tempDstOffset = 16*n*num_ees*(even)
                tempSrcOffset = 16*n*num_ees*(not even)
                # get parents for this level
                select_var_vals = [("int", "parent_jid", [str(jid) for jid in curr_parents])]
                self.gen_add_multi_threaded_select("ind", "<", [str(16*n*(i+1)) for i in range(num_ees)], select_var_vals)
                if (-1 in curr_parents):
                    self.gen_add_code_line("if(parent_jid == -1){continue;}")
                self.gen_add_code_line("const T *s_Xhom_dXhom = ((djid == parent_jid) ? s_dXhom : s_Xhom);")
                self.gen_add_code_line("s_temp[ind + " + str(tempDstOffset) + "] = dot_prod<T,4,4,1>" + \
                                       "(&s_Xhom_dXhom[16*parent_jid + row], &s_temp[" + str(tempSrcOffset) + " + colInd]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
                if self.DEBUG_MODE:
                    self.gen_add_sync(use_thread_group)
                    self.gen_add_serial_ops(use_thread_group)
                    self.gen_add_code_line("for (int i = 0; i < " + str(n*num_ees) + "; i++){printf(\"dX_chain[%d]\\n\",i); printMat<T,4,4>(&s_temp[16*i + " + str(tempDstOffset) + "],4);}")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync(use_thread_group)
    
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Now extract the eePos from the Tansforms")
    self.gen_add_code_line("// TODO: ADD OFFSETS")
    self.gen_add_code_line("//")
    tempOffset = 16*n*num_ees*(bfs_level % 2)
    # xyz position is easy (eePos_xyz1 = Xmat_hom * offset) where offset = [x,y,z,1]
    self.gen_add_parallel_loop("ind",str(3*n*num_ees),use_thread_group)
    self.gen_add_code_line("// xyz is easy")
    self.gen_add_code_line("int xyzInd = ind % 3; int deeInd = ind / 3; T *s_Xmat_hom = &s_temp[" + str(tempOffset) + " + 16*deeInd];")
    self.gen_add_code_line("s_deePos[6*deeInd + xyzInd] = s_Xmat_hom[12 + xyzInd];")
    # roll pitch yaw is a bit more difficult
    #
    #
    # TODO THESE ARE WRONG BECUASE THERE IS CHAIN RULE HERE
    #
    #
    self.gen_add_code_line("// roll pitch yaw is a bit more difficult")
    self.gen_add_code_line("//")
    self.gen_add_code_line("//")
    self.gen_add_code_line("// TODO THESE ARE WRONG BECUASE THERE IS CHAIN RULE HERE")
    self.gen_add_code_line("//")
    self.gen_add_code_line("//")
    self.gen_add_code_line("if(xyzInd > 0){continue;}")
    self.gen_add_code_line("s_deePos[6*deeInd + 3] = atan2(s_Xmat_hom[6],s_Xmat_hom[10]);")
    self.gen_add_code_line("s_deePos[6*deeInd + 4] = -atan2(s_Xmat_hom[2],sqrt(s_Xmat_hom[6]*s_Xmat_hom[6] + s_Xmat_hom[10]*s_Xmat_hom[10]));")
    self.gen_add_code_line("s_deePos[6*deeInd + 5] = atan2(s_Xmat_hom[1],s_Xmat_hom[0]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    self.gen_add_end_function()

def gen_end_effector_positions_gradient_device_temp_mem_size(self):
    n = self.robot.get_num_pos()
    wrapper_size = self.gen_topology_helpers_size() + 2*self.gen_get_Xhom_size() # for Xhom and dXhom
    return self.gen_end_effector_positions_inner_temp_mem_size() + wrapper_size

def gen_end_effector_positions_gradient_device(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    num_ees = self.robot.get_total_leaf_nodes()
    # construct the boilerplate and function definition
    func_params = ["s_deePos is a pointer to shared memory of size 6*NUM_JOINTS*NUM_EE where NUM_JOINTS = " + str(n) + " and NUM_EE = " + str(num_ees), \
                   "s_q is the vector of joint positions", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)"]
    func_notes = []
    func_def_start = "void end_effector_positions_gradient_device("
    func_def_middle = "T *s_deePos, const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def = func_def_start + func_def_middle + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes the Gradient of the End Effector Position with respect to joint position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    shared_mem_size = self.gen_end_effector_positions_inner_gradient_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XmatsHom_helpers_temp_shared_memory_code(shared_mem_size, include_gradients = True)
    # then load/update XI and run the algo
    self.gen_load_update_XmatsHom_helpers_function_call(use_thread_group, include_gradients = True)
    self.gen_end_effector_positions_gradient_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_end_effector_positions_gradient_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    num_ees = self.robot.get_total_leaf_nodes()
    # define function def and params
    func_params = ["d_deePos is the vector of end effector positions gradients", \
                   "d_q is the vector of joint positions", \
                   "stride_q is the stide between each q", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void end_effector_positions_gradient_kernel(T *d_deePos, const T *d_q, const int stride_q, "
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Computes the Gradient of the End Effector Position with respect to joint position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q[" + str(n) + "];", \
                       "__shared__ T s_deePos[" + str(6*n*num_ees) + "];"]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_end_effector_positions_gradient_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XmatsHom_helpers_temp_shared_memory_code(shared_mem_size, include_gradients = True)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q","stride_q",str(n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        # then load/update X and run the algo
        self.gen_load_update_XmatsHom_helpers_function_call(use_thread_group, include_gradients = True)
        self.gen_end_effector_positions_gradient_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("deePos",str(6*n*num_ees),str(6*n*num_ees),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q",str(n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        # then load/update X and run the algo
        self.gen_load_update_XmatsHom_helpers_function_call(use_thread_group, include_gradients = True)
        self.gen_end_effector_positions_gradient_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("deePos",str(6*n*num_ees),use_thread_group)
    self.gen_add_end_function()

def gen_end_effector_positions_gradient_host(self, mode = 0):
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void end_effector_positions_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end =   "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Computes the Gradient of the End Effector Position with respect to joint position",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "end_effector_positions_gradient_kernel<T><<<block_dimms,thread_dimms,DEE_POS_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_deePos,hd_data->d_q,stride_q,"
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "int stride_q;", \
                                 "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "else {stride_q = 3*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    # then compute but adjust for compressed mem and qdd usage
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    # add in compressed mem adjusts
    func_call_mem_adjust = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "else                    {" + func_call.replace("hd_data->d_q","hd_data->d_q_qd_u") + "}"
    # compule into a set of code
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_deePos,hd_data->d_deePos,6*NUM_EES*NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call DEEPOS %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_eepos_and_gradient(self, use_thread_group = False):
    # first generate the inner helpers
    self.gen_end_effector_positions_inner(use_thread_group)
    # then generate the device wrappers
    self.gen_end_effector_positions_device(use_thread_group)
    # then generate the kernels
    self.gen_end_effector_positions_kernel(use_thread_group,True)
    self.gen_end_effector_positions_kernel(use_thread_group,False)
    # then the host launch wrappers
    self.gen_end_effector_positions_host(0)
    self.gen_end_effector_positions_host(1)
    self.gen_end_effector_positions_host(2)

    # then for the gradients first generate the inner helpers
    self.gen_end_effector_positions_gradient_inner(use_thread_group)
    # then generate the device wrappers
    self.gen_end_effector_positions_gradient_device(use_thread_group)
    # then generate the kernels
    self.gen_end_effector_positions_gradient_kernel(use_thread_group,True)
    self.gen_end_effector_positions_gradient_kernel(use_thread_group,False)
    # then the host launch wrappers
    self.gen_end_effector_positions_gradient_host(0)
    self.gen_end_effector_positions_gradient_host(1)
    self.gen_end_effector_positions_gradient_host(2)
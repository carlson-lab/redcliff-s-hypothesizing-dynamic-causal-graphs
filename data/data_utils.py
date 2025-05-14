# <><><>
import os
import json
from json import JSONEncoder
import random
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

from general_utils.metrics import get_number_of_connected_components



class NumpyArrayEncoder(JSONEncoder): # see https://pynative.com/python-serialize-numpy-ndarray-into-json/
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
    
def save_data(save_path_for_data, samples, NUM_SAMPLES_IN_DATASET, NUM_SAMPS_PER_FILE, file_prefix="nc0-0ns1200_subset_"):
    curr_file_start_index = 0
    curr_file_counter = 0
    while curr_file_start_index < NUM_SAMPLES_IN_DATASET:
        curr_end_index = curr_file_start_index + NUM_SAMPS_PER_FILE
        with open(save_path_for_data+os.sep+file_prefix+str(curr_file_counter)+".pkl", 'wb') as outfile:
            pkl.dump(samples[curr_file_start_index:curr_end_index], outfile)
        curr_file_start_index += NUM_SAMPS_PER_FILE
        curr_file_counter += 1
    pass

def save_cached_args_file_for_data(data_root_path, num_channels, adjacency_tensors, final_file_name):
    file_str = '{'+'"data_root_path": "'+data_root_path+'", "num_channels": "' + str(num_channels) + '", '
    for i in range(len(adjacency_tensors)):
        net_id = i+1
        net_name = "net"+str(net_id)+"_adjacency_tensor"
        numpyData = {net_name: adjacency_tensors[i]} # see https://pynative.com/python-serialize-numpy-ndarray-into-json/
        encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)[1:-1] # see https://pynative.com/python-serialize-numpy-ndarray-into-json/
        updatedEncodedNumpyData = encodedNumpyData[:len(net_name)+4] + '"' + encodedNumpyData[len(net_name)+4:] + '"' # add " marks around array text
        file_str = file_str + updatedEncodedNumpyData + ", "
    file_str = file_str[:-2] + '}' # cut off final comma-space and add curly bracket
    with open(data_root_path+os.sep+final_file_name, 'w') as outfile: # see https://www.geeksforgeeks.org/saving-text-json-and-csv-to-a-file-in-python/
        outfile.write(file_str)
    pass


def multivariate_relational_nvar_sinusoid_with_gaussian_innovations(history, lagged_adjacencies, f=np.pi*np.ones([3,1]), mu=np.zeros((3,1)), var=np.ones([3,1]),
                                                                    innovation_amp=np.ones([3,1]), d=3, NUM_LAGS=2,
                                                                    nonlinear_functions_by_lagged_adjacency=[[[None for _ in range(2)] for _ in range(3)] for _ in range(3)]):
    """
    Function for forecasting/generating next dynamical system variable values according to a (potentially nonlinear) vector auto-regressive process

    Arguments:
        history: list of historical variable states with enough recorded time steps to generate next system configuration
        lagged_adjacencies: dxdxNUM_LAGS array defining the (lagged) edges between variables in a dynamical system
        f: 'base' frequency of each variable - i.e. the frequency a variable would oscilate at if it were independent/not noisy
        mu: mean of noise of each variable
        var: (gaussian) variance of noise of each variable
        innovation_amp: amplitude coefficient for scaling noise of each variable
        d: number of system variables
        NUM_LAGS: number of relevant time steps for forecasting system evolution (currently only 2 lags are supported to enforce regular periodicity of independent nodes)
        nonlinear_functions_by_lagged_adjacency: nonlinear activation functions that act on each edge in the lagged_adjacencies graph - defaults to no nonlinearities
    Returns:
        x_hat: dx1 ndarray representing the next state of the dynamical system (across all variables)
    """
    x_hat = np.zeros((d,1))
    for i in range(d):
        for j in range(d):
            if i == j: # self-connection
                lag1_contribution = lagged_adjacencies[i,j,0]*(2*np.cos(2*np.pi*f[i,0])*history[-1][i,0])
                if nonlinear_functions_by_lagged_adjacency[i][j][0] is not None:
                    lag1_contribution = nonlinear_functions_by_lagged_adjacency[i][j][0](lag1_contribution)
                lag2_contribution = 0.
                if NUM_LAGS > 1:
                    lag2_contribution = lagged_adjacencies[i,j,1]*(-1*history[-2][i,0])
                    if nonlinear_functions_by_lagged_adjacency[i][j][1] is not None:
                        lag2_contribution = nonlinear_functions_by_lagged_adjacency[i][j][1](lag2_contribution)
                x_hat[i,0] += lag1_contribution + lag2_contribution + innovation_amp[i,0]*np.random.normal(mu[i,0], var[i,0])# https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
            else:
                for l in range(NUM_LAGS):
                    curr_contribution = lagged_adjacencies[i,j,l]*history[-1*(l+1)][j,0]
                    if nonlinear_functions_by_lagged_adjacency[i][j][l] is not None:
                        curr_contribution = nonlinear_functions_by_lagged_adjacency[i][j][l](curr_contribution)
                    x_hat[i,0] += curr_contribution
    return x_hat


def sample_signal_from_system_state(curr_sys_state, INNOVATION_AMP_COEFFS, N_LAGS,
                                    D, LAGGED_ADJ_GRAPHS, NONLIN_FUNC_BY_ADJ_GRAPHS,
                                    BASE_FREQS, NOISE_MU, NOISE_VAR, RECORDING_LENGTH,
                                    BURNIN_PERIOD):
    # get initial conditions
    avg_innovation_amp = np.mean(INNOVATION_AMP_COEFFS)
    assert N_LAGS == 2
    X_t0 = np.random.uniform(-1*avg_innovation_amp,avg_innovation_amp,D).reshape(D,1)
    X_t1 = multivariate_relational_nvar_sinusoid_with_gaussian_innovations(
        [X_t0],
        LAGGED_ADJ_GRAPHS[curr_sys_state],
        f=BASE_FREQS,
        mu=NOISE_MU,
        var=NOISE_VAR,
        innovation_amp=INNOVATION_AMP_COEFFS,
        d=D,
        NUM_LAGS=1,
        nonlinear_functions_by_lagged_adjacency=NONLIN_FUNC_BY_ADJ_GRAPHS[curr_sys_state]
    )

    # propagate signal forward in time
    curr_samp = [X_t0, X_t1]+[]
    for n in range(N_LAGS,RECORDING_LENGTH+N_LAGS+BURNIN_PERIOD):
        curr_samp.append(
            multivariate_relational_nvar_sinusoid_with_gaussian_innovations(
                curr_samp,
                LAGGED_ADJ_GRAPHS[curr_sys_state],
                f=BASE_FREQS,
                mu=NOISE_MU,
                var=NOISE_VAR,
                innovation_amp=INNOVATION_AMP_COEFFS,
                d=D,
                NUM_LAGS=N_LAGS,
                nonlinear_functions_by_lagged_adjacency=NONLIN_FUNC_BY_ADJ_GRAPHS[curr_sys_state]
            )
        )
    curr_samp = np.concatenate(curr_samp[N_LAGS+BURNIN_PERIOD:], axis=1)
    return curr_samp


def sample_and_apply_linearly_interpolated_weights_to_signal(signal, num_steps):
    assert signal.shape[1] == num_steps
    start_weight = np.random.uniform()
    end_weight = np.random.uniform()
    interpolated_weights = np.linspace(start_weight, end_weight, num_steps)
    weighted_signal = signal * interpolated_weights
    return weighted_signal, interpolated_weights


def generate_synthetic_data(plot_save_path, NUM_SAMPLES_IN_DATASET, RECORDING_LENGTH, LABEL_TYPE, BURNIN_PERIOD,
                            D, NUM_POSSIBLE_SYS_STATES, NUM_LABELED_SYS_STATES, N_LAGS, LAGGED_ADJ_GRAPHS,
                            NONLIN_FUNC_BY_ADJ_GRAPHS, BASE_FREQS, NOISE_MU,
                            NOISE_VAR, INNOVATION_AMP_COEFFS, NOISE_AMP_COEFFS, NOISE_TYPE="white"):
    assert NUM_LABELED_SYS_STATES <= NUM_POSSIBLE_SYS_STATES
    if NUM_POSSIBLE_SYS_STATES > NUM_LABELED_SYS_STATES:
        NUM_LABELED_SYS_STATES += 1 # add an extra 'UNKNOWN' label to capture unsupervised system states
    assert NOISE_TYPE in ["gaussian", "white", "superpositional"]
    
    avg_innovation_amp = np.mean(INNOVATION_AMP_COEFFS)
    samples = []
    for s in range(NUM_SAMPLES_IN_DATASET):
        if s % 100 == 0:
            print("generate_synthetic_data: \t s == ", s, flush=True)
        
        # draw sample/label from system
        curr_samp = np.zeros((D, RECORDING_LENGTH))
        curr_true_label = np.zeros((NUM_LABELED_SYS_STATES,RECORDING_LENGTH))
        for sys_state in range(NUM_POSSIBLE_SYS_STATES):
            sys_samp = sample_signal_from_system_state(
                sys_state,
                INNOVATION_AMP_COEFFS,
                N_LAGS,
                D,
                LAGGED_ADJ_GRAPHS,
                NONLIN_FUNC_BY_ADJ_GRAPHS,
                BASE_FREQS,
                NOISE_MU,
                NOISE_VAR,
                RECORDING_LENGTH, 
                BURNIN_PERIOD
            )
            dynamic_sys_samp, sys_activation_weights = sample_and_apply_linearly_interpolated_weights_to_signal(sys_samp, RECORDING_LENGTH)
            curr_samp = curr_samp + dynamic_sys_samp
            if sys_state < NUM_LABELED_SYS_STATES-1:
                curr_true_label[sys_state,:] = curr_true_label[sys_state,:] + sys_activation_weights
            else:
                curr_true_label[-1,:] = curr_true_label[-1,:] + sys_activation_weights
        curr_true_label[-1,:] = curr_true_label[-1,:] / (1.*(NUM_POSSIBLE_SYS_STATES - (NUM_LABELED_SYS_STATES-1))) # normalize unsupervised labels to be within same range as supervised onese
                
        curr_label = np.zeros((NUM_LABELED_SYS_STATES,RECORDING_LENGTH))
        if LABEL_TYPE == "Oracle":
            curr_label = curr_label + curr_true_label
        elif LABEL_TYPE == "OneHot":
            for t in range(RECORDING_LENGTH):
                maximally_active_state = np.argmax(curr_true_label[:,t])
                curr_label[maximally_active_state,t] = 1.
        else:
            raise ValueError("Unrecognized LABEL_TYPE=="+str(LABEL_TYPE))

        # add noise to current (synthetic) recording
        additive_noise = None
        if NOISE_TYPE == "white":
            num_elements_in_curr_samp = np.prod(curr_samp.shape)
            additive_noise = NOISE_AMP_COEFFS*np.random.uniform(-1*avg_innovation_amp,avg_innovation_amp,num_elements_in_curr_samp).reshape(D, -1)
        elif NOISE_TYPE == "gaussian":
            num_elements_in_curr_samp = np.prod(curr_samp.shape)
            avg_innov_center = np.mean(NOISE_MU)
            avg_innov_var = np.mean(NOISE_VAR)
            additive_noise = NOISE_AMP_COEFFS*np.random.normal(avg_innov_center,avg_innov_var*avg_innovation_amp,num_elements_in_curr_samp).reshape(D, -1)
        elif NOISE_TYPE == "superpositional":
            raise ValueError("superpositional noise is DECREMENTED - now assumed as part of the original signal")
            additive_noise = np.zeros(curr_samp.shape)
            for background_state in range(NUM_POSSIBLE_SYS_STATES):
                if background_state != curr_sys_state:
                    curr_noise_sig = sample_signal_from_system_state(
                        background_state,
                        INNOVATION_AMP_COEFFS,
                        N_LAGS,
                        D,
                        LAGGED_ADJ_GRAPHS,
                        NONLIN_FUNC_BY_ADJ_GRAPHS,
                        BASE_FREQS,
                        NOISE_MU,
                        NOISE_VAR,
                        RECORDING_LENGTH,
                        BURNIN_PERIOD
                    )
                    additive_noise = additive_noise + curr_noise_sig
            additive_noise = NOISE_AMP_COEFFS*additive_noise
        else:
            raise ValueError()
        curr_samp = curr_samp + additive_noise

        # record sample and label
        samples.append([curr_samp.T, None, None, curr_label])

        # plot the first 10 synthetic signals for visualization
        if s < 10:
            plt.plot(curr_samp.T, alpha=0.7)
            plt.title("Sample "+str(s))
            plt.legend()
            plt.draw()
            plt.savefig(plot_save_path+os.sep+"full_sample_"+str(s)+"_vis.png")
            plt.close()
        if s < 10:
            plt.plot(curr_samp[:,:10].T, alpha=0.7)
            plt.title("Sample "+str(s)+": FIRST 10 STEPS")
            plt.legend()
            plt.draw()
            plt.savefig(plot_save_path+os.sep+"partial_sample_"+str(s)+"_vis.png")
            plt.close()

    return samples


def generate_lagged_ajacency_graphs_for_factor_model(plot_save_path, num_nodes, num_lags, num_factors, make_factors_orthogonal,
                                                     make_factors_singular_components, rand_seed=0,
                                                     off_diag_edge_strengths=[0.1,1.],
                                                     diag_receiving_node_forgetting_coeffs=[0.1,1.],
                                                     diag_sending_node_forgetting_coeffs=[0.9,1.],
                                                     num_edges_per_graph=None, max_formulation_attempts=100, 
                                                     nonlinear_off_diag_edge_activations=None):
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    graphs = None
    graph_activations = None
    restart_curration = True
    while restart_curration:
        restart_curration = False
        print("generate_lagged_ajacency_graphs_for_factor_model: STARTING CURRATION FROM SCRATCH")
        graphs = [None for _ in range(num_factors)]
        graph_activations = [None for _ in range(num_factors)]
        max_num_connected_comps = 1 if make_factors_singular_components else num_nodes

        if num_edges_per_graph is None:
            num_edges_per_graph = (num_nodes**2)//num_factors
        if make_factors_singular_components: # ensure parameters are mathematically compatible
            assert num_edges_per_graph >= num_nodes-1
        print("generate_lagged_ajacency_graphs_for_factor_model: GENERATING GRAPHS WITH ", num_edges_per_graph, " EDGES")

        available_edge_ids = []
        available_edges = []
        id_counter = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(num_lags):
                    if i != j:
                        available_edge_ids.append(id_counter)
                        available_edges.append((i,j,k))
                        id_counter += 1

        for i in range(num_factors):
            print("generate_lagged_ajacency_graphs_for_factor_model: now generating graph for factor i == ", i, flush=True)
            curr_num_connected_comps = num_nodes+1
            make_curr_graph_orthogonal = make_factors_orthogonal
            A = None
            A_activations = None
            curr_edge_ids = None
            curr_edges = None
            num_formulation_attempts = 0

            while curr_num_connected_comps > max_num_connected_comps and not restart_curration:
                if num_formulation_attempts % 10 == 0:
                    print("generate_lagged_ajacency_graphs_for_factor_model: \t num_formulation_attempts == ", num_formulation_attempts, flush=True)

                A = np.zeros((num_nodes, num_nodes, num_lags)) # lagged ajacency tensor
                for l in range(num_lags):
                    A[:,:,l] = A[:,:,l] + np.eye(num_nodes)
                A_activations = [[[None for _ in range(num_lags)] for _ in range(num_nodes)] for _ in range(num_nodes)]

                random.shuffle(available_edge_ids)
                curr_edge_ids = available_edge_ids[:num_edges_per_graph]
                curr_edges = [available_edges[id] for id in curr_edge_ids]
                for (x,y,z) in curr_edges:
                    A[x,y,z] = off_diag_edge_strengths[z]
                    A[x,x,0] *= diag_receiving_node_forgetting_coeffs[0]
                    A[x,x,1] *= diag_receiving_node_forgetting_coeffs[1]
                    A[y,y,0] *= diag_sending_node_forgetting_coeffs[0]
                    A[y,y,1] *= diag_sending_node_forgetting_coeffs[1]
                    if nonlinear_off_diag_edge_activations is not None and nonlinear_off_diag_edge_activations[i] is not None:
                        A_activations[x][y][z] = nonlinear_off_diag_edge_activations[i][z]

                # check stopping criteria
                A_no_lags = A.sum(axis=2)
                # determine the number of connected components in the non-lagged graph
                curr_num_connected_comps = get_number_of_connected_components(A_no_lags, add_self_connections=False)

                num_formulation_attempts += 1
                if num_formulation_attempts == max_formulation_attempts:
                    restart_curration = True
                pass
            if restart_curration:
                break

            graphs[i] = A
            graph_activations[i] = A_activations
            if make_factors_orthogonal: # orthogonal is a bit of a misnomer - the sets of off-diagonal edges will be orthogonal, but self-connections are preserved
                print("generate_lagged_ajacency_graphs_for_factor_model: curr_edge_ids == ", curr_edge_ids, flush=True)
                ids_to_exclude = list(curr_edge_ids)+[]
                for (xU,yU,_) in curr_edges:
                    for id in available_edge_ids[num_edges_per_graph:]:
                        x = available_edges[id][0]
                        y = available_edges[id][1]
                        if xU==x and yU==y:
                            ids_to_exclude.append(id)
                available_edge_ids = [id for id in available_edge_ids if id not in ids_to_exclude]

            print("generate_lagged_ajacency_graphs_for_factor_model: factor i == ", i, " now has graph A == ", A)
            for nl in range(num_lags):
                plt.imshow(A[:,:,nl])
                plt.draw()
                plt.savefig(plot_save_path+os.sep+"graph"+str(i)+"_lag"+str(nl)+"_vis.png")
                plt.close()
    
    assert len(graphs) == len(graph_activations)
    # the following block of code follows that found at https://www.geeksforgeeks.org/python-shuffle-two-lists-with-same-order/
    factor_inds = [i for i in range(len(graphs))]
    temp = list(zip(graphs, graph_activations, factor_inds))
    random.shuffle(temp)
    graphs_shuffled, graph_activations_shuffled, factor_inds_shuffled = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    graphs_shuffled, graph_activations_shuffled, factor_inds_shuffled = list(graphs_shuffled), list(graph_activations_shuffled), list(factor_inds_shuffled)
    
    print("generate_lagged_ajacency_graphs_for_factor_model: FACTORS HAVE BEEN SHUFFLED INTO THE FOLLOWING FINAL ORDER factor_inds_shuffled == ", factor_inds_shuffled, flush=True)
    
    return graphs_shuffled, graph_activations_shuffled
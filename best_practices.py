import argparse as ap
import fnmatch as fnm
import numpy as np
import os
import pickle as pk
import torch
import torch.nn as nn
import torch.optim as opt

from collections import OrderedDict

import dfatools.models as dfamodels
import mltools.data as data
import mltools.logging as log
import mltools.models as models
import mltools.processing as proc
import mltools.utilities as util


LOGTAG = "MAIN"


# Utility functions.
def remove_dropout(model_description):
    new_model = model_description.copy()
    for layer_name, layer in model_description.items():
        if type(layer) == nn.Dropout or type(layer) == nn.Dropout2d or type(layer) == nn.Dropout3d:
            del new_model[layer_name]
    return new_model


def remove_batchnorm(model_description):
    new_model = model_description.copy()
    for layer_name, layer in model_description.items():
        if type(layer) == nn.BatchNorm1d or type(layer) == nn.BatchNorm2d or type(layer) == nn.BatchNorm3d:
            del new_model[layer_name]
    return new_model


def run_once(data_path, save_path, test=False, use_dfa=True, use_feedback_normalization=True, use_PAI=False,
             use_conv=False, dropout_rate=None, use_batchnorm=False, activation=nn.Tanh, dataset='CIFAR-10',
             epochs=50, batch_size=128, past_state_path=None, seed=0, gpu_id=0):
    base_name = f"3_{'test' if test else 'eval'}_{'DFA' if use_dfa else 'BP'}" \
        f"_{'FBnorm' if use_feedback_normalization else 'nonorm'}" \
        f"_{'initPAI' if use_PAI else 'initSTD'}_{'CONV' if use_conv else 'FC'}" \
        f"_{'ND' if dropout_rate is None else f'e{dropout_rate[0]}_fc{dropout_rate[1]}_conv{dropout_rate[2]}'}" \
        f"_{'BN' if use_batchnorm else 'X'}_{activation().__str__()}_{dataset}_e{epochs}_bs{batch_size}" \
        f"_{'retrieved' if past_state_path is not None else 'from_scratch'}_s{seed}"

    # Set-up a logging file.
    if save_path is not None:
        log_file_name = "log_{0}.txt".format(base_name)
        log_save_path = os.path.join(save_path, log_file_name)
        log.setup_logging(log.Level.INFO, log_save_path)
    else:
        log.setup_logging(log.Level.INFO)
        log.log("You have not specified a save file, no data will be kept from the run!", LOGTAG, log.Level.ERROR)

    log.log("<b><u>Establishing Baselines for Direct Feedback Alignment</u></b>", LOGTAG, log.Level.WARNING)
    log.log("<b>Section 3 -- Establishing Best Practices for DFA</b>", LOGTAG, log.Level.WARNING)

    log.log("Setting-up processing back-end and seeds...", LOGTAG, log.Level.INFO)
    # For larger architectures that have high memory needs, the feedback matrix can be kept on another GPU (rp_device)
    # and the BP model used for angle calculations as well (bp_device). Implementation is not fully tested for BP on
    # a separate device, and some tensors may need to be moved around. This code has also not been tested on CPU only.
    device = proc.enable_cuda(gpu_id, seed)
    rp_device, bp_device = device, device

    # Setting-up random number generation.
    log.log(f"Seeding with <b>{seed}</b>.", LOGTAG, log.Level.DEBUG)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setting-up data: transforms and data loaders.
    log.log(f"Preparing data for dataset <b>{dataset}</b> with batch size {batch_size}...", LOGTAG, log.Level.INFO)

    train_loader, validation_loader = data.get_loaders(dataset, batch_size, test, data_path)
    if use_PAI:
        # Prepare a separate loader for prealignment. This allows to experiment with different batch size and transforms
        # on prealignment.
        PAI_loader, _ = data.get_loaders(dataset, batch_size, test, data_path)

    # Setting-up model.
    log.log("Creating model...", LOGTAG, log.Level.INFO)
    # We create a model description with all the features possible (batchnorm and dropout) and then remove them if
    # they are not required.
    keep_dropout = True
    if dropout_rate is None:
        keep_dropout = False
        dropout_rate = [0, 0, 0]
    if use_conv:
        log.log("Using a <b>convolutional</b> architecture.", LOGTAG, log.Level.DEBUG)
        model_description = OrderedDict([('drop1', nn.Dropout2d(dropout_rate[0])),
                                         ('conv1', nn.Conv2d(3, 32, 5, padding=2)),
                                         ('drop2', nn.Dropout2d(dropout_rate[2])),
                                         ('act1', activation()),
                                         ('bn1', nn.BatchNorm2d(32)),
                                         ('maxp1', nn.MaxPool2d(3, 2)),
                                         ('conv2', nn.Conv2d(32, 64, 5, padding=2)),
                                         ('drop3', nn.Dropout2d(dropout_rate[2])),
                                         ('act2', activation()),
                                         ('bn2', nn.BatchNorm2d(64)),
                                         ('maxp2', nn.MaxPool2d(3, 2)),
                                         ('conv3', nn.Conv2d(64, 64, 5, padding=2)),
                                         ('drop4', nn.Dropout2d(dropout_rate[2])),
                                         ('act3', activation()),
                                         ('bn3', nn.BatchNorm2d(64)),
                                         ('maxp3', nn.MaxPool2d(3, 2)),
                                         ('flat', util.Flatten()),
                                         ('lin1', nn.Linear(576, 128)),
                                         ('drop5', nn.Dropout(dropout_rate[1])),
                                         ('act4', activation()),
                                         ('bn4', nn.BatchNorm1d(128)),
                                         ('lin2', nn.Linear(128, 10))])
    else:
        log.log("Using a <b>fully-connected</b> architecture.", LOGTAG, log.Level.DEBUG)
        model_description = OrderedDict([('flat', util.Flatten()),
                                         ('drop1', nn.Dropout(dropout_rate[0])),
                                         ('lin1', nn.Linear(3072, 800)),
                                         ('drop2', nn.Dropout(dropout_rate[1])),
                                         ('act1', activation()),
                                         ('bn1', nn.BatchNorm1d(800)),
                                         ('lin2', nn.Linear(800, 800)),
                                         ('drop3', nn.Dropout(dropout_rate[1])),
                                         ('act2', activation()),
                                         ('bn2', nn.BatchNorm1d(800)),
                                         ('lin3', nn.Linear(800, 800)),
                                         ('drop4', nn.Dropout(dropout_rate[1])),
                                         ('act3', activation()),
                                         ('bn3', nn.BatchNorm1d(800)),
                                         ('lin4', nn.Linear(800, 10))])

    if not keep_dropout:
        model_description = remove_dropout(model_description)
    else:
        log.log(f"Using <b>dropout</b> with rate {dropout_rate[0]} for input, {dropout_rate[2]} for conv. layers,"
                      f" {dropout_rate[2]} for fully-connected layers.", LOGTAG, log.Level.DEBUG)
    if not use_batchnorm:
        model_description = remove_batchnorm(model_description)
    else:
        log.log("Using <b>batch normalization</b>.", LOGTAG, log.Level.DEBUG)

    if use_dfa or use_PAI:
        # Create the DFA model, even if we run BP we will need it to calculate the prealignment initialization.
        model_dfa = dfamodels.DFAClassifier(device, device, model_description, train_loader, validation_loader,
                                            saving=(base_name, save_path, 5))
        model_dfa = model_dfa.to(device)
        if not use_feedback_normalization:
            log.log("<b>Not</b> using feedback normalization.", LOGTAG, log.Level.INFO)
            model_dfa.model.feedback_normalization = False
        model_dfa.initialize()
        if use_PAI:
            log.log("Using <b>PAI</b> (Pre-Alignment Initialization).", LOGTAG, log.Level.DEBUG)
            model_dfa.prealignment(PAI_loader, nn.CrossEntropyLoss())
        if use_dfa:
            log.log("Will be using <b>DFA</b> for training!", LOGTAG, log.Level.WARNING)
            model = model_dfa
    if not use_dfa:
        log.log("Will be using <b>BP</b> for training!", LOGTAG, log.Level.WARNING)
        model = models.Classifier(model_description, train_loader, validation_loader,
                                  saving=(base_name, save_path, 5))
        model = model.to(device)
        if use_PAI:
            model.model.load_state_dict(model_dfa.model.state_dict())

    if past_state_path is not None:
        with open(past_state_path, 'rb') as state_file:
            model.model.load_state_dict(torch.load(state_file))

    # Setting-up optimizer.
    optimizer_description = (opt.SGD, {'lr': 5 * 1e-4})

    # Train the model.
    log.log(f"Training with {'DFA' if use_dfa else 'BP'} initialized by {'PAI' if use_PAI else 'STD'} "
            f"for {epochs} epochs on dataset {dataset} with a batch size of {batch_size}:", LOGTAG, log.Level.INFO)
    log.log(f"Model: {model}", LOGTAG, log.Level.DEBUG)
    model.train(epochs, optimizer_description, loss_criterion=nn.CrossEntropyLoss())

    # Validate the model one last time.
    log.log(f"Final validation with {'DFA' if use_dfa else 'BP'} initialized by {'PAI' if use_PAI else 'STD'} "
            f"for {epochs} epochs on dataset {dataset} with a batch size of {batch_size}:", LOGTAG, log.Level.INFO)
    model.validate(loss_criterion=nn.CrossEntropyLoss())

    # Save the model training log and the weights (message log and angles are already saved dynamically).
    training_log_file_name = "training_log_{0}.tl".format(base_name)
    training_log_file_path = os.path.join(save_path, training_log_file_name)
    log.log("Finishing up: saving training log to {0}...".format(training_log_file_path), LOGTAG, log.Level.INFO)
    with open(training_log_file_path, 'wb') as training_log_file:
        pk.dump(model.training_log, training_log_file)

    state_dict_file_name = "state_{0}.pt".format(base_name)
    state_dict_file_path = os.path.join(save_path, state_dict_file_name)
    log.log("Finishing up: saving model state to {0}...".format(state_dict_file_path), LOGTAG, log.Level.INFO)
    torch.save(model.model.state_dict(), state_dict_file_path)

    return model.training_log


def process_training_log(path, training_log_name):
    training_log_path = os.path.join(path, training_log_name)
    with open(training_log_path, 'rb') as training_log_file:
        training_log = pk.load(training_log_file)
        train_error, test_error = [], []
        for entry in training_log[:-1]:
            if entry[0] == 'TRAIN':
                train_error.append(entry[2]['topk'][0])
            elif entry[0] == 'EVAL':
                test_error.append(entry[2]['topk'][0])
    return train_error, test_error

def process_alignment_log(path, alignment_log_name):
    alignment_log_path = os.path.join(path, alignment_log_name)
    with open(alignment_log_path, 'rb') as alignment_log_file:
        alignment_log = pk.load(alignment_log_file)
        return alignment_log


def analyse_results(base_name, results_path):
    train_errors, test_errors = [], []
    alignments = []
    for file_name in os.listdir(results_path):
        if fnm.fnmatch(file_name, '*.tl'):
            if base_name in file_name:
                run_train_error, run_test_error = process_training_log(results_path, file_name)
                alignments.append(process_alignment_log(results_path, f"{file_name[13:-3]}_alignment.al")[-1])
                train_errors.append(run_train_error[-1])
                test_errors.append(run_test_error[-1])
    print(base_name)
    print('test error', np.mean(test_errors), np.std(test_errors))
    print('train error', np.mean(train_errors), np.std(train_errors))
    best_alignment = alignments[train_errors.index(max(train_errors))]
    for layer in best_alignment:
        print(layer[0], layer[1])

if __name__ == '__main__':
    # Setting-up arguments from command line.
    parser = ap.ArgumentParser(description='Experimenting with prealignment initialization.')
    parser.add_argument('-test', action='store_true', default=True, help='use test set for validation.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='ID of the GPU to use.')
    parser.add_argument('-p', '--path', type=str, default='/data/mldata', help='path to folder containing datasets')
    parser.add_argument('-r', '--savepath', type=str, default=None, help='path to save folder')
    args = parser.parse_args()

    run_number = 10

    # TODO: Select where to save results for later processing.
    data_path = ""
    save_path_bn = ""
    save_path_act = ""
    save_path_norm = ""
    save_path_drop = ""

    half_leaky_relu = lambda: nn.LeakyReLU(0.5)
    tenth_leaky_relu = lambda: nn.LeakyReLU(0.1)

    for i in range(run_number):
        # TODO: uncomment the lines for the experiments of interest.
        run_once(data_path, save_path_bn, test=True, use_conv=False, use_batchnorm=True, seed=i, gpu_id=0)
        #run_once(data_path, save_path_bn, test=True, use_conv=True, use_batchnorm=True, seed=i, gpu_id=1)
        #run_once(data_path, save_path_act, test=True, activation=nn.Tanh, seed=i, gpu_id=2)
        #run_once(data_path, save_path_act, test=True, activation=nn.ReLU, seed=i, gpu_id=3)
        #run_once(data_path, save_path_act, test=True, activation=half_leaky_relu, seed=i, gpu_id=0)
        #run_once(data_path, save_path_act, test=True, activation=tenth_leaky_relu, seed=i, gpu_id=1)
        #run_once(data_path, save_path_act, test=True, activation=nn.LeakyReLU, seed=i, gpu_id=0)
        #run_once(data_path, save_path_norm, test=True, use_feedback_normalization=False, seed=i, gpu_id=2)
        #run_once(data_path, save_path_norm, test=True, use_conv=True, use_feedback_normalization=False, seed=i, gpu_id=3)
        #run_once(data_path, save_path_drop, test=True, dropout_rate=[0.1, 0.5, 0.5], seed=i, gpu_id=0)
        #run_once(data_path, save_path_drop, test=True, dropout_rate=[0.1, 0.1, 0.1], seed=i, gpu_id=1)
        #run_once(data_path, save_path_drop, test=True, use_conv=True, dropout_rate=[0.1, 0.5, 0], seed=i, gpu_id=2)
        #run_once(data_path, save_path_drop, test=True, use_conv=True, dropout_rate=[0.1, 0.1, 0], seed=i, gpu_id=3)
        #run_once(data_path, save_path_bn, test=True, use_conv=True, seed=i, gpu_id=2)

    # TODO: uncomment the lines for the processing of the experiment.
    analyse_results('3_test_DFA_FBnorm_initSTD_FC_ND_BN_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_bn)
    #analyse_results('3_test_DFA_FBnorm_initSTD_CONV_ND_BN_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_bn)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_ND_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_act)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_ND_X_ReLU()_CIFAR-10_e50_bs128_from_scratch_s', save_path_act)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_ND_X_LeakyReLU(negative_slope=0.5)_CIFAR-10_e50_bs128_from_scratch_s', save_path_act)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_ND_X_LeakyReLU(negative_slope=0.1)_CIFAR-10_e50_bs128_from_scratch_s', save_path_act)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_ND_X_LeakyReLU(negative_slope=0.01)_CIFAR-10_e50_bs128_from_scratch_s', save_path_act)
    #analyse_results('3_test_DFA_nonorm_initSTD_FC_ND_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_norm)
    #analyse_results('3_test_DFA_nonorm_initSTD_CONV_ND_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_norm)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_e0.1_fc0.5_conv0.5_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_drop)
    #analyse_results('3_test_DFA_FBnorm_initSTD_FC_e0.1_fc0.1_conv0.1_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_drop)
    #analyse_results('3_test_DFA_FBnorm_initSTD_CONV_e0.1_fc0.5_conv0_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_drop)
    #analyse_results('3_test_DFA_FBnorm_initSTD_CONV_e0.1_fc0.1_conv0_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_drop)
    #analyse_results('3_test_DFA_FBnorm_initSTD_CONV_ND_X_Tanh()_CIFAR-10_e50_bs128_from_scratch_s', save_path_bn)
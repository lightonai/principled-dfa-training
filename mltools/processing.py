""" Tools for setting-up GPU and enabling parallelization of multiple models over multiple GPUs.

This doesn't implement any distributed training of a single model, only running different models on different GPUs, for
instance for searching optimal architectures.
"""
import os
import pickle as pk
import torch
import torch.cuda as cuda
import torch.multiprocessing as mp

import mltools.logging as log

from multiprocessing import Queue
from queue import Empty


LOGTAG = "PROC"


def enable_cuda(gpu_id=0, cuba_seed=None):
    """Set-up CUDA for a given GPU and initialize its seed.

    Note this doesn't set the cudnn.deterministic flag (for performance reason), so even though the seed is set
    some stochastic behaviors might still occur.

    :param gpu_id: device index to select.
    :type gpu_id: int
    :param cuba_seed: the desired seed.
    :type cuba_seed: int
    :return: initialized device.
    :rtype: torch.device
    """
    if not torch.cuda.is_available():
        # Check availability of CUDA and default to CPU otherwise.
        log.log("CUDA is not available on this platform! This code requires the use of a GPU.", LOGTAG, log.Level.ERROR)
        raise NotImplementedError("This code is not designed to run on cpu only.")
    else:
        device = torch.device("cuda:{0}".format(gpu_id))
        cuda.set_device(device)
        log.log("Using GPU {0} with seed {1}.".format(gpu_id, cuba_seed), LOGTAG, log.Level.INFO)
        if cuba_seed is not None:
            torch.cuda.manual_seed(cuba_seed)
    return device


def parallelize(devices, parameters, function, use_signature=False, save_folder=None):
    """Parallelize evaluation of a function over a set of different parameters on multiple GPUs.

    :param:`function` will be evaluated for every list of parameters in parameters, with the GPUs in devices
    automatically used to spread evaluation.

    :param:`function` will be called iteratively as function(device, *parameters[i]).

    If :param:`use_signature` is True, only the first entry of each parameter list in parameters will be saved along the
    function results instead of all the parameters.

    DON'T FORGET TO SET mp.set_start_method('spawn') FROM THE MAIN THREAD, OTHERWISE THIS WILL NOT WORK.

    :param devices: devices over which to parallelize.
    :type devices: list[torch.device]
    :param parameters: parameters with which to run the function.
    :type parameters: list[list]
    :param function: function to run over parameters multiple times.
    :type function: function
    :param use_signature: save only the first parameter instead of every parameters.
    :type use_signature: bool
    :param save_folder: path to folder where to save intermediary results.
    :type save_folder: str
    :return: parameters evaluated and values returned by the function.
    :rtype: dict
    """
    # TODO: automated spawn not set warning.
    # Transfer parameters to a multithreading-compatible queue and prepare queue for returned data.
    parameters_queue, return_queue = Queue(), Queue()
    for p in parameters:
        parameters_queue.put(p)

    # Spawn the processes and handle them.
    processes = []
    for device in devices:
        # For each GPU, start its own processing thread:
        device_process = mp.Process(target=_gpu_thread, args=(device, parameters_queue, return_queue,
                                                              function, use_signature, save_folder))
        device_process.start()
        processes.append(device_process)
    for process in processes:
        # Wait for the processes to finish:
        process.join()

    # Extract the returned data from the queue and put it in a dictionnary.
    returned_values = {}
    while not return_queue.empty():
        returned_data = return_queue.get()
        returned_values[returned_data[0]] = returned_data[1]

    # Clean-up and close the queues.
    parameters_queue.close()
    return_queue.close()
    return returned_values


def _gpu_thread(device, parameters_queue, return_queue, function, use_signature, save_folder):
    """Internal function to create a GPU thread when using :func:`parallelize`.

    This will look for parameters in the parameters queue, and if there are some, will run the provided function over
    them, saving the results afterward and putting them in the return queue.
    If no parameters can be found, it will stop running.

    :param devices: devices over which to parallelize.
    :type devices: list[torch.device]
    :param parameters_queue: multi-processing safe queue containing the parameters to explore.
    :type parameters_queue: Queue
    :param return_queue: multi-processing safe queue where to add the returned values?
    :type return_queue: Queue
    :param function: function to run over parameters multiple times.
    :type function: function
    :param use_signature: save only the first parameter instead of every parameters.
    :type use_signature: bool
    :param save_folder: path to folder where to save results.
    :type save_folder: str
    """
    while True:
        # While there are still parameters to explore, continue:
        try:
            # TODO: update timeout parameter.
            # If block=False, thread might exit while there are still parameters in the queue.
            parameters = parameters_queue.get(block=True, timeout=10)
        except Empty:
            # The queue is emptied, we can exit.
            return "QUEUE EMPTIED"
        returned_data = function(device, *parameters)  # run the function over the parameters fetched.
        if use_signature:
            # Don't save all the parameters, only a 'signature' (the first parameter passed):
            return_queue.put([parameters[0], returned_data])
        else:
            return_queue.put([parameters, returned_data])
        if save_folder is not None:
            # Intermediate save after each function run. 
            save_path = save_folder + os.sep + 'tempsave_{0}.sr'.format(parameters[0])
            with open(save_path, 'wb') as save_file:
                pk.dump(returned_data, save_file)

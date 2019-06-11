""" Tools for models, validation, and training.

The Model class provides basic tools that can be used for all kinds of neural networks. It can (and should) be
customised by inheriting from it and overriding its internal functions.
"""
import numpy as np
import os
import time


import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.checkpoint as chk

import mltools.logging as log

LOGTAG = "MODL"


class Model:
    """Wraps a PyTorch model and provides training, validation, and logging methods.

    This base implementation acts as a basic neural network built over :class:`torch.nn.Sequential`.

    Internal functions (marked by a leading underscore) can be overridden by more specific models architectures.

    :var:`training_log` is used to record every epochs of training individually and every validation, all sequentially.
    Entries will be of the form '(X, e, epoch_metrics)' where X is 'TRAIN' or 'EVAL' (depending if the metrics were
    calculated on the training or validation set), e is the last epoch number, and epochs_metrics a dictionnary of
    all the metrics calculated for this model.

    :param model_description: ordered dictionary containing the `torch.nn.Module` describing the model.
    :type model_description: OrderedDict[str, torch.nn.Module]
    :param train_loader: data loader for training.
    :type train_loader: Set(torch.utils.data.DataLoader, FastLoader)
    :param validation_loader: data loader for validation.
    :type validation_loader: Set(torch.utils.data.DataLoader, FastLoader)
    """
    def __init__(self, model_description, train_loader, validation_loader, checkpointing=None, saving=(None, None, None), tb_writer=None):
        self.model = self._build_model(model_description)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.device = torch.device('cuda:0')  # By default run on GPU 0.
        self.training_log = []
        self.epochs_metrics = []
        self.checkpointing = checkpointing
        self.model_name, self.saving_path, self.saving_frequency = saving
        self.tb_writer = tb_writer

    def _build_model(self, model_description):
        """Build model from model description.

        By default uses :class:`torch.nn.Sequential`. Override this function to change how the model is built.

        :param model_description: ordered dictionary containing the `torch.nn.Module` describing the model.
        :type model_description: OrderedDict[str, torch.nn.Module]
        :return: PyTorch model.
        :rtype: torch.nn.Sequential
        """
        return nn.Sequential(model_description)

    def to(self, device):
        """Push model to a device. All future tensors will also be pushed to this device.

        :param device: device on which to push the model.
        :type device: torch.device
        :return: the model itself, pushed to the device.
        :rtype: Model
        """
        self.device = device
        self.model = self.model.to(device)
        return self

    def parameters_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        parameters = sum([np.prod(p.size()) for p in model_parameters])
        return parameters

    def train(self, epochs, optimizer_description=(opt.Adam, {'lr': 1e-4}), scheduler_description=None,
              loss_criterion=nn.CrossEntropyLoss()):
        """Train the model for a fixed number of epochs, using the optimizer, scheduler, and loss criterion provided,
        and running evaluation according to the frequency selected.

        Optimizer and scheduler are created from their 'descriptions'. A description is a tuple first containing a base
        class from which to derive the optimizer/scheduler, then containing a dictionnary of parameters that will be
        unpacked and passed to the base class. The model parameters will always be passed as the first argument.

        The general organization is as follows:
            - :func:`_setup_optimization` is used to create the optimizer and scheduler from their descriptions;
            - For each epoch,
                - For each batch,
                    - We get the (input, target_output) batch from the train_loader and push it to the device;
                    - :func:`process` is used to infer model_output from input;
                    - :func:`_train` can be overridden to perform additionnal backward operations;
                    - :func:`_evaluate_loss` is called to evaluate the batch_loss using loss_criterion;
                    - :func:`_batch_metrics` is used to calculate the batch metrics from;
                    - :func:`_end_batch` is called at the very end of the batch to actually perform the backward step.
                - :func:`_epoch_metrics` combine the metrics from the batches into metrics for the epoch;
                - The epoch metrics are appended to the training_log;
                - :func:`_end_batch` is called at the very end of the epoch to step the scheduler.
                - If it is time, validation is run once.

        :param epochs: how many epochs to run.
        :type epochs: int
        :param optimizer_description: tuple of base class to use for optimizer and arguments to pass.
        :type optimizer_description: (torch.optim.Optimizer, dict)
        :param scheduler_description: tuple of base class to use for scheduler and arguments to pass.
        :type scheduler_description: (torch.optim.Scheduler, dict)
        :param loss_criterion: loss criterion to use to evaluate loss.
        :type loss_criterion: torch.nn._Loss
        :param evaluation_frequency: how often to evaluate model (in epochs).
        :type evaluation_frequency: int
        """
        # Setup optimizer and scheduler.
        optimizer, scheduler = self._setup_optimization(optimizer_description, scheduler_description)

        epoch_processing = [time.time(), None]
        for e in range(1, epochs + 1):
            if e!= 1:
                log.log("   EPOCH {0}/{1}, last epoch processed in {2}s:"
                        .format(e, epochs, epoch_processing[1] - epoch_processing[0]), LOGTAG, log.Level.INFO)
            else:
                log.log("   EPOCH {0}/{1}:"
                        .format(e, epochs), LOGTAG, log.Level.INFO)

            epoch_processing[0] = time.time()

            batches_metrics = []  # Metrics of every batch in the epoch.
            self.model.train()
            data_loading = [time.time(), None, None]
            for i, (input, target_output) in enumerate(self.train_loader):
                #if True:
                #    self._end_epoch(None, None, None, None, None, True, nn.CrossEntropyLoss())
                # Do one batch:
                data_loading[2] = time.time()
                # Push the data to GPU asynchronously.
                input, target_output = input.to(self.device, non_blocking=True), \
                                       target_output.to(self.device, non_blocking=True)

                input, target_output = self._process_data(input, target_output)
                if i != 0:
                    log.log('       Batch {0}/{1} (loaded in {2:.4f}s, '
                            'last batch forward in {3:.4f}s, and backward in {4:.4f}s)'
                            .format(i, len(self.train_loader), data_loading[2] - data_loading[1],
                                    forward_pass[1] - forward_pass[0], backward_pass[1] - backward_pass[0]),
                            LOGTAG, log.Level.INFO, temporary=True)
                else:
                    log.log('       Batch {0}/{1} (loaded in {0}s)'
                            .format(i, len(self.train_loader), data_loading[2] - data_loading[0]),
                            LOGTAG, log.Level.INFO, temporary=True)

                # Forward pass: infer output from input.
                forward_pass = [time.time(), None]
                model_output = self.infer(input)
                forward_pass[1] = time.time()


                # Backward pass: calculate loss and other metrics, and descend gradient.
                backward_pass = [time.time(), None]
                self._train(input, target_output, model_output)  # Perform additional computations on input/output.
                batch_loss = self._evaluate_loss(input, model_output, target_output, loss_criterion)
                batches_metrics.append(self._batch_metrics(input, model_output, target_output, batch_loss))
                self._end_batch(batch_loss, optimizer, input, model_output, target_output)  # Ascend gradient.
                backward_pass[1] = time.time()
                data_loading[1] = time.time()

            # Compute epoch metrics from batches metrics and log them.
            epoch_metrics = self._epoch_metrics('train', e, batches_metrics)
            self.epochs_metrics.append(epoch_metrics)
            self.training_log.append(('TRAIN', e, epoch_metrics))

            # Wrap-up everything, check validation performance, and step scheduler.
            epoch_validation_metrics = self.validate(loss_criterion=loss_criterion, epoch=e)
            self._end_epoch(e, epoch_metrics, epoch_validation_metrics, optimizer, scheduler, loss_criterion=loss_criterion)

            if self.saving_frequency is not None and e % self.saving_frequency == 0:
                state_dict_file_name = "state_{0}.pt".format(self.model_name)
                state_dict_file_path = os.path.join(self.saving_path, state_dict_file_name)
                log.log("Checkpoint! saving model state to {0}...".format(state_dict_file_path), LOGTAG, log.Level.INFO)
                torch.save(self.model.state_dict(), state_dict_file_path)

            epoch_processing[1] = time.time()

    def validate(self, loss_criterion=nn.CrossEntropyLoss(), epoch=-1):
        """Validate the model over the validation set and compute associated metrics.

        :param loss_criterion: loss criterion to use to evaluate loss.
        :type loss_criterion: torch.nn._Loss
        :param epoch: epoch at which the evaluation is ran (for logging purposes).
        :type: int
        """
        # Switch model to evaluation mode and disable gradients computation to save time.
        self.model.eval()
        with torch.no_grad():
            batches_metrics = []  # Metrics from every batch.
            for input, target_output in self.validation_loader:
                # We go over the validation set:
                input, target_output = input.to(self.device, non_blocking=True),\
                                       target_output.to(self.device, non_blocking=True)
                input, target_output = self._process_data(input, target_output)
                model_output = self.infer(input)
                batch_loss = self._evaluate_loss(input, model_output, target_output, loss_criterion)
                batches_metrics.append(self._batch_metrics(input, model_output, target_output, batch_loss))
            # Compute the metrics over all the set and then log them.
            evaluation_metrics = self._epoch_metrics('eval', epoch, batches_metrics)
            self.training_log.append(('EVAL', epoch, evaluation_metrics))
        return evaluation_metrics

    def infer(self, input):
        """Run inference with the model.

        Override this function to change how inference is made.

        :param input: input to feed the model with.
        :type input: torch.Tensor
        :return: prediction of the model.
        :rtype: torch.Tensor
        """
        if self.checkpointing is not None:
            return chk.checkpoint_sequential(self.model, self.checkpointing, input)
        else:
            return self.model(input)

    def _setup_optimization(self, optimizer_description, scheduler_description):
        """Setup optimizer and scheduler from their descriptions.

        Override to change how the optimizer and scheduler are created.
        While returning a 'None' scheduler is acceptable, a proper optimizer will more or less always be needed, unless
        you modify heavily :func:`_end_batch` as well.

        :param optimizer_description: tuple of base class to use for optimizer and arguments to pass.
        :type optimizer_description: (torch.optim.Optimizer, dict)
        :param scheduler_description: tuple of base class to use for scheduler and arguments to pass.
        :type scheduler_description: (torch.optim.Scheduler, dict)
        :return: optimizer and scheduler.
        :rtype: (torch.optim.Optimizer, torch.optim.Scheduler)
        """
        # Call the base optimizer class with the model parameters and the unpacked arguments.
        optimizer = optimizer_description[0](self.model.parameters(), **optimizer_description[1])
        if scheduler_description is not None:
            # Call the base scheduler class with the model parameters and the unpacked arguments:
            scheduler = scheduler_description[0](optimizer, **scheduler_description[1])
        else:
            # Default to no scheduler:
            scheduler = None
        return optimizer, scheduler

    def _process_data(self, input, target_output):
        return input, target_output

    def _train(self, input, target_output, model_output):
        """By default, this does nothing.

        Override to add behaviors on the backward pass, after inference, but before the weights are updated.

        :param input: batch of inputs to the model.
        :type input: torch.Tensor
        :param target_output: batch of target outputs.
        :type target_output: torch.Tensor
        :param model_output: batch of outputs inferred by the model.
        :type model_output: torch.Tensor
        """
        return

    def _evaluate_loss(self, input, model_output, target_output, loss_criterion):
        """Evaluate the loss for a given batch.

        Override to change loss behavior. It should still return a loss on which :func:`backward` can be used, unless
        you modify :func:`_end_batch` as well.

        :param input: batch of inputs to the model.
        :type input: torch.Tensor
        :param model_output:
        :param model_output: batch of outputs inferred by the model.
        :type model_output: torch.Tensor
        :param target_output: batch of target outputs.
        :type target_output: torch.Tensor
        :param loss_criterion: loss criterion to use to evaluate loss.
        :type loss_criterion: torch.nn._Loss
        :return: evaluation of the loss over the batch.
        :rtype: torch.Tensor
        """
        loss_criterion = loss_criterion.to(self.device)  # Push the loss to the GPU.
        return loss_criterion(model_output, target_output)

    def _batch_metrics(self, input, model_output, target_output, batch_loss):
        """Compute the metrics for the batch.

        Override to add/change metrics computation. It should always return a dictionnary with the metrics as entries.
        You will want to modify :func:`_epoch_metrics` as well to change how the batches metrics are processed.

        :param input: batch of inputs to the model.
        :type input: torch.Tensor
        :param model_output: batch of outputs inferred by the model.
        :type model_output: torch.Tensor
        :param target_output: batch of target outputs.
        :type target_output: torch.Tensor
        :param batch_loss: evaluation of the loss over the batch.
        :type batch_loss: torch.Tensor
        :return:
        """
        return {'length': model_output.size(0), 'loss': batch_loss.detach().item()}

    def _end_batch(self, batch_loss, optimizer, input, model_output, target_output):
        """Wrap-up the batch by zeroing the gradients and updating the weights.

        Override to alter operations done at the end of every batch.

        :param batch_loss: evaluation of the loss over the batch.
        :type batch_loss: torch.Tensor
        :param optimizer: optimizer of the model parameters.
        :type optimizer: torch.optim.Optimizer
        """
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    def _epoch_metrics(self, mode, epoch, batches_metrics):
        """Compute the epoch metrics from all of the batches metrics of that epoch.

        Override to add/change metrics computation. It should always return a dictionnary with the metrics as entries.
        You will want to modify :func:`_batch_metrics` as well to change how the batches metrics are calculated.

        :param epoch: current epoch number.
        :type epoch: int
        :param batches_metrics: list of the metrics at every batches of the epoch.
        :type: List[dict]
        :return: metrics of the epoch.
        :rtype: Dict[str:float]
        """
        epoch_metrics = {'loss': sum([batch_metrics['loss'] for batch_metrics in batches_metrics])
                 / sum([batch_metrics['length'] for batch_metrics in batches_metrics])}
        return epoch_metrics

    def _end_epoch(self, epoch, epoch_metrics, epoch_validation_metrics, optimizer, scheduler, verbose=True, loss_criterion=None):
        """Wrap-up the epoch by stepping the scheduler if there is one.

        Override to alter operations done at the very end of every epoch, such as stepping the scheduler, or
        printing some information about the performance of the network.

        :param epoch: current epoch number.
        :type epoch: int
        :param optimizer: optimizer of the model parameters.
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler of the model parameters.
        :type scheduler: torch.optim.Scheduler
        """
        if scheduler is not None:
            scheduler.step()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag='train_loss', scalar_value=epoch_metrics['loss'], global_step=epoch)
            self.tb_writer.add_scalar(tag='val_loss', scalar_value=epoch_validation_metrics['loss'], global_step=epoch)
            for name, param in self.model.named_parameters():
                self.tb_writer.add_histogram(tag=name, values=param.clone().cpu().data.numpy(), global_step=epoch)
                if param.grad is not None:
                    self.tb_writer.add_histogram(tag='{0}_grad'.format(name), values=param.grad.clone().cpu().data.numpy(), global_step=epoch)
        if verbose:
            log.log("   Training loss: {0} -- Validation loss: {1}.".format(epoch_metrics['loss'],
                                                                            epoch_validation_metrics['loss']),
                    LOGTAG, log.Level.INFO)

class Classifier(Model):
    """Provides additionnal metrics over :class:`Model` for classification tasks.

    :param top_predictions: list of k for top-k accuracies to compute.
    :type top_predictions: Tuple(int)
    """
    def __init__(self, model_description, train_loader, validation_loader, checkpointing=None, saving=(None, None, None), tb_writer=None, top_predictions=(1,)):
        self.top_predictions = top_predictions
        super(Classifier, self).__init__(model_description, train_loader, validation_loader, checkpointing, saving, tb_writer)

    def _batch_metrics(self, input, model_output, target_output, batch_loss):
        """On top of the base metrics calculated by :class:`Model`, will calculate top-k accuracies.
        """
        batch_metrics = super(Classifier, self)._batch_metrics(input, model_output, target_output, batch_loss)
        with torch.no_grad():
            max_top_predictions = max(self.top_predictions)
            batch_size = target_output.size(0)

            # We will start by calculating the top-k accuracy with the largest k.
            _, prediction = model_output.topk(max_top_predictions, 1)
            prediction = prediction.t()
            correct = prediction.eq(target_output.view(1, -1).expand_as(prediction))

            results = []
            for k in self.top_predictions:
                # From the top-k with largest k, we can derive all the others by truncating it:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                results.append(correct_k.mul_(100.0 / batch_size))
        batch_metrics['topk'] = [float(r) for r in results]
        return batch_metrics

    def _epoch_metrics(self, mode, epoch, batches_metrics):
        """Process the base parameters and the new top-k accuracies.
        """
        epoch_metrics = super(Classifier, self)._epoch_metrics(mode, epoch, batches_metrics)
        epoch_metrics['topk'] = [sum([batches_metrics[j]['topk'][i] for j in range(len(batches_metrics))])
                                 / len(batches_metrics)
                                 for i in range(len(self.top_predictions))]
        return epoch_metrics

    def _end_epoch(self, epoch, epoch_metrics, epoch_validation_metrics, optimizer, scheduler, verbose=True, loss_criterion=None):
        super(Classifier, self)._end_epoch(epoch, epoch_metrics, epoch_validation_metrics, optimizer, scheduler, False)
        if self.tb_writer is not None:
            for i, kacc in enumerate(epoch_metrics['topk']):
                self.tb_writer.add_scalar(tag='train_top{0}acc'.format(self.top_predictions[i]), scalar_value=kacc, global_step=epoch)
                self.tb_writer.add_scalar(tag='val_top{0}acc'.format(self.top_predictions[i]), scalar_value=epoch_validation_metrics['topk'][i], global_step=epoch)
        if verbose:
            train_accuracy = ', '.join(['top-{0}:{1:.1f}%'.format(k, epoch_metrics['topk'][i])
                                        for i, k in enumerate(self.top_predictions)])
            validation_accuracy = ', '.join(['top-{0}:{1:.1f}%'.format(k, epoch_validation_metrics['topk'][i])
                                             for i, k in enumerate(self.top_predictions)])
            log.log("   Training loss: {0} ({2}) -- Validation loss: {1} ({3})."
                    .format(epoch_metrics['loss'], epoch_validation_metrics['loss'],
                            train_accuracy, validation_accuracy),
                    LOGTAG, log.Level.INFO)

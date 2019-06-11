import copy
import inspect
import pickle as pk
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as torch_activations

import dfatools.dfatorch as dfa
import dfatools.utilities as dfa_util
import mltools.logging as log
import mltools.models as models
import mltools.utilities as util

from collections import OrderedDict

LOGTAG = "DMOD"


class Bottleneck(nn.Linear):
    def __init__(self, bottleneck_size, *args, **kwargs):
        self.bottleneck_size = bottleneck_size
        super(Bottleneck, self).__init__(*args, **kwargs)


class GradientHelper:
    def __init__(self, model):
        self.model = model
        print(self.model.model)
        self.architecture = []
        self.gradients = {}
        self.hooks_registry = []
        for (i, module) in enumerate(list(self.model.model.modules())[1:]):
            if not isinstance(module, dfa.AsymmetricFeedback):
                self.architecture.append(module)
            # if isinstance(module, tuple([c[1] for c in inspect.getmembers(torch_activations, inspect.isclass)
            #                              if c[1] != torch.nn.modules.module.Module])):
                self.hooks_registry.append(module.register_backward_hook(self.hook))

    def hook(self, module, grad_input, grad_output):
        self.gradients[module] = grad_output[0]
        if self.model.enable_weight_filtering and isinstance(module, Bottleneck):
            if not module in self.model.weight_filters:
                self.model.weight_filters[module] = [(torch.rand(grad_input[i].shape) > module.bottleneck_size).to(self.model.device).float() if grad_input[i] is not None else None for i in range(len(grad_input))]
            updated_grad = [grad_input[i] * self.model.weight_filters[module][i] if grad_input[i] is not None else None for i in range(len(grad_input))]
            return tuple(updated_grad)

    def unhook(self):
        for handle in self.hooks_registry:
            handle.remove()


class DFAModel(models.Model):
    def __init__(self, rp_device, bp_device, *args, **kargs):
        self.rp_device = rp_device
        self.bp_device = bp_device
        if self.bp_device is not None:
            model_description, train_loader, validation_loader = copy.deepcopy(args[0]), args[1], args[2]
            self.bp_model = models.Classifier(model_description, train_loader, validation_loader)
            self.bp_model = self.bp_model.to(self.bp_device)
            self.alignments, self.angles = [], []
        super(DFAModel, self).__init__(*args, **kargs)
        self.bp_model.gradient_helper = GradientHelper(self.bp_model)
        self.gradient_helper = GradientHelper(self)
        self.enable_weight_filtering, self.pruning_probability, self.weight_filters = True, 0.5, {}
        self.bp_model.enable_weight_filtering, self.bp_model.pruning_probability, self.bp_model.weight_filters = \
            self.enable_weight_filtering, self.pruning_probability, self.weight_filters

    def _build_model(self, model_description):
        return dfa.AsymmetricSequential(self.rp_device, dfa.build_dfa_from_bp(model_description))

    def _compute_error(self, model_output, target_output):
        one_hot_target_output = util.to_categorical(target_output, model_output.size(1)).to(self.device)
        return F.softmax(model_output, dim=1) - one_hot_target_output

    def _end_batch(self, batch_loss, optimizer, input, model_output, target_output):
        error = self._compute_error(model_output, target_output)
        optimizer.zero_grad()
        self.model.backward(error)
        batch_loss.backward()
        optimizer.step()

    def initialize(self):
        input_sample, _ = next(iter(self.train_loader))
        input_sample = input_sample.to(self.device)
        self.model.build_feedback(input_sample)

    def _end_epoch(self, epoch, epoch_metrics, epoch_validation_metrics, optimizer, scheduler, verbose=True, loss_criterion=None):
        if self.bp_device is not None:
            self.bp_model.model.load_state_dict(self.model.state_dict())
            input_sample, output_sample = next(iter(self.validation_loader))
            input_sample, output_sample = self._process_data(input_sample, output_sample)
            input_sample_dfa, output_sample_dfa = input_sample.to(self.device, non_blocking=True), \
                                                  output_sample.to(self.device, non_blocking=True)
            input_sample_bp, output_sample_bp = input_sample.to(self.bp_device, non_blocking=True), \
                                                output_sample.to(self.bp_device, non_blocking=True)

            model_output_dfa = self.model(input_sample_dfa)
            model_output_bp = self.bp_model.model(input_sample_bp)

            error = self._compute_error(model_output_dfa, output_sample_dfa)

            dfa_loss_crit = loss_criterion.to(self.device)
            bp_loss_crit = loss_criterion.to(self.bp_device)
            dfa_loss = dfa_loss_crit(model_output_dfa, output_sample_dfa)
            bp_loss = bp_loss_crit(model_output_bp, output_sample_bp)
            self.model.zero_grad()
            self.bp_model.model.zero_grad()
            dfa_loss.backward()
            self.model.backward(error)
            bp_loss.backward()

            self.alignments.append([])
            self.angles.append([])

            for bp_module, bp_gradient in self.bp_model.gradient_helper.gradients.items():
                module_id = self.bp_model.gradient_helper.architecture.index(bp_module)
                dfa_module = self.gradient_helper.architecture[module_id]
                bp_gradient = bp_gradient.view(input_sample.shape[0], -1)
                dfa_gradient = self.gradient_helper.gradients[dfa_module].view(input_sample.shape[0], -1)
                cosine_similarity = nn.CosineSimilarity(dim=1)
                angles = cosine_similarity(bp_gradient.to(self.device), dfa_gradient)
                self.angles[-1].append([bp_module, angles])
                angles_stats = [float(angles.mean()), float(angles.std())]
                self.alignments[-1].append([bp_module, angles_stats])

            self.alignments[-1].reverse()
            self.angles[-1].reverse()

            first_layer, offset = True, 0
            for i, module in enumerate(self.bp_model.gradient_helper.architecture[1:]):
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout2d)) and first_layer:
                    offset += 1
                else:
                    first_layer = False
                self.alignments[-1][i - offset][0] = f'{i}_{module.__str__()}'
                self.angles[-1][i - offset][0] = f'{i}_{module.__str__()}'

            if verbose:
                for layer_alignment in self.alignments[-1]:
                    log.log(f'{layer_alignment[0]} -- mean:{layer_alignment[1][0]:.4f}, std:{layer_alignment[1][1]:.4f}', LOGTAG, log.Level.INFO)

            if self.tb_writer is not None:
                for layer_alignment in self.alignments[-1]:
                    self.tb_writer.add_scalar(tag='{0}_angle_mean'.format(layer_alignment[0]), scalar_value=layer_alignment[1][0], global_step=epoch)
                    self.tb_writer.add_scalar(tag='{0}_angle_std'.format(layer_alignment[0]), scalar_value=layer_alignment[1][1], global_step=epoch)

            if self.saving_path is not None:
                alignment_file_path = os.path.join(self.saving_path, f'{self.model_name}_alignment.al')
                angles_file_path = os.path.join(self.saving_path, f'{self.model_name}_angle.ang')
                with open(alignment_file_path, 'wb') as alignment_file:
                    pk.dump(self.alignments, alignment_file)
                with open(angles_file_path, 'wb') as angles_file:
                    pk.dump(self.angles, angles_file)

        super(DFAModel, self)._end_epoch(epoch, epoch_metrics, epoch_validation_metrics, optimizer, scheduler, verbose)

    def prealignment(self, alignment_loader, loss):
        modules_dfa = list(self.model.modules())[1:]
        modules_bp = list(self.bp_model.model.modules())[1:]
        print("DFA Arch.", modules_dfa)
        print("BP Arch.", modules_bp)
        modules_dfa.reverse()
        modules_bp.reverse()
        layer_index, layer_offset, last_layer, first_layer = -1, 0, True, False
        first_init = True
        for module in modules_dfa:
            layer_index += 1
            if isinstance(module, dfa.AsymmetricFeedback):
                layer_offset += 1
                last_layer = False
            if isinstance(module, util.WEIGHT_MODULES):
                log.log(f"Prealigning {module}...", LOGTAG, log.Level.INFO)
                next_module_bp = modules_bp[layer_index - layer_offset]
                next_module_dfa = modules_dfa[layer_index + 1]
                print("bp", next_module_bp)
                print("dfa", next_module_dfa)

                self.bp_model.model.load_state_dict(self.model.state_dict())

                input_sample, output_sample = next(iter(alignment_loader))
                input_sample, output_sample = self._process_data(input_sample, output_sample)
                input_sample_dfa, output_sample_dfa = input_sample.to(self.device, non_blocking=True), \
                                                      output_sample.to(self.device, non_blocking=True)
                input_sample_bp, output_sample_bp = input_sample.to(self.bp_device, non_blocking=True), \
                                                    output_sample.to(self.bp_device, non_blocking=True)

                model_output_dfa = self.model(input_sample_dfa)
                model_output_bp = self.bp_model.model(input_sample_bp)

                error = self._compute_error(model_output_dfa, output_sample_dfa)

                dfa_loss_crit = loss.to(self.device)
                bp_loss_crit = loss.to(self.bp_device)
                dfa_loss = dfa_loss_crit(model_output_dfa, output_sample_dfa)
                bp_loss = bp_loss_crit(model_output_bp, output_sample_bp)
                self.model.zero_grad()
                self.bp_model.model.zero_grad()
                dfa_loss.backward()
                self.model.backward(error)
                bp_loss.backward()

                if isinstance(next_module_dfa, dfa.AsymmetricFeedback):
                    self.gradient_helper.gradients[next_module_dfa] = next_module_dfa.rp

                try:
                    print(self.bp_model.gradient_helper.gradients.keys())
                    gradient_dfa = self.gradient_helper.gradients[next_module_dfa]
                    gradient_bp = self.bp_model.gradient_helper.gradients[next_module_bp]
                    print("Be", gradient_dfa.shape)
                    print("W", module.weight.shape)
                    print("grad", gradient_bp.shape)
                    gradient_dfa = torch.t(gradient_dfa)
                    gradient_bp = torch.t(gradient_bp)

                    weight_transpose = torch.mm(gradient_dfa, torch.pinverse(gradient_bp))
                    print("norm pre-normalization", weight_transpose.norm())
                    weight_transpose = weight_transpose / weight_transpose.norm()
                    print("norm post-normalization", weight_transpose.norm())
                    module.weight.data = torch.t(weight_transpose)
                except:
                    log.log(f"Prealignment of module {module} failed!", LOGTAG, log.Level.ERROR)

class ParallelDFAModel(DFAModel):
    def _end_batch(self, batch_loss, optimizer, input, model_output, target_output):
        one_hot_target_output = util.to_categorical(target_output, model_output.size(1)).to(self.device)
        error = F.softmax(model_output, dim=1) - one_hot_target_output
        optimizer.zero_grad()
        batch_loss.backward()
        self.model.module.backward(error)
        optimizer.step()


class DFAClassifier(DFAModel, models.Classifier):
    def __init__(self, rp_device, bp_device=None, *args, **kargs):
        DFAModel.__init__(self, rp_device, bp_device, *args, **kargs)


class ParallelDFAClassifier(ParallelDFAModel, models.Classifier):
    def __init__(self, rp_device, *args, **kargs):
        DFAModel.__init__(self, rp_device, *args, **kargs)

import logging
import math
from pathlib import Path
import time
from typing import Literal, Optional, TypeAlias

import numpy as np
import torch

from . import checks
from .autopgd_base import APGDAttack, APGDAttack_targeted
from .fab_pt import FABAttack_PT
from .square import SquareAttack
from .state import EvaluationState


AttackName: TypeAlias = Literal['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']


class AutoAttack:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        attacks: Optional[list[AttackName]] = None,
        device: torch.device | str = 'cpu',
        eps: float = 0.3,
        norm: Literal['Linf', 'L1', 'L2'] = 'Linf',
        seed: Optional[int] = None,
        version: Literal['custom', 'plus', 'rand', 'standard'] = 'standard',
    ):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = True
        self.attacks_to_run = attacks
        self.version = version
        self.device = device
        self.logger = logging.getLogger('auto-attack')

        if version in ['standard', 'plus', 'rand'] and (attacks is not None):
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")

        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
            device=self.device, logger=self.logger)

        self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False, device=self.device)

        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)

        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
            logger=self.logger)

        if version in ['standard', 'plus', 'rand']:
            self._set_version(version)

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def run_standard_evaluation(self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        /,
        *,
        batch_size: int = 250,
        state_path: Optional[Path] = None,
    ):
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError("The state was created with a different set of attacks "
                                 "to run. You are probably using the wrong state file.")
            if self.verbose:
                self.logger.debug("Restored state from {}".format(state_path))
                self.logger.debug("Since the state has been restored, **only** "
                                "the adversarial examples from the current run "
                                "are going to be returned.")
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.debug("Created state in {}".format(state_path))

        attacks_to_run = list(filter(lambda attack: attack not in state.run_attacks, self.attacks_to_run))
        if self.verbose:
            self.logger.debug('using {} version including {}.'.format(self.version,
                  ', '.join(attacks_to_run)))
            if state.run_attacks:
                self.logger.debug('{} was/were already run.'.format(', '.join(state.run_attacks)))

        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:batch_size].to(self.device),
                y_orig[:batch_size].to(self.device), bs=batch_size, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:batch_size].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:batch_size].to(self.device), False,
            logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / batch_size))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min( (batch_idx + 1) * batch_size, x_orig.shape[0])

                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx: end_idx] = output
                    correct_batch = y.eq(output)
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': robust_accuracy}
                state.clean_accuracy = robust_accuracy

                if self.verbose:
                    self.logger.debug('initial accuracy: {:.2%}'.format(robust_accuracy))
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': state.clean_accuracy}
                if self.verbose:
                    self.logger.debug('initial clean accuracy: {:.2%}'.format(state.clean_accuracy))
                    self.logger.debug('robust accuracy at the time of restoring the state: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in attacks_to_run:
                self.logger.debug(f'running {attack}')

                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    self.logger.debug('no robust datapoints left, skipping remaining attacks')
                    break

                n_batches = int(np.ceil(num_robust / batch_size))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True

                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    else:
                        raise ValueError('Attack not supported')

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        self.logger.debug('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.debug('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))

            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            state.to_disk(force=True)

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.debug('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.debug('robust accuracy: {:.2%}'.format(robust_accuracy))

        return x_adv, y_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            self.logger.info('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))

        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        /,
        *,
        batch_size: int = 250,
    ):
        if self.verbose:
            self.logger.info('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, batch_size=batch_size)
            adv[c] = (x_adv, y_adv)
            if verbose_indiv:
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=batch_size)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.debug('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))

        return adv

    def _set_version(self, version='standard'):
        if self.verbose:
            self.logger.info('setting parameters for {} version'.format(version))

        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000

        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                self.logger.info('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))

        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

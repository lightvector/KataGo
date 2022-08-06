from typing import Any, Dict, List
import math

from model_pytorch import EXTRA_SCORE_DISTR_RADIUS, Model, compute_gain

import torch
import torch.nn
import torch.nn.functional

def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * torch.nn.functional.log_softmax(pred_logits, dim=dim), dim=dim)

def huber_loss(x, y, delta):
    abs_diff = torch.abs(x - y)
    return torch.where(
        abs_diff > delta,
        (0.5 * delta * delta) + delta * (abs_diff - delta),
        0.5 * abs_diff * abs_diff,
    )

def constant_like(data, other_tensor):
    return torch.tensor(data, dtype=other_tensor.dtype, device=other_tensor.device, requires_grad=False)

class Metrics:
    def __init__(self, batch_size: int, world_size: int, raw_model: Model):
        self.n = batch_size
        self.world_size = world_size
        self.pos_len = raw_model.pos_len
        self.pos_area = raw_model.pos_len * raw_model.pos_len
        self.policy_len = raw_model.pos_len * raw_model.pos_len + 1
        self.value_len = 3
        self.num_td_values = 3
        self.num_futurepos_values = 2
        self.num_seki_logits = 4
        self.scorebelief_len = 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)

        self.score_belief_offset_vector = raw_model.value_head.score_belief_offset_vector
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def state_dict(self):
        return dict(
            moving_unowned_proportion_sum = self.moving_unowned_proportion_sum,
            moving_unowned_proportion_weight = self.moving_unowned_proportion_weight,
        )
    def load_state_dict(self, state_dict: Dict[str,Any]):
        if isinstance(state_dict["moving_unowned_proportion_sum"],torch.Tensor):
            self.moving_unowned_proportion_sum = state_dict["moving_unowned_proportion_sum"].item()
        else:
            self.moving_unowned_proportion_sum = state_dict["moving_unowned_proportion_sum"]
        self.moving_unowned_proportion_weight = state_dict["moving_unowned_proportion_weight"]

    def loss_policy_player_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n, self.policy_len)
        assert target_probs.shape == (self.n, self.policy_len)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return global_weight * weight * loss

    def loss_policy_opponent_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n, self.policy_len)
        assert target_probs.shape == (self.n, self.policy_len)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.15 * global_weight * weight * loss


    def loss_value_samplewise(self, pred_logits, target_probs, global_weight):
        assert pred_logits.shape == (self.n, self.value_len)
        assert target_probs.shape == (self.n, self.value_len)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 1.20 * global_weight * loss

    def loss_td_value_samplewise(self, pred_logits, target_probs, global_weight):
        assert pred_logits.shape == (self.n, self.num_td_values, self.value_len)
        assert target_probs.shape == (self.n, self.num_td_values, self.value_len)
        assert global_weight.shape == (self.n,)
        loss = cross_entropy(pred_logits, target_probs, dim=2) - cross_entropy(torch.log(target_probs + 1.0e-30), target_probs, dim=2)
        return 1.20 * global_weight.unsqueeze(1) * loss

    def loss_td_score_samplewise(self, pred, target, weight, global_weight):
        assert pred.shape == (self.n, self.num_td_values)
        assert target.shape == (self.n, self.num_td_values)
        loss = torch.sum(huber_loss(pred, target, delta = 12.0), dim=1)
        return 0.0004 * global_weight * weight * loss


    def loss_ownership_samplewise(self, pred_pretanh, target, weight, mask, mask_sum_hw, global_weight):
        # This uses a formulation where each batch element cares about its average loss.
        # In particular this means that ownership loss predictions on small boards "count more" per spot.
        # Not unlike the way that policy and value loss are also equal-weighted by batch element.
        assert pred_pretanh.shape == (self.n, 1, self.pos_len, self.pos_len)
        assert target.shape == (self.n, self.pos_len, self.pos_len)
        assert mask.shape == (self.n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (self.n,)
        pred_logits = torch.cat((pred_pretanh, -pred_pretanh), dim=1).view(self.n,2,self.pos_area)
        target_probs = torch.stack(((1.0 + target) / 2.0, (1.0 - target) / 2.0), dim=1).view(self.n,2,self.pos_area)
        loss = torch.sum(cross_entropy(pred_logits, target_probs, dim=1) * mask.view(self.n,self.pos_area), dim=1) / mask_sum_hw
        return 1.5 * global_weight * weight * loss


    def loss_scoring_samplewise(self, pred_scoring, target, weight, mask, mask_sum_hw, global_weight):
        assert pred_scoring.shape == (self.n, 1, self.pos_len, self.pos_len)
        assert target.shape == (self.n, self.pos_len, self.pos_len)
        assert mask.shape == (self.n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (self.n,)

        loss = torch.sum(torch.square(pred_scoring.squeeze(1) - target) * mask, dim=(1,2)) / mask_sum_hw
        # Simple huberlike transform to reduce crazy values
        loss = 4.0 * (torch.sqrt(loss * 0.5 + 1.0) - 1.0)
        return global_weight * weight * loss


    def loss_futurepos_samplewise(self, pred_pretanh, target, weight, mask, mask_sum_hw, global_weight):
        # The futurepos targets extrapolate a fixed number of steps into the future independent
        # of board size. So unlike the ownership above, generally a fixed number of spots are going to be
        # "wrong" independent of board size, so we should just equal-weight the prediction per spot.
        # However, on larger boards often the entropy of where the future moves will be should be greater
        # and also in the event of capture, there may be large captures that don't occur on small boards,
        # causing some scaling with board size. So, I dunno, let's compromise and scale by sqrt(boardarea).
        # Also, the further out targets should be weighted a little less due to them being higher entropy
        # due to simply being farther in the future, so multiply by [1,0.25].
        assert pred_pretanh.shape == (self.n, self.num_futurepos_values, self.pos_len, self.pos_len)
        assert target.shape == (self.n, self.num_futurepos_values, self.pos_len, self.pos_len)
        assert mask.shape == (self.n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (self.n,)
        loss = torch.square(torch.tanh(pred_pretanh) - target) * mask.unsqueeze(1)
        loss = loss * constant_like([1.0,0.25], loss).view(1,2,1,1)
        loss = torch.sum(loss, dim=(1, 2, 3)) / torch.sqrt(mask_sum_hw)
        return 0.25 * global_weight * weight * loss


    def loss_seki_samplewise(self, pred_logits, target, target_ownership, weight, mask, mask_sum_hw, global_weight, is_training, skip_moving_update):
        assert self.num_seki_logits == 4
        assert pred_logits.shape == (self.n, self.num_seki_logits, self.pos_len, self.pos_len)
        assert target.shape == (self.n, self.pos_len, self.pos_len)
        assert target_ownership.shape == (self.n, self.pos_len, self.pos_len)
        assert mask.shape == (self.n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (self.n,)

        owned_target = torch.square(target_ownership)
        unowned_target = 1.0 - owned_target
        unowned_proportion = torch.sum(unowned_target * mask, dim=(1, 2)) / (1.0 + mask_sum_hw)
        unowned_proportion = torch.mean(unowned_proportion * weight)
        if is_training:
            if not skip_moving_update:
                self.moving_unowned_proportion_sum *= 0.998
                self.moving_unowned_proportion_weight *= 0.998
                self.moving_unowned_proportion_sum += unowned_proportion.item()
                self.moving_unowned_proportion_weight += 1.0
            moving_unowned_proportion = self.moving_unowned_proportion_sum / self.moving_unowned_proportion_weight
            seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
        else:
            seki_weight_scale = 7.0

        # Loss for predicting the exact sign of seki points
        sign_pred = pred_logits[:, 0:3, :, :]
        sign_target = torch.stack(
            (
                1.0 - torch.square(target),
                torch.nn.functional.relu(target),
                torch.nn.functional.relu(-target),
            ),
            dim=1,
        )
        loss_sign = torch.sum(cross_entropy(sign_pred, sign_target, dim=1) * mask, dim=(1, 2))

        # Loss for generally predicting points that nobody will own
        neutral_pred = torch.stack(
            (pred_logits[:, 3, :, :], torch.zeros_like(target_ownership)), dim=1
        )
        neutral_target = torch.stack((unowned_target, owned_target), dim=1)
        loss_neutral = torch.sum(cross_entropy(neutral_pred, neutral_target, dim=1) * mask, dim=(1, 2))

        loss = loss_sign + 0.5 * loss_neutral
        loss = loss / mask_sum_hw
        return (global_weight * seki_weight_scale * weight * loss, seki_weight_scale)


    def loss_scoremean_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == (self.n,)
        assert target.shape == (self.n,)
        loss = huber_loss(pred, target, delta = 12.0)
        return 0.0015 * global_weight * weight * loss


    def loss_scorebelief_cdf_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n,self.scorebelief_len)
        assert target_probs.shape == (self.n,self.scorebelief_len)
        pred_cdf = torch.cumsum(torch.nn.functional.softmax(pred_logits, dim=1), dim=1)
        target_cdf = torch.cumsum(target_probs, dim=1)
        loss = torch.sum(torch.square(pred_cdf-target_cdf),axis=1)
        return 0.020 * global_weight * weight * loss

    def loss_scorebelief_pdf_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape == (self.n,self.scorebelief_len)
        assert target_probs.shape == (self.n,self.scorebelief_len)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.020 * global_weight * weight * loss

    def loss_scorestdev_samplewise(self, pred, scorebelief_logits, global_weight):
        assert pred.shape == (self.n,)
        assert scorebelief_logits.shape == (self.n,self.scorebelief_len)
        assert self.score_belief_offset_vector.shape == (self.scorebelief_len,)
        scorebelief_probs = torch.nn.functional.softmax(scorebelief_logits, dim=1)
        expected_score_from_belief = torch.sum(scorebelief_probs * self.score_belief_offset_vector.view(1,-1),dim=1,keepdim=True)
        stdev_of_belief = torch.sqrt(0.001 + torch.sum(
            scorebelief_probs * torch.square(
                self.score_belief_offset_vector.view(1,-1) - expected_score_from_belief
            ),
            dim=1
        ))
        loss = huber_loss(pred, stdev_of_belief, delta = 10.0)
        return 0.001 * global_weight * loss

    def loss_lead_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == (self.n,)
        assert target.shape == (self.n,)
        loss = huber_loss(pred, target, delta = 8.0)
        return 0.0060 * global_weight * weight * loss

    def loss_variance_time_samplewise(self, pred, target, weight, global_weight):
        assert pred.shape == (self.n,)
        assert target.shape == (self.n,)
        loss = huber_loss(pred, target, delta = 50.0)
        return 0.0003 * global_weight * weight * loss


    def loss_shortterm_value_error_samplewise(self, pred, td_value_pred_logits, td_value_target_probs, weight, global_weight):
        td_value_pred_probs = torch.softmax(td_value_pred_logits[:,2,:],axis=1)
        predvalue = (td_value_pred_probs[:,0] - td_value_pred_probs[:,1]).detach()
        realvalue = td_value_target_probs[:,2,0] - td_value_target_probs[:,2,1]
        sqerror = torch.square(predvalue-realvalue)
        loss = huber_loss(pred, sqerror, delta = 0.4)
        return 2.0 * global_weight * weight * loss

    def loss_shortterm_score_error_samplewise(self, pred, td_score_pred, td_score_target, weight, global_weight):
        predscore = td_score_pred[:,2].detach()
        realscore = td_score_target[:,2]
        sqerror = torch.square(predscore-realscore)
        loss = huber_loss(pred, sqerror, delta = 100.0)
        return 0.00002 * global_weight * weight * loss

    def accuracy1(self, pred_logits, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * (torch.argmax(pred_logits,dim=1) == torch.argmax(target_probs,dim=1)))

    def target_entropy(self, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * -torch.sum(target_probs * torch.log(target_probs + 1e-30), dim=-1))

    def square_value(self, value_logits, global_weight):
        return torch.sum(global_weight * torch.square(torch.sum(torch.softmax(value_logits,dim=1) * constant_like([1,-1,0],global_weight), dim=1)))

    # Returns 0.5 times the sum of squared model weights, for each reg group of model weights
    @staticmethod
    def get_model_norms(raw_model):
        reg_dict : Dict[str,List] = {}
        raw_model.add_reg_dict(reg_dict)

        device = reg_dict["normal"][0].device
        dtype = torch.float32

        modelnorm_normal = torch.zeros([],device=device,dtype=dtype)
        modelnorm_normal_gamma = torch.zeros([],device=device,dtype=dtype)
        modelnorm_output = torch.zeros([],device=device,dtype=dtype)
        modelnorm_noreg = torch.zeros([],device=device,dtype=dtype)
        modelnorm_output_noreg = torch.zeros([],device=device,dtype=dtype)
        for tensor in reg_dict["normal"]:
            modelnorm_normal += torch.sum(tensor * tensor)
        for tensor in reg_dict["normal_gamma"]:
            modelnorm_normal_gamma += torch.sum(tensor * tensor)
        for tensor in reg_dict["output"]:
            modelnorm_output += torch.sum(tensor * tensor)
        for tensor in reg_dict["noreg"]:
            modelnorm_noreg += torch.sum(tensor * tensor)
        for tensor in reg_dict["output_noreg"]:
            modelnorm_output_noreg += torch.sum(tensor * tensor)
        modelnorm_normal *= 0.5
        modelnorm_normal_gamma *= 0.5
        modelnorm_output *= 0.5
        modelnorm_noreg *= 0.5
        modelnorm_output_noreg *= 0.5
        return (modelnorm_normal, modelnorm_normal_gamma, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg)

    def get_specific_norms_and_gradient_stats(self,raw_model):
        with torch.no_grad():
            params = {}
            for name, param in raw_model.named_parameters():
                params[name] = param

            stats = {}
            def add_norm_and_grad_stats(name):
                param = params[name]
                if name.endswith(".weight"):
                    fanin = param.shape[1]
                elif name.endswith(".gamma"):
                    fanin = 1
                elif name.endwith(".beta"):
                    fanin = 1
                else:
                    assert False, "unimplemented case to compute stats on parameter"

                # 1.0 means that the average squared magnitude of a parameter in this tensor is around where
                # it would be at initialization, assuming it uses the activation that the model generally
                # uses (e.g. relu or mish)
                param_scale = torch.sqrt(torch.mean(torch.square(param))) / compute_gain(raw_model.activation) * math.sqrt(fanin)
                stats[f"{name}.SCALE_batch"] = param_scale

                # How large is the gradient, on the same scale?
                stats[f"{name}.GRADSC_batch"] = torch.sqrt(torch.mean(torch.square(param.grad))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

                # And how large is the component of the gradient that is orthogonal to the overall magnitude of the parameters?
                orthograd = param.grad - param * (torch.sum(param.grad * param) / (1e-20 + torch.sum(torch.square(param))))
                stats[f"{name}.OGRADSC_batch"] = torch.sqrt(torch.mean(torch.square(orthograd))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

            add_norm_and_grad_stats("blocks.1.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.1.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.1.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.6.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.6.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.6.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.10.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.10.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.10.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.16.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.16.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.16.normactconvq.norm.gamma")

            add_norm_and_grad_stats("policy_head.conv1p.weight")
            add_norm_and_grad_stats("value_head.conv1.weight")
            add_norm_and_grad_stats("intermediate_policy_head.conv1p.weight")
            add_norm_and_grad_stats("intermediate_value_head.conv1.weight")

        return stats

    def metrics_dict_batchwise(
        self,
        raw_model,
        model_output_postprocessed_byheads,
        batch,
        is_training,
        soft_policy_weight_scale,
        value_loss_scale,
        td_value_loss_scales,
        main_loss_scale,
        intermediate_loss_scale,
        intermediate_distill_scale,
    ):
        results = self.metrics_dict_batchwise_single_heads_output(
            raw_model,
            model_output_postprocessed_byheads[0],
            batch,
            is_training=is_training,
            soft_policy_weight_scale=soft_policy_weight_scale,
            value_loss_scale=value_loss_scale,
            td_value_loss_scales=td_value_loss_scales,
            is_intermediate=False
        )
        if main_loss_scale is not None:
            results["loss_sum"] = main_loss_scale * results["loss_sum"]

        if raw_model.get_has_intermediate_head():
            assert len(model_output_postprocessed_byheads) > 1
            if raw_model.training:
                assert intermediate_loss_scale is not None or intermediate_distill_scale is not None
            else:
                if intermediate_loss_scale is None and intermediate_distill_scale is None:
                    intermediate_loss_scale = 1.0

            if intermediate_loss_scale is not None:
                iresults = self.metrics_dict_batchwise_single_heads_output(
                    raw_model,
                    model_output_postprocessed_byheads[1],
                    batch,
                    is_training=is_training,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    is_intermediate=True
                )
                for key,value in iresults.items():
                    if key != "loss_sum":
                        results["I"+key] = value
                results["loss_sum"] = results["loss_sum"] + intermediate_loss_scale * iresults["loss_sum"]
            if intermediate_distill_scale is not None:
                iresults = self.metrics_dict_self_distill(
                    model_output_postprocessed_byheads[0],
                    model_output_postprocessed_byheads[1],
                    batch,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                )
                for key,value in iresults.items():
                    if key != "loss_sum":
                        results["SD"+key] = value
                results["loss_sum"] = results["loss_sum"] + intermediate_distill_scale * iresults["loss_sum"]

        return results


    def metrics_dict_batchwise_single_heads_output(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        is_training,
        soft_policy_weight_scale,
        value_loss_scale,
        td_value_loss_scales,
        is_intermediate,
    ):
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_pretanh,
            pred_scoring,
            futurepos_pretanh,
            seki_logits,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            scorebelief_logits,
        ) = model_output_postprocessed

        input_binary_nchw = batch["binaryInputNCHW"]
        input_global_nc = batch["globalInputNC"]
        target_policy_ncmove = batch["policyTargetsNCMove"]
        target_global_nc = batch["globalTargetsNC"]
        score_distribution_ns = batch["scoreDistrN"]
        target_value_nchw = batch["valueTargetsNCHW"]

        mask = input_binary_nchw[:, 0, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(1,2))

        n = input_binary_nchw.shape[0]
        h = input_binary_nchw.shape[2]
        w = input_binary_nchw.shape[3]

        policymask = torch.cat((mask.view(n,h*w),mask.new_ones((n,1))),dim=1)

        target_policy_player = target_policy_ncmove[:, 0, :]
        target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
        target_policy_opponent = target_policy_ncmove[:, 1, :]
        target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)
        target_policy_player_soft = (target_policy_player + 1e-7) * policymask
        target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
        target_policy_player_soft /= torch.sum(target_policy_player_soft, dim=1, keepdim=True)
        target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
        target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
        target_policy_opponent_soft /= torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_policy_opponent = target_global_nc[:, 28]

        target_value = target_global_nc[:, 0:3]
        target_scoremean = target_global_nc[:, 3]
        target_td_value = torch.stack(
            (target_global_nc[:, 4:7], target_global_nc[:, 8:11], target_global_nc[:, 12:15]), dim=1
        )
        target_td_score = torch.cat(
            (target_global_nc[:, 7:8], target_global_nc[:, 11:12], target_global_nc[:, 15:16]), dim=1
        )
        target_lead = target_global_nc[:, 21]
        target_variance_time = target_global_nc[:, 22]
        global_weight = target_global_nc[:, 25]
        target_weight_ownership = target_global_nc[:, 27]
        target_weight_lead = target_global_nc[:, 29]
        target_weight_futurepos = target_global_nc[:, 33]
        target_weight_scoring = target_global_nc[:, 34]

        target_score_distribution = score_distribution_ns / 100.0

        target_ownership = target_value_nchw[:, 0, :, :]
        target_seki = target_value_nchw[:, 1, :, :]
        target_futurepos = target_value_nchw[:, 2:4, :, :]
        target_scoring = target_value_nchw[:, 4, :, :] / 120.0

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :],
            target_policy_opponent,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :],
            target_policy_player_soft,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :],
            target_policy_opponent_soft,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        loss_value = self.loss_value_samplewise(
            value_logits, target_value, global_weight
        ).sum()

        loss_td_value_unsummed = self.loss_td_value_samplewise(
            td_value_logits, target_td_value, global_weight
        )
        assert self.num_td_values == 3
        loss_td_value1 = loss_td_value_unsummed[:,0].sum()
        loss_td_value2 = loss_td_value_unsummed[:,1].sum()
        loss_td_value3 = loss_td_value_unsummed[:,2].sum()

        loss_td_score = self.loss_td_score_samplewise(
            pred_td_score, target_td_score, target_weight_ownership, global_weight
        ).sum()

        loss_ownership = self.loss_ownership_samplewise(
            ownership_pretanh,
            target_ownership,
            target_weight_ownership,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_scoring = self.loss_scoring_samplewise(
            pred_scoring,
            target_scoring,
            target_weight_scoring,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_futurepos = self.loss_futurepos_samplewise(
            futurepos_pretanh,
            target_futurepos,
            target_weight_futurepos,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        (loss_seki,seki_weight_scale) = self.loss_seki_samplewise(
            seki_logits,
            target_seki,
            target_ownership,
            target_weight_ownership,
            mask,
            mask_sum_hw,
            global_weight,
            is_training,
            skip_moving_update=is_intermediate,
        )
        loss_seki = loss_seki.sum()
        seki_weight_scale = seki_weight_scale.sum() if not isinstance(seki_weight_scale,float) else seki_weight_scale
        loss_scoremean = self.loss_scoremean_samplewise(
            pred_scoremean,
            target_scoremean,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_cdf = self.loss_scorebelief_cdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_pdf = self.loss_scorebelief_pdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorestdev = self.loss_scorestdev_samplewise(
            pred_scorestdev,
            scorebelief_logits,
            global_weight,
        ).sum()
        loss_lead = self.loss_lead_samplewise(
            pred_lead,
            target_lead,
            target_weight_lead,
            global_weight,
        ).sum()
        loss_variance_time = self.loss_variance_time_samplewise(
            pred_variance_time,
            target_variance_time,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_value_error = self.loss_shortterm_value_error_samplewise(
            pred_shortterm_value_error,
            td_value_logits,
            target_td_value,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_score_error = self.loss_shortterm_score_error_samplewise(
            pred_shortterm_score_error,
            pred_td_score,
            target_td_score,
            target_weight_ownership,
            global_weight,
        ).sum()

        loss_sum = (
            loss_policy_player
            + loss_policy_opponent
            + loss_policy_player_soft * soft_policy_weight_scale
            + loss_policy_opponent_soft * soft_policy_weight_scale
            + loss_value * value_loss_scale
            + loss_td_value1 * td_value_loss_scales[0]
            + loss_td_value2 * td_value_loss_scales[1]
            + loss_td_value3 * td_value_loss_scales[2]
            + loss_td_score
            + loss_ownership
            + loss_scoring
            + loss_futurepos
            + loss_seki
            + loss_scoremean
            + loss_scorebelief_cdf
            + loss_scorebelief_pdf
            + loss_scorestdev
            + loss_lead
            + loss_variance_time
            + loss_shortterm_value_error
            + loss_shortterm_score_error
        )

        policy_acc1 = self.accuracy1(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        square_value = self.square_value(value_logits, global_weight)

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "vloss_sum": loss_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdsloss_sum": loss_td_score,
            "oloss_sum": loss_ownership,
            "sloss_sum": loss_scoring,
            "fploss_sum": loss_futurepos,
            "skloss_sum": loss_seki,
            "smloss_sum": loss_scoremean,
            "sbcdfloss_sum": loss_scorebelief_cdf,
            "sbpdfloss_sum": loss_scorebelief_pdf,
            "sdregloss_sum": loss_scorestdev,
            "leadloss_sum": loss_lead,
            "vtimeloss_sum": loss_variance_time,
            "evstloss_sum": loss_shortterm_value_error,
            "esstloss_sum": loss_shortterm_score_error,
            "loss_sum": loss_sum,
            "pacc1_sum": policy_acc1,
            "vsquare_sum": square_value,
        }

        if is_intermediate:
            return results
        else:
            weight = global_weight.sum()
            nsamples = int(global_weight.shape[0])
            policy_target_entropy = self.target_entropy(
                target_policy_player,
                target_weight_policy_player,
                global_weight,
            )
            soft_policy_target_entropy = self.target_entropy(
                target_policy_player_soft,
                target_weight_policy_player,
                global_weight,
            )

            (modelnorm_normal, modelnorm_normal_gamma, modelnorm_output, modelnorm_noreg, modelnorm_output_noreg) = self.get_model_norms(raw_model)

            extra_results = {
                "wsum": weight * self.world_size,
                "nsamp": nsamples * self.world_size,
                "ptentr_sum": policy_target_entropy,
                "ptsoftentr_sum": soft_policy_target_entropy,
                "sekiweightscale_sum": seki_weight_scale * weight,
                "norm_normal_batch": modelnorm_normal,
                "norm_normal_gamma_batch": modelnorm_normal_gamma,
                "norm_output_batch": modelnorm_output,
                "norm_noreg_batch": modelnorm_noreg,
                "norm_output_noreg_batch": modelnorm_output_noreg,
            }
            for key,value in extra_results.items():
                results[key] = value
            return results


    def metrics_dict_self_distill(
        self,
        model_output_postprocessed_main,
        model_output_postprocessed_inter,
        batch,
        soft_policy_weight_scale,
        value_loss_scale,
        td_value_loss_scales,
    ):
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_pretanh,
            pred_scoring,
            futurepos_pretanh,
            _seki_logits,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            _pred_shortterm_value_error,
            _pred_shortterm_score_error,
            scorebelief_logits,
        ) = model_output_postprocessed_inter

        model_output_postprocessed_main = tuple(tensor.detach() for tensor in model_output_postprocessed_main)
        (
            target_policy_logits,
            target_value_logits,
            target_td_value_logits,
            target_pred_td_score,
            target_ownership_pretanh,
            target_pred_scoring,
            target_futurepos_pretanh,
            _target_seki_logits,
            target_pred_scoremean,
            _target_pred_scorestdev,
            target_pred_lead,
            target_pred_variance_time,
            _target_pred_shortterm_value_error,
            _target_pred_shortterm_score_error,
            target_scorebelief_logits,
        ) = model_output_postprocessed_main

        input_binary_nchw = batch["binaryInputNCHW"]
        target_global_nc = batch["globalTargetsNC"]

        target_policy_probs = torch.nn.functional.softmax(target_policy_logits, dim=2)
        target_policy_player = target_policy_probs[:, 0, :]
        target_policy_opponent = target_policy_probs[:, 1, :]
        target_policy_player_soft = target_policy_probs[:, 2, :]
        target_policy_opponent_soft = target_policy_probs[:, 3, :]

        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_policy_opponent = target_global_nc[:, 28]

        target_value = torch.nn.functional.softmax(target_value_logits, dim=1)
        target_scoremean = target_pred_scoremean
        target_td_value = torch.nn.functional.softmax(target_td_value_logits, dim=2)
        target_td_score = target_pred_td_score

        target_lead = target_pred_lead
        target_variance_time = target_pred_variance_time
        global_weight = target_global_nc[:, 25]
        target_weight_ownership = target_global_nc[:, 27]
        target_weight_lead = target_global_nc[:, 29]
        target_weight_futurepos = target_global_nc[:, 33]
        target_weight_scoring = target_global_nc[:, 34]

        target_score_distribution = torch.nn.functional.softmax(target_scorebelief_logits, dim=1)

        target_ownership = torch.tanh(target_ownership_pretanh).squeeze(1)
        target_futurepos = torch.tanh(target_futurepos_pretanh).squeeze(1)
        target_scoring = target_pred_scoring.squeeze(1)

        mask = input_binary_nchw[:, 0, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(1,2))

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :],
            target_policy_opponent,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :],
            target_policy_player_soft,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :],
            target_policy_opponent_soft,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        loss_value = self.loss_value_samplewise(
            value_logits, target_value, global_weight
        ).sum()

        loss_td_value_unsummed = self.loss_td_value_samplewise(
            td_value_logits, target_td_value, global_weight
        )
        assert self.num_td_values == 3
        loss_td_value1 = loss_td_value_unsummed[:,0].sum()
        loss_td_value2 = loss_td_value_unsummed[:,1].sum()
        loss_td_value3 = loss_td_value_unsummed[:,2].sum()

        loss_td_score = self.loss_td_score_samplewise(
            pred_td_score, target_td_score, target_weight_ownership, global_weight
        ).sum()

        loss_ownership = self.loss_ownership_samplewise(
            ownership_pretanh,
            target_ownership,
            target_weight_ownership,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_scoring = self.loss_scoring_samplewise(
            pred_scoring,
            target_scoring,
            target_weight_scoring,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_futurepos = self.loss_futurepos_samplewise(
            futurepos_pretanh,
            target_futurepos,
            target_weight_futurepos,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_scoremean = self.loss_scoremean_samplewise(
            pred_scoremean,
            target_scoremean,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_cdf = self.loss_scorebelief_cdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_pdf = self.loss_scorebelief_pdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorestdev = self.loss_scorestdev_samplewise(
            pred_scorestdev,
            target_scorebelief_logits, # pred_scorestdev chases stdev of MAIN head's score belief
            global_weight,
        ).sum()
        loss_lead = self.loss_lead_samplewise(
            pred_lead,
            target_lead,
            target_weight_lead,
            global_weight,
        ).sum()
        loss_variance_time = self.loss_variance_time_samplewise(
            pred_variance_time,
            target_variance_time,
            target_weight_ownership,
            global_weight,
        ).sum()

        loss_sum = (
            loss_policy_player
            + loss_policy_opponent
            + loss_policy_player_soft * soft_policy_weight_scale
            + loss_policy_opponent_soft * soft_policy_weight_scale
            + loss_value * value_loss_scale
            + loss_td_value1 * td_value_loss_scales[0]
            + loss_td_value2 * td_value_loss_scales[1]
            + loss_td_value3 * td_value_loss_scales[2]
            + loss_td_score
            + loss_ownership
            + loss_scoring
            + loss_futurepos
            + loss_scoremean
            + loss_scorebelief_cdf
            + loss_scorebelief_pdf
            + loss_scorestdev
            + loss_lead
            + loss_variance_time
        )

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "vloss_sum": loss_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdsloss_sum": loss_td_score,
            "oloss_sum": loss_ownership,
            "sloss_sum": loss_scoring,
            "fploss_sum": loss_futurepos,
            "smloss_sum": loss_scoremean,
            "sbcdfloss_sum": loss_scorebelief_cdf,
            "sbpdfloss_sum": loss_scorebelief_pdf,
            "sdregloss_sum": loss_scorestdev,
            "leadloss_sum": loss_lead,
            "vtimeloss_sum": loss_variance_time,
            "loss_sum": loss_sum,
        }

        return results

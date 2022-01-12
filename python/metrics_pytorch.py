from typing import Any, Dict, List

from model_pytorch import EXTRA_SCORE_DISTR_RADIUS, Model

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
    def __init__(self, batch_size: int, model: Model):
        self.n = batch_size
        self.pos_len = model.pos_len
        self.pos_area = model.pos_len * model.pos_len
        self.policy_len = model.pos_len * model.pos_len + 1
        self.value_len = 3
        self.num_td_values = 3
        self.num_futurepos_values = 2
        self.num_seki_logits = 4
        self.scorebelief_len = 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)

        self.score_belief_offset_vector = model.value_head.score_belief_offset_vector
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def state_dict(self):
        return dict(
            moving_unowned_proportion_sum = self.moving_unowned_proportion_sum,
            moving_unowned_proportion_weight = self.moving_unowned_proportion_weight,
        )
    def load_state_dict(self, state_dict: Dict[str,Any]):
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
        loss = cross_entropy(pred_logits, target_probs, dim=2) - cross_entropy(torch.log(target_probs + 1.0e-30), target_probs, dim=2)
        loss = torch.sum(loss * constant_like([0.55,0.55,0.15], loss), dim=1)
        return global_weight * loss

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


    def loss_seki_samplewise(self, pred_logits, target, target_ownership, weight, mask, mask_sum_hw, global_weight, is_training):
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
            self.moving_unowned_proportion_sum *= 0.998
            self.moving_unowned_proportion_weight *= 0.998
            self.moving_unowned_proportion_sum += unowned_proportion
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
        return torch.sum(global_weight * torch.sum(torch.softmax(value_logits,dim=1) * constant_like([1,-1,0],global_weight), dim=1))

    def metrics_dict_batchwise(self,model,model_output_postprocessed,batch,is_training):
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

        target_policy_player = target_policy_ncmove[:, 0, :]
        target_policy_player /= torch.sum(target_policy_player, dim=1, keepdim=True)
        target_policy_opponent = target_policy_ncmove[:, 1, :]
        target_policy_opponent /= torch.sum(target_policy_opponent, dim=1, keepdim=True)

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
        loss_value = self.loss_value_samplewise(
            value_logits, target_value, global_weight
        ).sum()
        loss_td_value = self.loss_td_value_samplewise(
            td_value_logits, target_td_value, global_weight
        ).sum()
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
        )
        loss_seki = loss_seki.sum()
        seki_weight_scale = seki_weight_scale.sum()
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
            + loss_value
            + loss_td_value
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

        weight = global_weight.sum()
        nsamples = int(global_weight.shape[0])
        policy_acc1 = self.accuracy1(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        policy_target_entropy = self.target_entropy(
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        square_value = self.square_value(value_logits, global_weight)

        modelnorm_normal = torch.zeros_like(loss_policy_player)
        modelnorm_output = torch.zeros_like(loss_policy_player)
        modelnorm_noreg = torch.zeros_like(loss_policy_player)
        reg_dict : Dict[str,List] = {}
        model.add_reg_dict(reg_dict)
        for tensor in reg_dict["normal"]:
            modelnorm_normal += torch.sum(tensor * tensor)
        for tensor in reg_dict["output"]:
            modelnorm_output += torch.sum(tensor * tensor)
        for tensor in reg_dict["noreg"]:
            modelnorm_noreg += torch.sum(tensor * tensor)
        modelnorm_normal *= 0.5
        modelnorm_output *= 0.5
        modelnorm_noreg *= 0.5

        return {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "vloss_sum": loss_value,
            "tdvloss_sum": loss_td_value,
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
            "wsum": weight,
            "nsamp": nsamples,
            "pacc1_sum": policy_acc1,
            "ptentr_sum": policy_target_entropy,
            "vsquare_sum": square_value,
            "sekiweightscale_sum": seki_weight_scale * weight,
            "norm_normal_batch": modelnorm_normal,
            "norm_output_batch": modelnorm_output,
            "norm_noreg_batch": modelnorm_noreg,
        }




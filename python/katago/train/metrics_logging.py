import logging
import math
import json
import numpy as np

# _sum metrics dict entries will get reported as a moving average of their values
# _batch metrics dict entries will reported as the average per-batch value over the time since the last log
# All other values will get reported as a total sum across the entire run so far.

def accumulate_metrics(metric_sums, metric_weights, metrics, batch_size, decay, new_weight):
    if decay != 1.0:
        for metric in metric_sums:
            if metric.endswith("_sum"):
                metric_sums[metric] *= decay
                metric_weights[metric] *= decay

    for metric in metrics:
        if metric.endswith("_sum"):
            metric_sums[metric] += metrics[metric] * new_weight
            metric_weights[metric] += batch_size * new_weight
        elif metric.endswith("_batch"):
            metric_sums[metric] += metrics[metric] * new_weight
            metric_weights[metric] += 1 * new_weight
        else:
            metric_sums[metric] += metrics[metric]
            metric_weights[metric] += batch_size

def log_metrics(metric_sums, metric_weights, metrics, metrics_out):
    metrics_to_print = {}
    for metric in metric_sums:
        if metric.endswith("_sum"):
            # Use np.float64 to avoid division by 0 complaint
            metrics_to_print[metric[:-4]] = np.float64(metric_sums[metric]) / metric_weights[metric]
        elif metric.endswith("_batch"):
            # Use np.float64 to avoid division by 0 complaint
            metrics_to_print[metric] = np.float64(metric_sums[metric]) / metric_weights[metric]
            metric_sums[metric] *= 0.001
            metric_weights[metric] *= 0.001
        else:
            metrics_to_print[metric] = metric_sums[metric]
    for metric in metrics:
        if metric not in metric_sums:
            metrics_to_print[metric] = metrics[metric]

    logging.info(", ".join(["%s = %f" % (metric, metrics_to_print[metric]) for metric in metrics_to_print]))
    if metrics_out:
        metrics_out.write(json.dumps(metrics_to_print) + "\n")
        metrics_out.flush()

def clear_metric_nonfinite(metric_sums, metric_weights):
    for metric in metric_sums:
        if not math.isfinite(metric_sums[metric]):
            logging.warning(f"NONFINITE VALUE OF METRIC {metric}, CLEARING IT BACK TO EMPTY")
            metric_sums[metric] = 0.0
            metric_weights[metric] = 0.0

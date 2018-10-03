import logging
import os

import numpy as np
import torch
import utils
# import model.net as net
from model.data_loader import DataLoader

def evaluate(model, loss_fn, data_iterator, params, num_steps):
    """
    Evaluate the model on `num_steps` batches
    :param model:
    :param loss_fn:
    :param data_iterator:
    :param metrics:
    :param params:
    :param num_steps:
    :return:
    """

    # set model to evaluation mode
    model.eval()
    device = torch.device(params.device)

    # summary for current eval loop
    # summ = []
    metric_watcher = utils.MetricCalculator()

    # compute metrics over the dataset
    for ix, batch in enumerate(data_iterator):
        train_batch = batch.babyname.to(device)
        labels_batch = batch.sex.to(device)
        labels_batch.data.sub_(1)

        hidden = model.init_hidden(train_batch.size(0))

        # compute model output and loss
        output_batch, attention, hidden = model(train_batch, hidden)
        loss = loss_fn(output_batch, labels_batch, attention, params)
        metric_watcher.update(output_batch, labels_batch, loss)
        # output_batch = model(train_batch, hidden)
        # loss = loss_fn(output_batch, labels_batch)

        # summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        # summary_batch['loss'] = loss.item()
        # summ.append(summary_batch)

    metric_watcher.calculate_metric()

    # compute mean of all metrics in summary
    # metrics_mean = {metric: np.nanmean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    metrics_string = ("loss: {:05.3f}, acc: {:05.3f}, prec_0: {:05.3f}, prec_1: {:05.3f}, reca_0: {:05.3f}, reca_1: {:05.3f}".format(
        metric_watcher.average_loss,
        metric_watcher.accuracy,
        metric_watcher.precision_0, metric_watcher.precision_1,
        metric_watcher.recall_0, metric_watcher.recall_1
    ))
    logging.info("- Eval metrics: " + metrics_string)

    return metric_watcher.export()





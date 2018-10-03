import argparse
import logging
import os
import pickle

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from tqdm import trange

import utils
import importlib
# import model.bilstm as net
from model.data_loader import DataLoader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/full_version', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/bilstm', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir "
                                                         "containing weights to reload before training")

def train(model, optimizer, loss_fn, data_iterator, params, num_steps):
    """
    Train the model on `num_steps` batches
    :param model:
    :param optimizer:
    :param loss_fn:
    :param data_iterator:
    :param metrics:
    :param params:
    :param num_steps:
    :return:
    """

    model.train()
    device = torch.device(params.device)

    # summary for current training loop and a running average object for loss
    summ = []
    # loss_avg = utils.RunningAverage()
    # acc_avg = utils.RunningAverage()
    metric_watcher = utils.MetricCalculator()

    # Use tqdm for progress bar
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

        # clear previous gradients, compute gradients of all variables w.r.t loss
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)

        # performs updates using calculated gradients
        optimizer.step()


    # compute mean of all metrics in summary
    metric_watcher.calculate_metric()
    metrics_string = "loss: {:05.3f}, acc: {:05.3f}, prec_0: {:05.3f}, prec_1: {:05.3f}, reca_0: {:05.3f}, reca_1: {:05.3f}".format(
        metric_watcher.average_loss,
        metric_watcher.accuracy,
        metric_watcher.precision_0, metric_watcher.precision_1,
        metric_watcher.recall_0, metric_watcher.recall_1
    )
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_data_iterator, val_data_iterator, optimizer, loss_fn, params, model_dir, restore_file=None):

    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(params.num_epochs):
        # scheduler.step()
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = len(train_data_iterator.dataset.examples) // params.batch_size + 1
        train(model, optimizer, loss_fn, train_data_iterator, params, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = len(val_data_iterator.dataset.examples) // params.batch_size + 1
        val_metrics = evaluate(model, loss_fn, val_data_iterator, params, num_steps)

        val_acc = val_metrics['accuracy']
        is_best = val_acc > best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir,
                              epoch=epoch)

        # if best eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save val metrics in a json file in the model directory
        epoch_json_path = os.path.join(model_dir, "metrics_val_e{:02d}_weights.json".format(epoch))
        utils.save_dict_to_json(val_metrics, epoch_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    # json_path = 'params.json'

    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.device != "cpu": torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    babyname_vocab_path = os.path.join(args.model_dir, 'babyname_vocab.pkl')
    with open(babyname_vocab_path, 'wb') as f:
        pickle.dump(data_loader.BABYNAME.vocab, f)

    class_vocab_path = os.path.join(args.model_dir, 'class_vocab.pkl')
    with open(class_vocab_path, 'wb') as f:
        pickle.dump(data_loader.SEX.vocab, f)

    train_data = data_loader.train_ds
    val_data = data_loader.val_ds

    train_iter = data_loader.train_iter
    val_iter = data_loader.val_iter

    # specify the train and val dataset sizes
    params.train_size = len(train_data.examples)
    params.val_size = len(val_data.examples)

    nb_girls = data_loader.SEX.vocab.freqs['girl']
    nb_boys = data_loader.SEX.vocab.freqs['boy']
    params.girl_weight = nb_girls / (nb_girls + nb_boys)
    params.boy_weight = nb_boys / (nb_girls + nb_boys)

    # add vocab size to params
    params.vocab_size = len(data_loader.BABYNAME.vocab)
    params.save(json_path)

    logging.info("- done.")

    # Define the model and optimizer
    device = torch.device(params.device)
    model_module = importlib.import_module('model.{}'.format(params.model))
    model = model_module.Net(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # fetch loss function and metrics
    loss_fn = model_module.loss_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_iter, val_iter, optimizer, loss_fn, params, args.model_dir, args.restore_file)
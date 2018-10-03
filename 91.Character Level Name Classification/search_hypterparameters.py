import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/model',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/full_version',
                    help='Directory containing the dataset')


def launching_training_job(parent_dir, data_dir, job_name, params):
    """

    :param parent_dir:
    :param data_dir:
    :param job_name:
    :param params:
    :return:
    """

    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(
        python=PYTHON, model_dir=model_dir, data_dir=data_dir
    )
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    search_field = args.parent_dir.split("/")[1]
    search_area = ["bilstm"]
    for search_param in search_area:

        params.__dict__[search_field] = search_param
        job_name = "{}_{}".format(search_field, search_param)
        launching_training_job(args.parent_dir, args.data_dir, job_name, params)

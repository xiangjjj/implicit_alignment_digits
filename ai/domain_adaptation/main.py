import tqdm
import argparse

from ai.domain_adaptation.training import load_data_and_train, load_data_and_test
from ai.domain_adaptation.models.MDD import MDD
from ai.domain_adaptation.utils.config import parse_yaml_to_dict
from ai.domain_adaptation.utils import system, logger, randomness

tqdm.tqdm.get_lock().locks = []


def create_model_instance(args, model_summary='default'):
    if args.dataset == 'Office-31':
        width = 1024
        srcweight = 4
    elif args.dataset == 'Office-Home':
        width = 2048
        srcweight = 2
    else:
        srcweight = 1
        width = 128

    grl_config = parse_yaml_to_dict(args.grl_config)
    model_instance = MDD(base_net=args.model_architecture, classifier_width=width, use_gpu=True,
                         class_num=args.class_num,
                         srcweight=srcweight, name=model_summary, classifier_depth=args.classifier_depth,
                         bottleneck_dim=args.bottleneck_dim,
                         disable_dropout=args.disable_dropout, use_batchnorm=args.use_batchnorm,
                         grl_config=grl_config, freeze_backbone=args.freeze_backbone, args=args)
    return model_instance


def print_config_info(args):
    print('model config:')
    print(f'name: {args.name}')
    print(f'src_address: {args.src_address}')
    print(f'tgt_address: {args.tgt_address}')
    print(f'k_shot: {args.k_shot}')


def run():
    system.filter_deprecation_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, help='all sets of configuration parameters',
                        default='../config/sgd_0.001.yml')
    parser.add_argument('--grl_config', type=str, help='all sets of configuration parameters',
                        default='../config/grl_default.yml')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--freeze_backbone',
                        action='store_true', help="whether the backbone is trained")
    parser.add_argument('--scheduler_config', type=str, help='all sets of configuration parameters',
                        default='../config/lr_scheduler.yml')
    parser.add_argument('--source_sample_mode',
                        action='store_true', help="whether to train the source domain with n way k shot sampled tasks")
    parser.add_argument('--target_sample_mode',
                        action='store_true', help="whether to train the target domain with n way k shot sampled tasks")
    parser.add_argument('--eval_mode',
                        action='store_true', help="whether the model is in evaluation mode")
    parser.add_argument('--model_architecture', default='ResNet50', type=str,
                        help='type of model architecture')
    parser.add_argument('--disable_dropout',
                        action='store_true', help="whether the model will disable dropout")
    parser.add_argument('--use_batchnorm',
                        action='store_true', help="whether the model will use batchnorm")
    parser.add_argument('--datasets_dir', type=str,
                        help='directory for all domain adaptation datasets')
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--name', default='default', type=str,
                        help='name of the model')
    parser.add_argument('--load_from', default='none', type=str,
                        help='load checkpoint')
    parser.add_argument('--bottleneck_dim', default=2048, type=int,
                        help="the dim of the bottleneck layer")
    parser.add_argument('--classifier_depth', default=2, type=int,
                        help="the depth of the classifier")
    parser.add_argument('--class_num', default=None, type=int,
                        help='number of classes for the classification task')
    parser.add_argument('--not_save',
                        action='store_true', help="whether to save the trained model")
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--train_steps', default=10000, type=int,
                        help='number of training steps')
    parser.add_argument('--base_learner', default='train_batch', type=str,
                        help="algorithm used for base training, e.g., train_batch")
    parser.add_argument('--eval_interval', default=500, type=int,
                        help="eval interval for tensorboard")
    parser.add_argument('--train_loss', default='total_loss', type=str,
                        help="loss for training, e.g., total_loss, classifier_loss")
    parser.add_argument('--group_name', default='default', type=str,
                        help="group_name in tensorboard")
    parser.add_argument('--machine', default='default', type=str,
                        help='which machine this experiment is running on')
    parser.add_argument('--tensorboard_dir', default='tensorboard', type=str,
                        help='where to store tensorboard logs')
    parser.add_argument('--disable_prompt',
                        action='store_true', help="disable option to add model descriptions from the user")

    # implicit alignment
    parser.add_argument('--self_train',
                        action='store_true', help="whether to use self-training for sampling the target domain")
    parser.add_argument('--yhat_update_freq', type=int, default=20,
                        help='frequency to update the self-training predictions on the target domain')
    parser.add_argument('--self_train_sampler', type=str, default='SelfTrainingVannilaSampler',
                        help='sampler for self training, e.g., SelfTrainingVannilaSampler, SelfTrainingConfidentSampler')
    parser.add_argument('--sample_without_replacement', action='store_true',
                        help="this is only for self-training, to sample the target without replacement")
    parser.add_argument('--mask_classifier',
                        action='store_true', help="whether to mask classifier outputs in sampling mode")
    parser.add_argument('--mask_divergence',
                        action='store_true', help="whether to mask classifier outputs in sampling mode")
    parser.add_argument('--n_way', default=None, type=int,
                        help='number of classes for each classification task')
    parser.add_argument('--k_shot', default=None, type=int,
                        help='number of examples per class per domain, default using all examples available.')

    # digits dataset
    parser.add_argument('--digits', action='store_true', help="use digits dataset")

    # entropy based options
    parser.add_argument('--conditional_entropy_weight', type=float, help='weight for entropy loss', default=1.0)
    parser.add_argument('--label_entropy_weight', type=float, help='weight for entropy loss', default=1.0)
    parser.add_argument('--total_entropy_weight', type=float, help='weight for integration with other losses',
                        default=0)

    # metric-based options
    parser.add_argument('--proto_loss', action='store_true', help='explicit alignment')
    parser.add_argument('--protoloss_weight', type=float, help='weight for protoloss', default=0.1)
    parser.add_argument('--proto_decay_rate', type=float, help='weight for protoloss', default=0.3)
    parser.add_argument('--moving_centroid', action='store_true', help='whether to use moving centroid for alignment')
    parser.add_argument('--proto_curriculum', action='store_true', help='whether to use confidence threshold')
    parser.add_argument('--normalize_metric_space', action='store_true')
    parser.add_argument('--distance', type=str, default='euclidean')
    parser.add_argument('--use_proto', action='store_true', help='whether to use protonet for pseudo-labels')
    parser.add_argument('--use_knn', action='store_true', help='whether to use knn for pseudo-labels')
    parser.add_argument('--normalize_features', action='store_true',
                        help='L2 normalization of the features in the metric space')
    parser.add_argument('--confidence_threshold', type=float, default=None, help='threshold for conditional sampling')

    # dump data for visualization
    parser.add_argument('--dump_model_predictions', action='store_true',
                        help="whether to dump predictions on the target domain to pickle")
    parser.add_argument('--dump_features', action='store_true',
                        help="whether to dump features to pickle")
    parser.add_argument('--dump_pickle_name', type=str, default='default', help='name for pickle dump')

    args = parser.parse_args()

    print_config_info(args)

    # set random seed
    randomness.set_global_random_seed(args.seed)

    model_summary = logger.get_model_summary_str(args)
    model_instance = create_model_instance(args, model_summary)

    model_instance.c_net.load_from_path_if_exists(args.load_from)

    if args.eval_mode is True:
        load_data_and_test(model_instance, args)
    else:
        load_data_and_train(model_instance, args)


def main():
    run()


if __name__ == '__main__':
    main()

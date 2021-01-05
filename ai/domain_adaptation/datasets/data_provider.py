from ai.domain_adaptation.datasets.data_list import ImageList
from ai.domain_adaptation.datasets import digits, transformer
from ai.domain_adaptation.datasets.sampler import N_Way_K_Shot_BatchSampler, TaskSampler
from ai.domain_adaptation.datasets import sampler, image_index
import torch.utils.data as util_data
from random import sample
from ai.domain_adaptation.evaluator import evaluate
from ai.domain_adaptation.trainer import metric


def get_office_or_digits_dataloader(
        images_file_path, args, batch_size, resize_size=256, is_train=True, crop_size=224, is_source=None,
        sample_mode_with_ground_truth_labels=False, sample_mode_with_self_training=False, shuffle_test=False):
    img_transformer = transformer.get_img_transformer(images_file_path, is_train, resize_size, crop_size, args)
    dataset = get_dataset(images_file_path, is_train, img_transformer, args)
    data_sampler = None

    if is_train is not True:  # eval mode
        images_loader = util_data.DataLoader(dataset, batch_size=batch_size, shuffle=False or shuffle_test,
                                             num_workers=1)
    else:  # training mode
        if sample_mode_with_ground_truth_labels is False and sample_mode_with_self_training is False:
            images_loader = util_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        elif sample_mode_with_ground_truth_labels is True:
            images_loader, data_sampler = nway_kshot_dataloader(dataset, args)
            if is_source is False:
                print('warning! you are sampling with ground-truth labels')
        elif sample_mode_with_self_training is True:
            images_loader, data_sampler = self_training_dataloader(dataset, args)
        else:
            raise ValueError('could not create dataloader under the given config')

    return images_loader, data_sampler


def get_dataset(images_file_path, is_train, transformer, args):
    if 'mnist' in images_file_path.lower() or 'svhn' in images_file_path.lower():
        dataset_cls = getattr(digits, images_file_path)
        dataset = dataset_cls(root=args.datasets_dir, train=True, transform=transformer, download=True)
    else:
        dataset = image_index.get_image_files(images_file_path, args, is_train)
        dataset = ImageList(dataset, transform=transformer)
    return dataset


def nway_kshot_dataloader(images, args):
    task_sampler = TaskSampler(set(images.labels), args)
    n_way_k_shot_sampler = N_Way_K_Shot_BatchSampler(images.labels, args.train_steps, task_sampler)
    meta_loader = util_data.DataLoader(images, shuffle=False, batch_sampler=n_way_k_shot_sampler)
    return meta_loader, n_way_k_shot_sampler


def self_training_dataloader(images, args):
    task_sampler = TaskSampler(set(images.labels), args)
    self_train_sampler_cls = getattr(sampler, args.self_train_sampler)
    self_train_sampler = self_train_sampler_cls(args.train_steps, task_sampler, args)
    self_train_dataloader = util_data.DataLoader(images, shuffle=False, batch_sampler=self_train_sampler)
    return self_train_dataloader, self_train_sampler


class DataLoaderManager:
    def __init__(self, args):
        self.args = args
        self.train_source_loader, self.train_source_sampler = get_office_or_digits_dataloader(
            args.src_address, args, batch_size=args.batch_size,
            sample_mode_with_ground_truth_labels=args.source_sample_mode, is_source=True)
        if args.self_train is True:
            self.train_target_loader, self.train_target_sampler = get_office_or_digits_dataloader(
                args.tgt_address, args, batch_size=args.batch_size,
                sample_mode_with_ground_truth_labels=False,
                sample_mode_with_self_training=True,
                is_source=False)
        else:
            self.train_target_loader, self.train_target_sampler = get_office_or_digits_dataloader(
                args.tgt_address, args, batch_size=args.batch_size,
                sample_mode_with_ground_truth_labels=args.target_sample_mode,
                is_source=False)

        self.test_target_loader, self.test_target_sampler = get_office_or_digits_dataloader(
            args.tgt_address, args, batch_size=args.batch_size, is_train=False, is_source=False)
        self.multi_domain_meta_train = False

    def load_images_from_multiple_datasets(self, args):
        current_dataset_name = None
        for d in self.domain_addresses:
            this_dataset_name = d.split('/')[-2]
            if this_dataset_name != current_dataset_name:
                self.dataset_names_dict[this_dataset_name] = [d]
            else:
                self.dataset_names_dict[this_dataset_name].append(d)

            self.domain_loader_dict[f'{this_dataset_name}.{d}'] = get_office_or_digits_dataloader(d, args,
                                                                                                  batch_size=32)
            current_dataset_name = this_dataset_name

    def get_train_source_target_loader(self):
        return self.train_source_loader, self.train_target_loader

    def get_multi_dataset_source_target_loader(self):
        # sample dataset
        sampled_dataset = sample(list(self.dataset_names_dict), 1)[0]

        # sample domain names
        while True:
            source_name, target_name = sample(self.dataset_names_dict[sampled_dataset], 2)
            if source_name == self.args.src_address and target_name == self.args.tgt_address:
                if self.args.exclude_src_tgt_domains:
                    continue
            source_domain = self.domain_loader_dict[f'{sampled_dataset}.{source_name}']
            target_domain = self.domain_loader_dict[f'{sampled_dataset}.{target_name}']
            return source_domain, target_domain

    def get_test_target_loader(self):
        return self.test_target_loader

    def update_self_training_labels(self, model_instance):
        if self.args.use_proto is False and self.args.use_knn is False:
            _, pred_probs = evaluate.evaluate_from_dataloader(
                model_instance, self.get_test_target_loader())
        elif self.args.use_proto is True:
            model_instance.set_train(True)
            protonet = metric.ProtoNet(self.args, encoder=model_instance.c_net.feature_forward)
            onepass_source_loader, onepass_target_loader = self.get_source_target_onepass_loader()
            pred_probs = protonet.predict_from_dataloader(onepass_source_loader, onepass_target_loader)
        elif self.args.use_knn is True:
            knn = metric.KNN(self.args, encoder=model_instance.c_net.feature_forward)
            onepass_source_loader, onepass_target_loader = self.get_source_target_onepass_loader()
            pred_probs = knn.predict_from_dataloader(onepass_source_loader, onepass_target_loader)

        self.train_target_sampler.update_predicted_probs(pred_probs)

    def get_source_target_onepass_loader(self):
        onepass_source_loader, _ = get_office_or_digits_dataloader(
            self.args.src_address, self.args, batch_size=self.args.batch_size, is_train=False, is_source=True,
            shuffle_test=True)
        onepass_target_loader, _ = get_office_or_digits_dataloader(
            self.args.tgt_address, self.args, batch_size=self.args.batch_size, is_train=False, is_source=False,
            shuffle_test=True)
        return onepass_source_loader, onepass_target_loader

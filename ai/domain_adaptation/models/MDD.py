import torch.nn as nn
import ai.domain_adaptation.models.backbone as backbone
from ai.domain_adaptation.models.abstract import AbstractModel
import torch.nn.functional as F
import torch
import numpy as np
from ai.domain_adaptation.models.grl import GradientReverseLayer


class MDDNet(nn.Module, AbstractModel):
    def __init__(self, base_net='ResNet50', use_bottlen=True, bottleneck_dim=1024, classifier_width=1024,
                 class_num=31, classifier_depth=2, name='default', disable_dropout=False, use_batchnorm=False,
                 grl_config=None,
                 freeze_backbone=False, normalize_features=False):
        AbstractModel.__init__(self, name)
        nn.Module.__init__(self)
        self.class_num = class_num
        self.bottleneck_dim = bottleneck_dim
        self.freeze_backbone = freeze_backbone
        self.normalize_features = normalize_features

        # self.base_network = backbone.network_dict[base_net]()
        self.base_network = backbone.DigitFeatures()
        # self.base_network = backbone.SVHNFeatures()

        # init GradientReverseLayer parameterGradientReverseLayers
        GradientReverseLayer.alpha = grl_config.grl.alpha
        GradientReverseLayer.low_value = grl_config.grl.low_value
        GradientReverseLayer.high_value = grl_config.grl.high_value
        GradientReverseLayer.max_iter = grl_config.grl.max_iter

        self.create_f_and_fhat_classifiers(
            bottleneck_dim, classifier_width, class_num, classifier_depth,
            use_dropout=not disable_dropout, use_batchnorm=use_batchnorm)

        self.softmax = nn.Softmax(dim=1)

        # collect parameters
        self.parameter_list = [{"params": self.base_network.parameters(), 'lr_scale': 1},
                               {"params": self.classifier_layer.parameters(), 'lr_scale': 1},
                               {"params": self.classifier_layer_2.parameters(), 'lr_scale': 1}]

    def create_bottleneck_layer(self, use_dropout):
        bottleneck_layer_list = [
            nn.Linear(self.base_network.output_num(), self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU()
        ]
        if use_dropout is True:
            bottleneck_layer_list.append(nn.Dropout(0.5))

        # init
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

    def create_f_and_fhat_classifiers(self, bottleneck_dim, classifier_width, class_num, classifier_depth,
                                      use_dropout=True, use_batchnorm=False):
        self.classifier_layer = nn.Sequential(
            nn.Linear(classifier_width, class_num)
        )

        # adversarial domain classifier
        self.classifier_layer_2 = nn.Sequential(
            nn.Linear(classifier_width, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.initialize_classifiers()

    def create_classifier(self, bottleneck_dim, width, class_num, depth=2, use_dropout=True, use_batchnorm=False):
        layer_list = []
        input_size = bottleneck_dim
        for ith_layer in range(depth - 1):
            layer_list.append(nn.Linear(input_size, width))

            if use_batchnorm is True:
                layer_list.append(nn.BatchNorm1d(width))

            layer_list.append(nn.ReLU())

            if use_dropout is True:
                layer_list.append(nn.Dropout(0.5))

            input_size = width

        layer_list.append(nn.Linear(width, class_num))
        classifier = nn.Sequential(*layer_list)
        return classifier

    def forward(self, inputs):
        features = self.feature_forward(inputs)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        # gradient reversal layer helps fuse the minimax problem into one loss function
        features_adv = GradientReverseLayer.apply(features)
        outputs_adv = self.classifier_layer_2(features_adv)

        return features, outputs, softmax_outputs, outputs_adv

    def feature_forward(self, inputs):
        if self.freeze_backbone is True:
            with torch.no_grad():
                features = self.base_network(inputs)
        else:
            features = self.base_network(inputs)

        return features

    def logits_forward(self, inputs):
        features = self.feature_forward(inputs)
        logits = self.classifier_layer(features)
        return logits

    def initialize_classifiers(self):
        self.xavier_initialization(self.classifier_layer)
        self.xavier_initialization(self.classifier_layer_2)

    def xavier_initialization(self, layers):
        for layer in layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.0)

    def initialize_bottleneck(self):
        for b_layer in self.bottleneck_layer:
            if type(b_layer) == nn.Linear:
                torch.nn.init.xavier_normal_(b_layer.weight)
                b_layer.bias.data.fill_(0.0)


class MDD(object):
    def __init__(self, base_net='ResNet50', classifier_width=1024, class_num=31, use_bottleneck=True, use_gpu=True,
                 srcweight=3,
                 classifier_depth=2, name='default_MDD', bottleneck_dim=2048,
                 disable_dropout=False, use_batchnorm=False, grl_config=None,
                 freeze_backbone=False, args=None):
        self.c_net = MDDNet(base_net, classifier_width=classifier_width,
                            class_num=class_num, name=name,
                            classifier_depth=classifier_depth, bottleneck_dim=bottleneck_dim,
                            disable_dropout=disable_dropout, use_batchnorm=use_batchnorm, grl_config=grl_config,
                            freeze_backbone=freeze_backbone, normalize_features=args.normalize_features)
        self.args = args
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight
        self.epsilon = 1e-7
        self.step_num = 0

        if self.args.proto_loss is True:
            self.s_centroid = torch.zeros(self.class_num, bottleneck_dim).cuda()
            self.t_centroid = torch.zeros(self.class_num, bottleneck_dim).cuda()
            self.proto_loss_function = nn.MSELoss()

    def get_loss(self, inputs, labels_source, step_num=0):  # inputs is a concatenation of source and target data
        class_criterion = nn.CrossEntropyLoss()

        features_both, outputs, _, outputs_adv = self.c_net(inputs)

        # f(x)
        outputs_src = outputs.narrow(0, 0, labels_source.size(0))
        outputs_tgt = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        probs_tgt = F.softmax(outputs_tgt, dim=1)

        # classification loss on source domain
        classifier_loss = class_criterion(outputs_src, labels_source)

        # adversarial domain loss
        if self.args.digits is True:
            # because the features are the concantenation of source and target
            # the ground truth domain labels can be directly inferred from the data dimension
            domain_label = torch.cat([torch.zeros(outputs_src.shape[0]), torch.ones(outputs_tgt.shape[0])]).cuda().long()
            # domain loss
            domain_loss = class_criterion(outputs_adv, domain_label)
            domain_pred = outputs_adv.max(1)[1]
            domain_accuracy = (domain_label == domain_pred).sum().float() / domain_label.size(0)

        # use $f$ as the target for $f'$
        target_adv = outputs.max(1)[1]  # categorical labels from $f$
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))

        # source classification acc
        classifier_acc = (target_adv_src == labels_source).sum().float() / labels_source.size(0)

        self.iter_num += 1

        train_losses = {
            'classifier_loss': classifier_loss,
            'dann': classifier_loss + domain_loss,
        }

        train_loss_stats = {
            'train_acc_source': classifier_acc.item(),
            'domain_acc': domain_accuracy.item(),
            'domain_loss': domain_loss.item()
        }

        return train_losses, train_loss_stats

    def get_prorotype_loss(self, s_feature, t_feature, s_labels, t_probs):
        n, d = s_feature.shape

        # get labels
        t_prob, t_labels = torch.max(t_probs, 1)

        if self.args.proto_curriculum is True:
            threshold = 1 - 1 / (1 + np.exp(self.step_num / 100))
            indices = (t_prob > threshold).nonzero().flatten()
            if len(indices) == 0:
                return torch.zeros(1)[0].cuda()
            t_feature = t_feature[indices]
            t_labels = t_labels[indices]

        # count number of examples in each class
        zeros = torch.zeros(self.class_num).cuda()
        s_n_classes = zeros.scatter_add(
            0,
            s_labels,
            torch.ones_like(s_labels, dtype=torch.float))
        t_n_classes = zeros.scatter_add(
            0,
            t_labels,
            torch.ones_like(t_labels, dtype=torch.float))

        # image count cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.class_num, d).cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.class_num, 1))
        t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.class_num, 1))

        # Moving Centroid
        decay = self.args.proto_decay_rate
        if self.args.moving_centroid is True and self.iter_num > 1:
            s_centroid = (1 - decay) * self.s_centroid + decay * s_centroid
            t_centroid = (1 - decay) * self.t_centroid + decay * t_centroid

        semantic_loss = self.proto_loss_function(s_centroid, t_centroid)

        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()
        return semantic_loss

    def mask_clf_outputs(self, outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source):
        mask = torch.zeros(outputs_src.shape[1])
        mask[labels_source.unique()] = 1
        mask = mask.repeat((outputs_src.shape[0], 1)).cuda()
        outputs_src = outputs_src * mask
        outputs_adv_src = outputs_adv_src * mask
        outputs_adv_tgt = outputs_adv_tgt * mask
        return outputs_src, outputs_adv_src, outputs_adv_tgt

    def predict(self, inputs):
        _, _, softmax_outputs, _ = self.c_net(inputs)
        return softmax_outputs

    def get_features(self, inputs):
        features, _, _, _ = self.c_net(inputs)
        return features

    def get_parameter_list(self):
        c_net_params = self.c_net.parameter_list
        return c_net_params

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def copy_params_from(self, source):
        for this_param, source_param in zip(self.c_net.parameters(), source.c_net.parameters()):
            this_param.data = source_param.detach().clone()

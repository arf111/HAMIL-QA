import timm
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),  # matrix V x h
            nn.Tanh(),
            nn.Linear(self.D, self.K)  # matrix W x tanh(V x h)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, isNorm=True):
        A = self.attention(x)  # N x K

        if len(A.shape) == 4:
            A = torch.transpose(A, 2, 3)
        elif len(A.shape) == 3:
            A = torch.transpose(A, 1, 2) # (bs, K, N)
        else:
            A = torch.transpose(A, 1, 0) # K x N

        if isNorm:
            if len(A.shape) == 4:
                A = torch.softmax(A, dim=3)
            elif len(A.shape) == 3:
                A = torch.softmax(A, dim=2)  # softmax over N,
            else:
                A = self.softmax(A)  # softmax over N,

        return A  # K x N or (bs, K, N)

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, isNorm=True):
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:
            A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            if len(A.shape) == 3:
                A = torch.softmax(A, dim=2)
            else:
                A = self.softmax(A)

        return A  ### K x N

class AttentionMILPseudoBagTier1(nn.Module):
    def __init__(self, encoder_name, block=None, layers=None, block_inplanes=None, spatial_dims=None, n_input_channels=None, num_classes=2):
        super(AttentionMILPseudoBagTier1, self).__init__()
        
        self.encoder_2d = None
        self.L = 512 # feature space dimension
        self.D = 32 # hidden layer dimension
        self.K = 1 # number of attention heads

        if encoder_name == 'resnet':
            self.encoder_2d = timm.create_model('resnet10t', pretrained=True, num_classes=0, in_chans=n_input_channels)

            # # Freeze all layers
            for param in self.encoder_2d.parameters():
                param.requires_grad = False

            # Unfreeze only layer4
            for param in self.encoder_2d.layer4.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError('Only ResNet2D is implemented')

        self.embedding_attention = Attention(L=self.L, D=self.D, K=self.K)

        self.classifier = nn.Linear(self.L*self.K, num_classes)
        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.classifier:
                    nn.init.xavier_normal_(module.weight)

                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, Attention):
                for attention_layer in module.attention:
                    if isinstance(attention_layer, nn.Linear):
                        nn.init.xavier_normal_(attention_layer.weight)
                        if attention_layer.bias is not None:
                            nn.init.constant_(attention_layer.bias, 0.0)

    def forward(self, x):
        # if len of x shape is 5 then add a single dimension as batch in the beginning
        if len(x.shape) == 4:
            x = x.unsqueeze(0) # (bs=1, no_of_instances_in_a_bag, 1, h, w)

        sh = x.shape # (bs, pseudo_bags, no_of_instances_in_a_bag, 1, h, w)
        if len(sh) == 6:
            x = x.reshape(sh[0] * sh[1] * sh[2], sh[3], sh[4], sh[5])
        else:
            x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])     

        x = self.encoder_2d(x) # h, Shape: (no_of_instances_in_a_bag, h=512),

        if type(x) == list:
            x = x[-1]

        assert torch.isfinite(x).all(), "Input to feature extractor contains NaN or Inf values."

        if len(sh) == 6:
            x = x.reshape(sh[0], sh[1], sh[2], -1)
        else:
            x = x.reshape(sh[0], sh[1], -1)
        feature_space = x # Shape: (bs, no_of_instances_in_a_bag, h=512)
        
        # Embedding classifier
        embedding_attention_weights = self.embedding_attention(feature_space)  # (K=1, no_of_instances_in_a_bag)
        if len(sh) == 6:
            embedding_attention_weights_reshaped = embedding_attention_weights.reshape(sh[0], sh[1], sh[2], 1)
        else:
            embedding_attention_weights_reshaped = embedding_attention_weights.reshape(sh[0], sh[1], -1) # (bs, no_of_instances_in_a_bag, 1)
        
        hadamard_weighted_feature_space = torch.mul(embedding_attention_weights_reshaped, feature_space) # (bs, no_of_instances_in_a_bag, L)
        if len(sh) == 6:
            hadamard_weighted_feature_space_sum = torch.sum(hadamard_weighted_feature_space, dim=2) # (bs, L)
        else:
            hadamard_weighted_feature_space_sum = torch.sum(hadamard_weighted_feature_space, dim=1).view(sh[0], -1) # (bs, L)

        embedding_logits = self.classifier(hadamard_weighted_feature_space_sum) # (bs, 1)
        
        return embedding_logits, embedding_attention_weights, None, hadamard_weighted_feature_space, feature_space

    
class AttentionMILPseudoBagTier2(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionMILPseudoBagTier2, self).__init__()
        
        self.L_tier2 = 512
        
        self.D_tier2 = 8 # hidden layer dimension
        self.K_tier2 = 1 # number of attention heads

        self.embedding_attention_tier2 = Attention(L=self.L_tier2, D=self.D_tier2, K=self.K_tier2)

        self.classifier_tier2 = nn.Linear(self.L_tier2 * self.K_tier2, num_classes)

        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.classifier_tier2:
                    nn.init.xavier_normal_(module.weight)

                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, Attention_Gated):
                for attention_layer in module.attention_V:
                    if isinstance(attention_layer, nn.Linear):
                        nn.init.xavier_normal_(attention_layer.weight)
                        if attention_layer.bias is not None:
                            nn.init.constant_(attention_layer.bias, 0.0)
                for attention_layer in module.attention_U:
                    if isinstance(attention_layer, nn.Linear):
                        nn.init.xavier_normal_(attention_layer.weight)
                        if attention_layer.bias is not None:
                            nn.init.constant_(attention_layer.bias, 0.0)

    def forward(self, x):
        # Embedding classifier
        embedding_attention_weights_tier2 = self.embedding_attention_tier2(x)  # (K=1, no_of_instances_in_a_bag)

        if len(embedding_attention_weights_tier2.shape) == 3:
            M_tier2 = torch.bmm(embedding_attention_weights_tier2, x) # (bs, K, L) , motivation: to get the weighted feature space
        else:
            M_tier2 = torch.mm(embedding_attention_weights_tier2, x) # (K=1, L) , motivation: to get the weighted feature space

        embedding_logits_tier2 = self.classifier_tier2(M_tier2) # (K, 1)

        return embedding_logits_tier2.squeeze(1), embedding_attention_weights_tier2


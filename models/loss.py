from torch import nn
import torch

class PixWiseBCELoss(nn.Module):
    """ Custom loss function combining binary classification loss and pixel-wise binary loss
    Args:
        beta (float): weight factor to control weighted sum of two losses
                    beta = 0.5 in the paper implementation
    Returns:
        combined loss
    """
    def __init__(self, beta):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    
    def forward(self, net_mask, net_label, target_mask, target_label):
        # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
        # Target should be the first arguments, otherwise "RuntimeError: the derivative for 'target' is not implemented"
        loss_pixel_map = self.criterion(net_mask, target_mask)
        loss_bce = self.criterion(net_label, target_label)

        loss = self.beta * loss_bce + (1 - self.beta) * loss_pixel_map
        return loss


class LGSCLoss(nn.Module):
    """ Custom loss function combining binary classification loss ,pixel-wise binary loss and tripletLoss
    Args:
        w_reg (float): weight factor to control weighted regression loss
        w_cls (float): weight factor to control weighted classification loss
        w_tri (float): weight factor to control weighted triplet  loss
    Returns:
        combined loss
    """

    def __init__(self, cfg=None):
        super().__init__()           
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.triple_loss = TripletLoss()
        self.header_type = cfg['model']['header_type']
        if self.header_type == 'patch_pixel':  
            self.criterion = nn.BCELoss()
        if self.header_type == 'binary_classification':  
            self.criterion = nn.CrossEntropyLoss()         
        self.w_reg = cfg['train']['loss']['w_reg']
        self.w_cls = cfg['train']['loss']['w_cls']
        self.w_tri = cfg['train']['loss']['w_tri']

    def forward(self, output, target_mask, target_label):
        """
        only reg_loss.backward()
        """
        tri_loss = torch.tensor(0.0)
        cls_loss = torch.tensor(0.0)
        reg_loss = torch.tensor(0.0)
        
        if self.header_type == 'patch_pixel':  
            loss = self.criterion(output, target_mask)  # [N, 1, map_size, map_size] [N, 1, map_size, map_size]
        elif self.header_type == 'binary_classification':  
            loss = self.criterion(output.type(torch.FloatTensor), target_label.squeeze().type(torch.LongTensor)) # [N, 2]  [N] 
        losses = [loss, cls_loss, reg_loss, tri_loss]
        return losses
    
#     def forward(self, net_mask, net_label, target_mask, target_label, features):
#         # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
#         # Target should be the first arguments, otherwise "RuntimeError: the derivative for 'target' is not implemented"
#         loss_pixel_map = self.criterion(net_mask, target_mask)
#         loss_bce = self.criterion(net_label, target_label)
#         loss_tri = 0
#         for feature in features:
#             fea = self.avg_pool(feature).squeeze()
#             loss_tri += self.triple_loss(fea, target_label)
#         loss_tri = loss_tri/len(features)
#         cls_loss = self.w_cls * loss_bce
#         reg_loss = self.w_reg * loss_pixel_map
#         tri_loss = self.w_tri * loss_tri
#         loss = cls_loss + reg_loss + tri_loss
#         losses = [loss, cls_loss, reg_loss, tri_loss]
#         return losses


# Adapted from https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/triplet_loss.py
class TripletLoss(nn.Module):
    """Standard triplet loss
    Calculates the triplet loss of a mini-batch.
    Args (kwargs only):
        margin (float, 0.3): Margin constraint to use in triplet loss. If not provided,
        mine (str): Mining method. Default 'hard'. Supports ['hard', 'all'].
    Methods:
        __call__: Returns loss given features and labels.
    """

    def __init__(self, **kwargs):
        super(TripletLoss, self).__init__()

        self.margin = kwargs.get('margin', 0.3)
        mine = kwargs.get("mine", "hard")

        if mine == "hard":
            self.loss_fn = self.hard_mining
        elif mine == "all":
            self.loss_fn = self.all_mining
        else:
            raise NotImplementedError()

    def forward(self, features, labels):
        """ Returns the triplet loss with either batch hard mining or batch all mining.
        Args:
            features: features matrix with shape (batch_size, emb_dim)
            labels: ground truth labels with shape (batch_size)
        """

        return self.loss_fn(features, labels, self.margin)

    def hard_mining(self, features, labels, margin, squared=False, device='cuda'):
        """Build the triplet loss over a batch of features.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size,)
            features: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss

        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(features, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels, device).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss

    def all_mining(self, features, labels, margin, squared=False):
        """Build the triplet loss over a batch of features.
        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            features: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(features, squared=squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss  # , fraction_positive_triplets

    def _pairwise_distances(self, features, squared=False):
        """Compute the 2D matrix of distances between all the features.
        Args:
            features: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                      If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(features, features.t())

        # Get squared L2 norm for each features. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0)).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels & distinct_indices

    def _get_anchor_positive_triplet_mask(self, labels, device):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0)).bool().to(device)
#         indices_equal = torch.eye(labels.size(0)).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return labels_equal & indices_not_equal
    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    
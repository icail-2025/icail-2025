import torch
import numpy as np
from torch.nn import functional as F


# ------------------------------
# Custom Loss Functions
# ------------------------------

def inverse_freq_weighted_bce(labels, logits, samples_per_cls, no_of_classes, epsilon=1e-10):
    """
    Compute the inverse frequency weighted binary cross entropy loss.
    Weights are calculated as inverse of class frequencies and normalized.
    
    Reference: This is a common baseline approach used in many papers, but doesn't have a single
    canonical reference as it's a standard technique.
    
    Args:
        labels: A float tensor of size [batch, no_of_classes] containing multi-hot encoded labels.
        logits: A float tensor of size [batch, no_of_classes].
        samples_per_cls: A python list of size [no_of_classes] containing number of samples per class.
        no_of_classes: Total number of classes (integer).
        epsilon: Small constant to avoid division by zero.
        
    Returns:
        loss: A float tensor representing weighted BCE loss
    """

    # Convert samples per class to tensor and move to correct device
    device = logits.device
    
    # Calculate inverse frequencies
    total_samples = sum(samples_per_cls)
    class_freqs = torch.tensor(samples_per_cls, dtype=torch.float32) / total_samples
    inverse_freqs = 1.0 / (class_freqs + epsilon)
    
    # Normalize weights so they sum to no_of_classes
    weights = (inverse_freqs / torch.sum(inverse_freqs) * no_of_classes).to(device)
    
    # Expand weights to match batch dimension
    weights = weights.unsqueeze(0).expand(labels.size(0), -1)
    
    # Compute weighted BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        input=logits,
        target=labels,
        weight=weights,
        reduction='none'
    )
    
    return torch.mean(bce_loss)


def focal_loss(labels, logits, alpha, gamma, asymmetric=False):
    """
    Compute focal loss for binary classification.
    
    Reference: "Focal Loss for Dense Object Detection"
    Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
    https://arxiv.org/abs/1708.02002
    
    Args:
        labels: A float tensor of size [batch, num_classes] or [batch, 1].
        logits: A float tensor of size [batch, num_classes] or [batch, 1].
        alpha: (optional) A float scalar or tensor of size [batch_size]
            specifying weight for balanced cross entropy. Default is 0.5.
        gamma: (optional) A float scalar modulating loss from hard and easy 
            examples. Default is 1.
        asymmetric (optional): If True, gamma is only applied to the
            positive class. Default is False (both classes).

    Returns:
        focal_loss: A float32 scalar representing normalized total loss.
    """

    # Check if labels need reshaping to match logits
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(1)
        
    # Create weight tensor from alpha
    if isinstance(alpha, (float, int)):
        weights = torch.zeros_like(labels)
        weights[labels == 1] = alpha
        weights[labels == 0] = (1 - alpha)
    else:
        weights = alpha
        
    # Pass to PyTorch's sigmoid + BCE with weights
    WBCE_loss = F.binary_cross_entropy_with_logits(
        input=logits, 
        target=labels,
        weight=weights,
        reduction='none'
    )
    
    # Compute modulator
    if gamma == 0.0:
        modulator = 1.0
    else:
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)

        if asymmetric:
            modulator = torch.where(labels == 1, (1 - pt) ** gamma, torch.ones_like(pt))
        else:
            modulator = (1 - pt) ** gamma
        
    # Compute final loss
    loss = modulator * WBCE_loss
    #weighted_loss = alpha.view_as(loss) * loss
    
    # Normalize and return
    focal_loss = torch.mean(loss)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Pytorch implementation of Class-Balanced-Loss and Focal Loss (internal function)

    Reference:  "Class-Balanced Loss Based on Effective Number of Samples" 
    Authors:    Yin Cui, Menglin Jia, Tsung Yi Lin, Yang Song, Serge J. Belongie
    https://arxiv.org/abs/1901.05555

    Args:
      labels: A float tensor of size [batch, no_of_classes] containing one-hot/multi-hot encoded labels.
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    

    def focal_loss_internal(labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.

        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.

        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    # Calculate the weights
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.tensor(weights).float().to(labels.device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels.shape[0], 1) * labels
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss_internal(labels, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=labels, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = torch.nn.functional.binary_cross_entropy(input=pred, target=labels, weight=weights)
    return cb_loss


def dice_loss(labels, logits, epsilon=1e-6, smoothing="both"):
    """
    Dice Loss for binary classification.
    
    Reference: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    Authors: Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi
    https://arxiv.org/abs/1606.04797
    
    Args:
    - pred (torch.Tensor): predicted values (logits or probabilities) of shape (batch_size, 1)
    - target (torch.Tensor): ground truth values (0 or 1) of shape (batch_size, 1)
    - epsilon (float): small constant to avoid division by zero
    
    Returns:
    - dice loss (torch.Tensor)
    """

    # Smoothing
    if smoothing == "numerator":
        epsilon = [epsilon, 0]
    elif smoothing == "denominator":
        epsilon = [0, epsilon]
    else:
        epsilon = [epsilon, epsilon]

    # Sigmoid activation to get probabilities if logits are provided
    pred = torch.sigmoid(logits)
    
    # Compute Dice score
    intersection = torch.sum(pred * labels)
    dice_coeff = (2. * intersection + epsilon[0]) / (torch.sum(pred) + torch.sum(labels) + epsilon[1])
    
    # Dice loss is 1 - Dice coefficient
    return 1 - dice_coeff


def tversky_index(labels, logits, omega, tau, epsilon, smoothing): # omega = alpha, tau = beta
    """
    Tversky index for binary classification, using the provided formula.
    
    Args:
    - pred (torch.Tensor): predicted values (logits or probabilities) of shape (batch_size, 1)
    - target (torch.Tensor): ground truth values (0 or 1) of shape (batch_size, 1)
    - tau (float): weight for false positives
    - omega (float): weight for false negatives
    - epsilon (float): small constant to avoid division by zero
    
    Returns:
    - tversky index (torch.Tensor)
    """

    # Smoothing
    if smoothing == "numerator":
        epsilon = [epsilon, 0]
    elif smoothing == "denominator":
        epsilon = [0, epsilon]
    else:
        epsilon = [epsilon, epsilon]

    # Sigmoid activation to get probabilities if logits are provided
    pred = torch.sigmoid(logits)
    
    # Compute the components of the Tversky Index formula
    intersection = torch.sum(pred * labels)  # p * y
    false_pos = torch.sum(omega * pred * (1 - labels))  # omega [=alpha in many formulas] * p * (1 - y)
    false_neg = torch.sum(tau * (1 - pred) * labels)  # tau [= beta in many formulas] * (1 - p) * y

    
    # Compute the Tversky Index
    numerator = intersection + epsilon[0]
    denominator = intersection + false_neg + false_pos + epsilon[1]
    tversky_index = numerator / denominator

    return tversky_index


def tversky_loss(labels, logits, omega=0.3, tau=0.7, epsilon=1e-6, smoothing="both", alpha=None):
    """
    Tversky index for binary classification, using the provided formula.
    
    Reference: "Tversky loss function for image segmentation using 3D fully convolutional deep networks"
    Authors: Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour
    https://arxiv.org/abs/1706.05721
    
    Args:
    - pred (torch.Tensor): predicted values (logits or probabilities) of shape (batch_size, 1)
    - target (torch.Tensor): ground truth values (0 or 1) of shape (batch_size, 1)
    - tau (float): weight for false positives
    - omega (float): weight for false negatives
    - epsilon (float): small constant to avoid division by zero
    
    Returns:
    - tversky index (torch.Tensor)
    """

    # tau and omega are kept so that AUFL-CB can get (Focal) Tversky with CB weights;
    # many formulas call the weights in Tversky alpha and beta; compared to those, the
    # following code SWITCHES alpha and beta; this is done because more weight is
    # typically assigned to the false negatives (in the formulas, beta > 0.5; for us,
    # alpha > 0.5) and the usage of alpha is thus standardized w.r.t. weighted BCE.
    if alpha:
        omega = 1 - alpha
        tau = alpha
    # Tversky Loss is 1 - Tversky Index
    return 1- tversky_index(labels, logits, omega=omega, tau=tau, epsilon=epsilon, smoothing=smoothing)


def focal_tversky_loss(labels, logits, omega=0.3, tau=0.7, gamma=0.5, epsilon=1e-6, smoothing="both", alpha=None):
    # tau and omega are kept so that AUFL-CB can get (Focal) Tversky with CB weights;
    # many formulas call the weights in Tversky alpha and beta; compared to those, the
    # following code SWITCHES alpha and beta; this is done because more weight is
    # typically assigned to the false negatives (in the formulas, beta > 0.5; for us,
    # alpha > 0.5) and the usage of alpha is thus standardized w.r.t. weighted BCE.
    if alpha:
        omega = 1 - alpha
        tau = alpha
    ti = tversky_index(labels, logits, omega=omega, tau=tau, epsilon=epsilon, smoothing=smoothing)
    ft_loss = ((1 - ti) ** gamma).mean()
    return ft_loss


def combo_loss(labels, logits, alpha=0.5, lambd=0.5):
    """
    Implements Combo Loss (combination of Dice loss and weighted cross entropy loss)
    
    Reference: "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation"
    Authors: Saeid Asgari Taghanaki, Yefeng Zheng, S. Kevin Zhou, Bogdan Georgescu, Puneet Sharma, 
             Daguang Xu, Dorin Comaniciu, Ghassan Hamarneh
    https://arxiv.org/abs/1805.02798
    
    Args:
        labels: Ground truth binary labels
        logits: Raw logits from the model
        alpha: Weight for positive class in focal loss component
        lambd: Weight between Dice (lambd) and focal loss (1-lambd) components
    
    Returns:
        Combined loss value
    """

    # Compute Dice and weighted cross entropy components
    dice = dice_loss(labels, logits)
    wcb = focal_loss(labels, logits, alpha=alpha, gamma=0)

    # Compute the final combined loss
    combo = (lambd * dice) + ((1 - lambd) * wcb)

    return combo


def unified_loss(labels, logits, lambd=0.5, alpha=0.7, gamma=2.0, epsilon=1e-6):
    """
    Implements Asymmetric Unified Focal Loss (AUFL) that combines asymmetric focal loss
    with Tversky components for multi-label classification.
    
    Reference: "Asymmetric Loss For Multi-Label Classification"
    Authors: Emmanuel Zablocki, Hédi Ben-Younes, Patrick Pérez, Matthieu Cord
    https://arxiv.org/abs/2009.14119
    
    Args:
        labels: Ground truth binary labels of shape [batch_size, num_classes]
        logits: Raw logits from the model of shape [batch_size, num_classes]
        lambd: Weight balancing factor between focal and Tversky components
        alpha: Asymmetry controlling parameter for positive/negative samples
        gamma: Focusing parameter for negative samples
        epsilon: Small constant to avoid numerical instability
        
    Returns:
        Computed AUFL loss value
    """

    # Get probabilities from logits
    probs = torch.sigmoid(logits)
    
    # Compute focal components for positive and negative samples simultaneously
    pos_focal = -alpha * labels * (1 - probs).clamp(min=epsilon) * torch.log(probs.clamp(min=epsilon))
    neg_focal = -(1 - alpha) * (1 - labels) * probs.pow(gamma) * torch.log((1 - probs).clamp(min=epsilon))
    
    # Compute Tversky components
    intersection = probs * labels
    fps = alpha * probs * (1 - labels)
    fns = (1 - alpha) * (1 - probs) * labels
    
    tversky_coeff = (intersection + epsilon) / (intersection + fps + fns + epsilon)
    tversky = 1 - tversky_coeff
    
    # Compute focal Tversky component
    focal_tversky = tversky.pow(gamma)
    
    # Combine components
    focal_component = pos_focal + neg_focal
    tversky_component = labels * tversky + (1 - labels) * focal_tversky
    
    # Compute final loss
    final_loss = lambd * focal_component + (1 - lambd) * tversky_component
    
    return final_loss.mean()


def effective_unified_loss(labels, logits, samples_per_cls, no_of_classes, lambd=0.5, beta=0.9999, gamma=2.0, epsilon=1e-6):
    """
    Implements Asymmetric Unified Focal Loss with effective number-based class balancing.
    
    Args:
      labels: A float tensor of size [batch, no_of_classes] containing one-hot/multi-hot encoded labels.
        logits: Raw logits from the model of shape [batch, no_of_classes]
        samples_per_cls: List containing number of samples per class
        no_of_classes: Total number of classes
        beta: Hyperparameter for effective number calculation (CB Loss)
        lambd: Weight balancing factor between focal and Tversky components (Combo Loss)
        gamma: Focusing parameter for negative samples (Focal Loss)
        tau: Weight for false negatives in Tversky loss
        omega: Weight for false positives in Tversky loss
        epsilon: Small constant to avoid numerical instability
        
    Returns:
        Computed loss value combining effective numbers and AUFL
    """
    device = logits.device
    
    # Calculate weights using the original method
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels.shape[0], 1) * labels
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)
    
    # Get probabilities from logits
    probs = torch.sigmoid(logits)
    
    # Split samples into positive and negative
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    
    # Initialize loss components
    focal_component = torch.zeros_like(probs).to(device)
    tversky_component = torch.zeros_like(probs).to(device)
    
    # Compute focal component for positive samples
    if pos_mask.any():
        pos_probs = probs[pos_mask]
        pos_weights = weights[pos_mask]
        focal_component[pos_mask] = -pos_weights * (1 - pos_probs) * torch.log(pos_probs + epsilon)
        
        # Compute Tversky component for positive samples
        tversky_component[pos_mask] = tversky_loss(
            labels[pos_mask],
            logits[pos_mask],
            omega=1 - weights[pos_mask],
            tau=weights[pos_mask],
            epsilon=epsilon
        )
    
    # Compute focal component for negative samples
    if neg_mask.any():
        neg_probs = probs[neg_mask]
        neg_weights = weights[neg_mask]
        focal_component[neg_mask] = -(1 - neg_weights) * neg_probs.pow(gamma) * torch.log(1 - neg_probs + epsilon)
        
        # Compute Focal Tversky component for negative samples
        tversky_component[neg_mask] = focal_tversky_loss(
            labels[neg_mask],
            logits[neg_mask],
            omega=1 - weights[neg_mask],
            tau=weights[neg_mask],
            gamma=gamma,
            epsilon=epsilon
        )
    
    # Compute final loss
    final_loss = lambd * focal_component + (1 - lambd) * tversky_component

    return final_loss.mean()
import torch


def get_mask_loss(prediction, mask):
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = mask * torch.log(pred_mask + 1e-12)
    mask_loss = mask_loss + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    return mask


def get_regression_loss(prediction, mask, regression_target):
    regression_prediction = prediction[:, 1:]
    regression_loss = (torch.abs(
        regression_prediction - regression_target).sum(1) * mask
    ).sum(1).sum(1) / mask.sum(1).sum(1)
    regression_loss = regression_loss.mean(0)
    return regression_loss


def train_criterion(prediction, mask, regression_target, size_average: bool = True):
    mask_loss = get_mask_loss(prediction, mask)
    regression_loss = get_regression_loss(prediction, mask, regression_target)
    loss = mask_loss + regression_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss

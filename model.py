import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(backbone: str, num_classes: int, pretrained: bool = True):
    if backbone == "fasterrcnn_resnet50_fpn":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
            )
    elif backbone == "fasterrcnn_mobilenet_v3":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        else:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
            )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Replace the pretrained head with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

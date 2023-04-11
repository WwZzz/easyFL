import segmentation_models_pytorch as smp
import flgo.utils.fmodule as fmodule
import torch.nn.functional as F

class Model(fmodule.FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=21,                      # model output channels (number of classes in your dataset)
        )

    def forward(self, images):
        h, w = images.shape[-2:]
        origin_shape = images.shape
        pad_h = (32-h%32)%32
        pad_w = (32-w%32)%32
        pad_size = [pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2]
        images = F.pad(images, pad_size)
        res = self.model(images)
        x1 = pad_size[0]
        x2 = pad_h//2-pad_h
        y1 = pad_w//2
        y2 = pad_w//2-pad_w
        if x2==0: res = res[:,:,x1:,:]
        else: res = res[:,:,x1:x2,:]
        if y2==0: res = res[:,:,:,y1:]
        else: res = res[:,:,:,y1:y2]
        return {'out':res}

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)
from model import *
from data import *
from loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = r'C:\PyCharm\Code\MA\Mono512\reconstruction'
Legs_dir = 'Legs'
Metal_dir = 'Metal'
dataloader = DataLoader(MyData(data_path, Legs_dir, Metal_dir), batch_size=1)

L1_roi = L1_ROI()
SSIM = SSIM()
PSNR = PSNR()
ViT_G = ViT_G().to(device)
path_checkpoint = './ckpt/ckpt_ViT.pth'
checkpoint = torch.load(path_checkpoint)
epoch = checkpoint['epoch']
ViT_G.load_state_dict(checkpoint['ViT_G'])
print(f'successful load {epoch}th ckpt.pth')

ViT_G.eval()
with torch.no_grad():
    L = []
    S = []
    N = []
    for step, (leg, metal) in enumerate(dataloader):
        scale, image, mask, target = preprocessing(leg, metal, data_augmentation=False)
        scale, image, mask, target = scale.to(device), image.to(device), mask.to(device), target.to(device)

        out = ViT_G(image, mask)

        name = dataloader.batch_sampler.sampler.data_source.Legs_list[step]
        prediction = (out * scale)[0].cpu().numpy().squeeze(0)
        tiff.imwrite(fr'C:\PyCharm\Code\MA\Mono512\reconstruction\Prediction\ViT\{name}', prediction)

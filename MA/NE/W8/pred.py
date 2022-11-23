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
Swin_G = Swin_G().to(device)
path_checkpoint = './ckpt/ckpt_Swin.pth'
checkpoint = torch.load(path_checkpoint)
epoch = checkpoint['epoch']
Swin_G.load_state_dict(checkpoint['Swin_G'])
print(f'successful load {epoch}th ckpt.pth')

Swin_G.eval()
with torch.no_grad():
    L = []
    S = []
    N = []
    for step, (leg, metal) in enumerate(dataloader):
        scale, image, mask, target = preprocessing(leg, metal, data_augmentation=False)
        scale, image, mask, target = scale.to(device), image.to(device), mask.to(device), target.to(device)

        out = Swin_G(image, mask)

        name = dataloader.batch_sampler.sampler.data_source.Legs_list[step]
        prediction = (out * scale)[0].cpu().numpy().squeeze(0)
        tiff.imwrite(fr'C:\PyCharm\Code\MA\Mono512\reconstruction\Prediction\W8\{name}', prediction)

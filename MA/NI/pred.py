from NI import *
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

L = []
S = []
N = []
for step, (leg, metal) in enumerate(dataloader):
    scale, image, mask, target = preprocessing(leg, metal, data_augmentation=False)

    image = image[0].squeeze(0).numpy().astype(np.float32)
    mask = mask[0].squeeze(0).numpy().astype(np.float32)
    target = target[0].squeeze(0).numpy().astype(np.float32)

    NI_prediction = KN_image_inpainting(image, mask)

    image = transforms.ToTensor()(image).unsqueeze(0)
    mask = transforms.ToTensor()(mask).unsqueeze(0)
    target = transforms.ToTensor()(target).unsqueeze(0)
    NI_prediction = transforms.ToTensor()(NI_prediction).unsqueeze(0)

    name = dataloader.batch_sampler.sampler.data_source.Legs_list[step]
    prediction = (NI_prediction * scale)[0].cpu().numpy().squeeze(0)
    tiff.imwrite(fr'C:\PyCharm\Code\MA\Mono512\reconstruction\Prediction\NI\{name}', prediction)

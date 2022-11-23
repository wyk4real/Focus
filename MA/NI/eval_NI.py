import argparse
from NI import *
from data import *
from loss import *

'''NI evaluation-Mono300'''
data_path = r'C:\PyCharm\Code\MA\Mono512\val'
Legs_dir = 'Legs'
Metal_dir = 'Metal'
# '''NI evaluation-Cadaver4000'''
# data_path = r'C:\PyCharm\Code\MA\Cadaver'
# Legs_dir = 'V'
# Metal_dir = 'M'

dataloader = DataLoader(MyData(data_path, Legs_dir, Metal_dir), batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L1_roi = L1_ROI()
SSIM = SSIM()
PSNR = PSNR()

M = []
L = []
S = []
N = []
for step, (leg, metal) in enumerate(dataloader):
    scale, image, mask, target = preprocessing(leg, metal, data_augmentation=False)

    image = image[0].squeeze(0).numpy().astype(np.float32)
    mask = mask[0].squeeze(0).numpy().astype(np.float32)
    target = target[0].squeeze(0).numpy().astype(np.float32)
    NI_prediction = KN_image_inpainting(image, mask)
    # if (step + 1) % 100 == 0:
    #     tiff.imwrite(f'./results/_{step + 1}_image.tif', image)
    #     tiff.imwrite(f'./results/_{step + 1}_mask.tif', mask)
    #     tiff.imwrite(f'./results/_{step + 1}_target.tif', target)
    #     tiff.imwrite(f'./results/_{step + 1}_NI_prediction.tif', NI_prediction)

    image = transforms.ToTensor()(image).unsqueeze(0)
    mask = transforms.ToTensor()(mask).unsqueeze(0)
    target = transforms.ToTensor()(target).unsqueeze(0)
    NI_prediction = transforms.ToTensor()(NI_prediction).unsqueeze(0)

    Loss, roi = L1_roi(NI_prediction * scale, mask, target * scale)
    Ssim = SSIM(NI_prediction * scale, target * scale)
    Psnr = PSNR(NI_prediction * scale, target * scale)

    M.append(roi.item())
    L.append(Loss.item())
    S.append(Ssim.item())
    N.append(Psnr.item())
    print(f'Current Position {step + 1} / {len(dataloader)}')
    print(f'Loss: {Loss}, ROI: {roi}, SSIM: {Ssim}, PSNR: {Psnr}')

print(M)
print(L)
print(S)
print(N)

L_mean = np.mean(np.array(L))
S_mean = np.mean(np.array(S))
N_mean = np.mean(np.array(N))
L_std = np.std(np.array(L))
S_std = np.std(np.array(S))
N_std = np.std(np.array(N))
print(f'L_mean:{L_mean}, L_std:{L_std}, '
      f'\nS_mean:{S_mean}, S_std:{S_std}, '
      f'\nN_mean:{N_mean}, N_std:{N_std}')

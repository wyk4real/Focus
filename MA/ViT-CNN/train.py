import argparse
from model import *
from data import *
from loss import *
from torch import optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--job_id', type=str, help='Job position')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--l1_lambda', type=int, default=100, help='l1_lambda')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training')

    return parser.parse_args()


if __name__ == '__main__':
    initial_ssim = 0.9
    start_epoch = 1

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = os.path.join('/scratch', args.job_id)
    train_Legs_dir = 'train/Legs'
    train_Metal_dir = 'train/Metal'
    val_Legs_dir = 'val/Legs'
    val_Metal_dir = 'val/Metal'
    train_dataloader = DataLoader(MyData(data_path, train_Legs_dir, train_Metal_dir), args.batch_size, num_workers=2)
    val_dataloader = DataLoader(MyData(data_path, val_Legs_dir, val_Metal_dir), args.batch_size, num_workers=2)

    L1_roi = L1_ROI()
    L2 = nn.MSELoss()
    SSIM = SSIM()
    PSNR = PSNR()
    CNN_D = CNN_D().to(device)
    opt_CNN_D = optim.Adam(CNN_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler_CNN_D = optim.lr_scheduler.LambdaLR(opt_CNN_D, lr_lambda=lambda epoch: 0.95 ** epoch)
    ViT_G = ViT_G().to(device)
    opt_ViT_G = optim.Adam(ViT_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler_ViT_G = optim.lr_scheduler.LambdaLR(opt_ViT_G, lr_lambda=lambda epoch: 0.95 ** epoch)

    if args.resume:
        path_checkpoint = './ckpt/ckpt_ViT.pth'
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        CNN_D.load_state_dict(checkpoint['CNN_D'])
        opt_CNN_D.load_state_dict(checkpoint['opt_CNN_D'])
        scheduler_CNN_D.load_state_dict(checkpoint['scheduler_CNN_D'])
        ViT_G.load_state_dict(checkpoint['ViT_G'])
        opt_ViT_G.load_state_dict(checkpoint['opt_ViT_G'])
        scheduler_ViT_G.load_state_dict(checkpoint['scheduler_ViT_G'])
        print(f'successful load {start_epoch}th ckpt.pth')
    else:
        print('there are currently no weights files!')

    plot_lr = []
    plot_D_train_loss = []
    plot_G_train_loss = []
    plot_train_loss = []
    plot_train_ssim = []
    plot_train_psnr = []
    plot_val_loss = []
    plot_val_ssim = []
    plot_val_psnr = []
    for epoch in range(start_epoch, args.epochs + 1):
        # train
        CNN_D.train()
        ViT_G.train()
        D_train_loss = 0.  # L2 Loss + L2 Loss
        G_train_loss = 0.  # L2 Loss + lambda * L1 Loss
        train_loss = 0.
        train_ssim = 0.
        train_psnr = 0.
        for step, (leg, metal) in enumerate(train_dataloader):
            _, image, mask, target = preprocessing(leg, metal, data_augmentation=True)
            image, mask, target = image.to(device), mask.to(device), target.to(device)

            out = ViT_G(image, mask)

            # Train Discriminator
            _D_fake = CNN_D(out.detach(), mask)
            _D_fake_loss = L2(_D_fake, torch.zeros_like(_D_fake))
            _D_real = CNN_D(target, mask)
            _D_real_loss = L2(_D_real, torch.ones_like(_D_real))

            D_loss = _D_fake_loss + _D_real_loss

            opt_CNN_D.zero_grad()
            D_loss.backward()
            opt_CNN_D.step()

            # Train Generator
            _D_fake = CNN_D(out, mask)
            _G_fake_loss = L2(_D_fake, torch.ones_like(_D_fake))
            content_loss, _ = L1_roi(out, mask, target)

            G_loss = _G_fake_loss + content_loss * args.l1_lambda

            opt_ViT_G.zero_grad()
            G_loss.backward()
            opt_ViT_G.step()

            Ssim = SSIM(out.detach(), target)
            Psnr = PSNR(out.detach(), target)

            D_train_loss += D_loss.item()
            G_train_loss += G_loss.item()
            train_loss += content_loss.item()
            train_ssim += Ssim.item()
            train_psnr += Psnr.item()

            if epoch % 20 == 0 and step == 0:
                save_i = image[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                save_m = mask[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                save_t = target[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                save_o = out[0].detach().cpu().numpy().squeeze(0).astype(dtype=np.float32)
                tiff.imwrite(f'./train/{epoch}th_input.tif', save_i)
                tiff.imwrite(f'./train/{epoch}th_mask.tif', save_m)
                tiff.imwrite(f'./train/{epoch}th_target.tif', save_t)
                tiff.imwrite(f'./train/{epoch}th_prediction.tif', save_o)

        D_train_loss /= len(train_dataloader)
        G_train_loss /= len(train_dataloader)
        train_loss /= len(train_dataloader)
        train_ssim /= len(train_dataloader)
        train_psnr /= len(train_dataloader)
        print('*' * 40)
        print('epoch: {},'
              '\nlr: {},'
              '\nDiscriminator(train): {},'
              '\nGenerator(train): {},'
              '\nL1(train): {},'
              '\nSSIM(train): {},'
              '\nPSNR(train): {},'
              .format(epoch,
                      opt_CNN_D.param_groups[0]['lr'],
                      D_train_loss,
                      G_train_loss,
                      train_loss,
                      train_ssim,
                      train_psnr))
        # evaluate
        CNN_D.eval()
        ViT_G.eval()
        with torch.no_grad():
            L = []
            S = []
            N = []
            for step, (leg, metal) in enumerate(val_dataloader):
                _, image, mask, target = preprocessing(leg, metal, data_augmentation=True)
                image, mask, target = image.to(device), mask.to(device), target.to(device)

                out = ViT_G(image, mask)

                content_loss, _ = L1_roi(out, mask, target)
                Ssim = SSIM(out, target)
                Psnr = PSNR(out, target)

                L.append(content_loss.item())
                S.append(Ssim.item())
                N.append(Psnr.item())

                if epoch % 20 == 0 and step == 0:
                    save_i = image[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                    save_m = mask[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                    save_t = target[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                    save_o = out[0].cpu().numpy().squeeze(0).astype(dtype=np.float32)
                    tiff.imwrite(f'./val/{epoch}th_input.tif', save_i)
                    tiff.imwrite(f'./val/{epoch}th_mask.tif', save_m)
                    tiff.imwrite(f'./val/{epoch}th_target.tif', save_t)
                    tiff.imwrite(f'./val/{epoch}th_prediction.tif', save_o)

            L_mean = np.mean(np.array(L))
            S_mean = np.mean(np.array(S))
            N_mean = np.mean(np.array(N))
            L_std = np.std(np.array(L))
            S_std = np.std(np.array(S))
            N_std = np.std(np.array(N))
            print('L1(mean): {},''\nSSIM(mean): {},''\nPSNR(mean): {},'
                  '\nL1(std): {},''\nSSIM(std): {},''\nPSNR(std): {},'
                  .format(L_mean, S_mean, N_mean,
                          L_std, S_std, N_std))

            if S_mean > initial_ssim:
                initial_ssim = S_mean
                checkpoint = {
                    'epoch': epoch,
                    'CNN_D': CNN_D.state_dict(),
                    'opt_CNN_D': opt_CNN_D.state_dict(),
                    'scheduler_CNN_D': scheduler_CNN_D.state_dict(),
                    'ViT_G': ViT_G.state_dict(),
                    'opt_ViT_G': opt_ViT_G.state_dict(),
                    'scheduler_ViT_G': scheduler_ViT_G.state_dict()
                }
                torch.save(checkpoint, './ckpt/ckpt_ViT.pth')
                print(f'The current best model is {epoch}, which has been saved!')

            plot_lr.append(opt_CNN_D.param_groups[0]['lr'])
            plot_D_train_loss.append(D_train_loss)
            plot_G_train_loss.append(G_train_loss)
            plot_train_loss.append(train_loss)
            plot_train_ssim.append(train_ssim)
            plot_train_psnr.append(train_psnr)
            plot_val_loss.append(L_mean)
            plot_val_ssim.append(S_mean)
            plot_val_psnr.append(N_mean)

        scheduler_CNN_D.step()
        scheduler_ViT_G.step()

    plt.figure()
    plt.plot(plot_lr, label='lr')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('learning rate vs Number of epochs')
    plt.legend()
    plt.grid(alpha=0.4, linestyle=':')
    plt.savefig('./plots/lr.png')

    plt.figure()
    plt.plot(plot_D_train_loss, label='D_Loss(train)')
    plt.plot(plot_G_train_loss, label='G_Loss(train)')
    plt.xlabel('epoch')
    plt.ylabel('L1 loss / L2 loss')
    plt.title('Loss vs Number of epochs')
    plt.legend()
    plt.grid(alpha=0.4, linestyle=':')
    plt.savefig('./plots/GAN_Loss.png')

    plt.figure()
    plt.plot(plot_train_loss, label='train')
    plt.plot(plot_val_loss, label='val')
    plt.xlabel('epoch')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss vs Number of epochs')
    plt.legend()
    plt.grid(alpha=0.4, linestyle=':')
    plt.savefig('./plots/Loss.png')

    plt.figure()
    plt.plot(plot_train_ssim, label='train')
    plt.plot(plot_val_ssim, label='val')
    plt.xlabel('epoch')
    plt.ylabel('SSIM')
    plt.title('Structural similarity vs Number of epochs')
    plt.legend()
    plt.grid(alpha=0.4, linestyle=':')
    plt.savefig('./plots/Ssim.png')

    plt.figure()
    plt.plot(plot_train_psnr, label='train')
    plt.plot(plot_val_psnr, label='val')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.title('Peak signal-to-noise ratio vs Number of epochs')
    plt.legend()
    plt.grid(alpha=0.4, linestyle=':')
    plt.savefig('./plots/PSNR.png')

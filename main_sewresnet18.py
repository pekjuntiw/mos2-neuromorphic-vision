import json
import math
import os
import shutil
import time
import torch
import torch.utils.data as data
import numpy as np
from spikingjelly.clock_driven import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Karlsruhe, loader
from model import SEWResnet18
from torchvision import transforms
import pandas as pd


"""
ALL PARAMETERS SHOULD BE IN STANDARD S.I. UNITS TO PREVENT CONFUSION!
taus scalable
"""

# torch.autograd.set_detect_anomaly(True)
# torch.set_printoptions(profile="full")
torch.set_printoptions(edgeitems=100)
torch.set_printoptions(precision=2)
torch.set_printoptions(linewidth=1000)


def main():
    device = 'cuda'

    FLAGS = {
        "dataset_data_dir": 'karlsruhe/objects_2011_a/labeldata_344x100',
        "dataset_target_dir": 'karlsruhe/objects_2011_a/labeldata',
        "augmentation_brightness": 0.3,
        "augmentation_contrast": 0.3,
        "augmentation_rotation": 15,
        "augmentation_jitter": 0.6,
        "mixup_alpha": 1.5,
        "torch_seed": np.random.randint(0, 2**31 - 1),
        "numpy_seed": np.random.randint(0, 2**31 - 1),
        "split_seed": 1183289531,

        "train_epoch": 455,
        "snapshots": [i for i in range(0, 1000, 20)],
        "batch_size": 25,
        "train_size": 1575,
        "lr": 1e-1,
        "decay_step": 1260,  # 1960,  # 375,  # 50,
        "decay_rate": 0.8,
        "loss_param": {"high": 2, "low": 0.2},
        "cos_t_0": 30*63,
        "cos_t_mult": 2,
        "cos_eta_min": 5e-3,
        "linear_start_factor": 0.01,
        "linear_total_iters": 5*63,

        "num_in": [100, 344],  # input size
        "num_out": 2,
        "l2_reg": 0.0002,

        "save_model": True,

        "dt": 0.4,
        "T": 40.,
        "tau": 2,  # LIF tau
    }
    
    export = False

    if 'torch_seed' in FLAGS.keys() and not FLAGS['torch_seed'] is None:
        torch.manual_seed(FLAGS['torch_seed'])
    if 'numpy_seed' in FLAGS.keys() and not FLAGS['numpy_seed'] is None:
        np.random.seed(FLAGS['numpy_seed'])
    split_seed = FLAGS['split_seed'] if 'split_seed' in FLAGS.keys() else 20250703
    print('torch seed:', FLAGS['torch_seed'])
    print('numpy seed:', FLAGS['numpy_seed'])
    print('split seed:', split_seed)

    save_model = FLAGS["save_model"]
    snapshots = [] if "snapshots" not in FLAGS.keys() else FLAGS["snapshots"]
    dataset_data_dir = FLAGS["dataset_data_dir"]
    dataset_target_dir = FLAGS["dataset_target_dir"]
    model_num = 1
    model_dir = f'karlsruhe_sewresnet18/{model_num}'
    model_output_dir = f'{model_dir}/model'
    log_dir = f'{model_dir}/log'
    fig_dir = f'{model_dir}/fig'
    model_output_name = f'{model_output_dir}/karlsruhe_sewresnet18'  # to be completed below

    # set if resume training from a saved model
    resume_training = False
    resume_from_epoch = 411

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # save flags
    if save_model:
        with open(f'{model_dir}/flags.json', 'w') as fp:
            json.dump(FLAGS, fp, sort_keys=True, indent=4)

    augmentation_brightness = FLAGS["augmentation_brightness"]
    augmentation_contrast = FLAGS["augmentation_contrast"]
    augmentation_rotation = FLAGS["augmentation_rotation"]
    augmentation_jitter = FLAGS["augmentation_jitter"]
    mixup_alpha = FLAGS["mixup_alpha"]

    batch_size = FLAGS["batch_size"]
    train_size = FLAGS["train_size"]
    train_epoch = FLAGS["train_epoch"]
    lr = FLAGS["lr"]
    cos_t_0 = FLAGS["cos_t_0"]
    cos_t_mult = FLAGS["cos_t_mult"]
    cos_eta_min = FLAGS["cos_eta_min"]
    linear_start_factor = FLAGS["linear_start_factor"]
    linear_total_iters = FLAGS["linear_total_iters"]

    num_in = FLAGS["num_in"]
    assert dataset_data_dir.split('_')[-1] == f'{num_in[1]}x{num_in[0]}'
    num_out = FLAGS["num_out"]
    l2_reg = FLAGS["l2_reg"]

    dt = FLAGS["dt"]
    T = FLAGS["T"]
    tau = FLAGS["tau"]

    writer = SummaryWriter(log_dir)

    # initialize dataloader
    dataset = Karlsruhe(
        data_path=dataset_data_dir,
        target_path=dataset_target_dir,
        dt=dt, T=T, mixup_alpha=mixup_alpha, jitter=augmentation_jitter, invert=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=augmentation_brightness, contrast=augmentation_contrast),
            transforms.RandomRotation(augmentation_rotation),
            transforms.ToTensor(),
        ])
    )
    test_size = len(dataset) - train_size
    test_batch_size = batch_size * 1
    train_batch_per_epoch = int(train_size / batch_size)
    test_batch_per_epoch = math.ceil(test_size / test_batch_size)
    train_dataset, test_dataset = data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(split_seed)
    )
    train_data_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False,
    )
    dataset.train_indices = train_dataset.indices

    # initialize model
    net = SEWResnet18(num_classes=num_out, tau=tau, dt=dt, )
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=l2_reg, nesterov=True)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=linear_start_factor, total_iters=linear_total_iters)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cos_t_0, T_mult=cos_t_mult, eta_min=cos_eta_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[linear_total_iters])

    # check size of model
    mem_params = sum([param.nelement() * param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(net)
    print(f"Params: {mem_params}bytes      Bufs: {mem_bufs}bytes      Total: {mem}bytes")
    if device == 'cuda':
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # load save model if resume training
    if resume_training:
        print(f'Loading model, scheduler and optimizer from {model_output_name}_ep{resume_from_epoch}.ckpt')
        chkpt = torch.load(f'{model_output_name}_ep{resume_from_epoch}.ckpt')
        net.load_state_dict(chkpt["net"])
        optimizer.load_state_dict(chkpt["optimizer"])
        scheduler.load_state_dict(chkpt["scheduler"])

    test_accs = np.load(f'{model_output_dir}/test_accs_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_accs = np.load(f'{model_output_dir}/train_accs_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_loss = np.load(f'{model_output_dir}/test_loss_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_loss = np.load(f'{model_output_dir}/train_loss_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_accs_step = np.load(
        f'{model_output_dir}/test_accs_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_accs_step = np.load(
        f'{model_output_dir}/train_accs_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_loss_step = np.load(
        f'{model_output_dir}/test_loss_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_loss_step = np.load(
        f'{model_output_dir}/train_loss_step_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    train_times = len(train_loss_step) if resume_training else 0
    test_times = len(test_loss_step) if resume_training else 0
    max_test_accuracy = max(test_accs) if resume_training else 0
    confusion_matrix = np.zeros([num_out, 2, 2], dtype=int)

    NAN_STOP_FLAG = False

    start_epoch = resume_from_epoch + 1 if resume_training else 0

    t_start = time.perf_counter()
    for epoch in range(start_epoch, train_epoch):
        print(f"Epoch {epoch}: lr={scheduler.get_last_lr()}")
        train_correct_sum = 0
        train_sum = 0
        train_loss_batches = []

        # train model
        net.train()
        # set use transform
        dataset.use_transform = True
        # get data in batches
        for i, (img, label) in enumerate(pbar := tqdm(
                loader(train_data_loader, device), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                total=train_batch_per_epoch, desc="Training", ascii=" >>>>="
        )):
            optimizer.zero_grad()

            # forward pass
            output = net(img.float())

            results = output[-1]

            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(results, label.float())

            loss.backward()
            optimizer.step()
            scheduler.step()

            # reset SNN state after each forward pass
            functional.reset_net(net)

            results = (torch.sigmoid(results) > 0.5).int()
            label = (label > 0.5).int()
            is_correct = (results == label).int()
            sample_accuracy = is_correct.sum(dim=1) / label.size(1)  # accuracy per sample, final accuracy is the mean of sample accuracies
            train_correct_sum += sample_accuracy.sum().item()
            train_sum += label.size(0)
            train_batch_accuracy = sample_accuracy.mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs_step.append(train_batch_accuracy)

            train_loss_batches.append(loss.item())
            train_loss_step.append(loss.item())
            writer.add_scalar('train_batch_loss', loss, train_times)

            pbar.set_postfix_str(
                f'    (Step {train_times}: acc={train_batch_accuracy * 100:.4f}, loss={loss.item():.4f})'
            )

            train_times += 1

            if i % 20 == 0:
                fig, ax = plt.subplots(1, 1)
                ax1 = ax.twinx()
                ax1.plot(np.arange(train_times) / train_batch_per_epoch, train_loss_step, color='tab:green', alpha=0.4)
                ax1.plot(np.arange(test_times) / test_batch_per_epoch, test_loss_step, color='tab:red', alpha=0.6)
                ax1.set_ylim(0.01, 4)
                ax1.set_yscale('log')
                ax1.set_ylabel("train loss")
                ax.plot(np.arange(train_times) / train_batch_per_epoch, train_accs_step, color='tab:blue', alpha=0.4)
                ax.plot(np.arange(test_times) / test_batch_per_epoch, test_accs_step, color='tab:orange', alpha=0.6)
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel("train accs")
                ax.set_xlabel("epochs")
                fig.savefig('live_accuracy.png')
                print(f"\n===========Saved live accuracy plot @ epoch {epoch}")
                plt.close(fig)

        train_accuracy = train_correct_sum / train_sum
        train_loss_avg = np.mean(train_loss_batches)
        train_accs.append(train_accuracy)
        train_loss.append(train_loss_avg)

        # test model
        net.eval()
        # reset use transform
        dataset.use_transform = False
        with torch.no_grad():
            test_correct_sum = 0
            test_sum = 0
            test_loss_batches = []

            # get data in batches
            for i, (img, label) in enumerate(tqdm(
                    loader(test_data_loader, device),
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", total=test_batch_per_epoch,
                    desc="Testing", ascii=" >>>>="
            )):
                # forward pass
                output = net(img.float())

                results = output[-1]

                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(results, label.float())

                # reset SNN state after each forward pass
                functional.reset_net(net)

                results = (torch.sigmoid(results) > 0.5).int()
                is_correct = (results == label).int()
                sample_accuracy = is_correct.sum(dim=1) / label.size(1)  # accuracy per sample, final accuracy is the mean of sample accuracies
                test_correct_sum += sample_accuracy.sum().item()
                test_sum += label.size(0)
                test_accs_step.append(sample_accuracy.mean().item())

                test_loss_batches.append(loss.item())
                test_loss_step.append(loss.item())
                writer.add_scalar('test_batch_loss', loss, test_times)

                test_times += 1

            test_accuracy = test_correct_sum / test_sum
            test_loss_avg = np.mean(test_loss_batches)
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            test_loss.append(test_loss_avg)

        # save model if test accuracy is improved or is final epoch or is snapshot
        if save_model and (test_accuracy >= max_test_accuracy or epoch == train_epoch - 1 or epoch in snapshots):
            if test_accuracy >= max_test_accuracy:
                print(
                    f'Improved. Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
            elif epoch == train_epoch - 1:
                print(
                    f'Final epoch. Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
            else:
                print(
                    f'Snapshot. Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
            chkpt = {
                "net": net.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(chkpt, f'{model_output_name}_ep{epoch}.ckpt')

            print(f'Saving losses and accuracies to {model_output_dir}')
            np.save(f'{model_output_dir}/train_accs_ep{epoch}.npy', np.array(train_accs))
            np.save(f'{model_output_dir}/test_accs_ep{epoch}.npy', np.array(test_accs))
            np.save(f'{model_output_dir}/train_loss_ep{epoch}.npy', np.array(train_loss))
            np.save(f'{model_output_dir}/test_loss_ep{epoch}.npy', np.array(test_loss))
            np.save(f'{model_output_dir}/train_accs_step_ep{epoch}.npy', np.array(train_accs_step))
            np.save(f'{model_output_dir}/test_accs_step_ep{epoch}.npy', np.array(test_accs_step))
            np.save(f'{model_output_dir}/train_loss_step_ep{epoch}.npy', np.array(train_loss_step))
            np.save(f'{model_output_dir}/test_loss_step_ep{epoch}.npy', np.array(test_loss_step))

        max_test_accuracy = max(max_test_accuracy, test_accuracy)

        print(
            f"Epoch {epoch}: train_acc = {train_accuracy}, test_acc={test_accuracy}, train_loss_avg={train_loss_avg}, test_loss_avg={test_loss_avg}, max_test_acc={max_test_accuracy}, train_times={train_times}")

        if NAN_STOP_FLAG:
            print('NaN in weights, abort training!')
            break

        print()

    t_end = time.perf_counter()
    duration = t_end - t_start
    if train_epoch - start_epoch > 1:
        print(
            f"\nTotal training time {duration:.3f} s. Average {(duration / (train_times / train_batch_per_epoch - start_epoch)):.3f} s/epoch.\n")
    
    
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "bn" not in name and "bias" not in name:
                param_min = torch.min(param).item()
                param_max = torch.max(param).item()
                abs_max = max(abs(param_min), abs(param_max))
                quantized_step = abs_max / 15.
                scale = quantized_step / 0.07  # weight per uS
                print(f"Parameter Name: {name}, min={param_min:.5f}, max={param_max:.5f}, step={quantized_step:.5f}, scale={scale:.5f}")
                param.copy_(torch.round(param / quantized_step) * quantized_step)
                plt.figure()
                plt.hist(param.data.cpu().numpy().flatten(), bins=100, alpha=0.4, color='r')
                plt.hist((torch.round(param / quantized_step) * quantized_step).data.cpu().numpy().flatten(), bins=100, alpha=0.4, color='g')
                plt.title(name)
    
    
    # final run
    net.eval()
    # reset use transform
    dataset.use_transform = False
    with torch.no_grad():
        test_correct_sum = 0
        test_sum = 0
        test_loss_batches = []

        # get data in batches
        for i, (img, label) in enumerate(tqdm(
                loader(test_data_loader, device),
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", total=test_batch_per_epoch,
                desc="Testing", ascii=" >>>>="
        )):
            # forward pass
            output = net(img.float())

            results = output[-1]

            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(results, label.float())

            # reset SNN state after each forward pass
            functional.reset_net(net)

            results = (torch.sigmoid(results) > 0.5).int()
            is_correct = (results == label).int()
            sample_accuracy = is_correct.sum(dim=1) / label.size(1)  # accuracy per sample, final accuracy is the mean of sample accuracies
            test_correct_sum += sample_accuracy.sum().item()
            test_sum += label.size(0)
            test_accs_step.append(sample_accuracy.mean().item())

            test_loss_batches.append(loss.item())
            test_loss_step.append(loss.item())
            writer.add_scalar('test_batch_loss', loss, test_times)

            # calculate confusion matrix
            for target, predicted in zip(label.cpu().numpy(), results.cpu().numpy()):
                for n, (t, p) in enumerate(zip(target, predicted)):
                    # 00: TN, 01: FN, 10: FP, 11: TP
                    confusion_matrix[n, p, t] += 1

        test_accuracy = test_correct_sum / test_sum
        print(test_accuracy, (confusion_matrix[0, 0, 0] + confusion_matrix[0, 1, 1] + confusion_matrix[1, 0, 0] + confusion_matrix[1, 1, 1])/confusion_matrix.sum())
    
    

    # plot some figures
    net.eval()
    with torch.no_grad():
        img, label = test_dataset[1]
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)

        # forward pass
        output_full = net(img.float())
        output_full_sigmoid = torch.sigmoid(output_full)
        output_full = output_full.squeeze().cpu().numpy()
        output_full_sigmoid = output_full_sigmoid.squeeze().cpu().numpy()

        # get model dynamical states
        l4_out = net.l4_out.cpu().numpy()
        fc_in = net.fc_in.cpu().numpy()

        try:
            plt.figure()
        except RuntimeError:
            plt.switch_backend('Agg')
        finally:
            plt.close()

        plt.figure()
        plt.plot(np.arange(l4_out.shape[0]), l4_out)
        plt.legend()
        plt.title(f"L4 output")
        plt.savefig(f'{fig_dir}/l4_out.png')
        plt.figure()
        plt.plot(np.arange(fc_in.shape[0]), fc_in[:, :20])
        plt.legend()
        plt.title(f"FC in")
        plt.savefig(f'{fig_dir}/fc_in.png')
        plt.figure()
        plt.plot(np.arange(output_full.shape[0]), output_full, label=dataset.get_classes(False))
        plt.legend()
        plt.title(f"Output Full: {label.cpu().numpy()}")
        plt.savefig(f'{fig_dir}/out_full.png')
        plt.figure()
        plt.plot(np.arange(output_full.shape[0]), output_full_sigmoid, label=dataset.get_classes(False))
        plt.legend()
        plt.title(f"Sigmoid Output Full: {label.cpu().numpy()}")
        plt.savefig(f'{fig_dir}/out_sigmoid_full.png')

    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    if save_model:
        np.save(f'{model_dir}/train_accs.npy', train_accs)
        np.save(f'{model_dir}/test_accs.npy', test_accs)
        np.save(f'{model_dir}/train_loss.npy', train_loss)
        np.save(f'{model_dir}/test_loss.npy', test_loss)

    plt.figure()
    plt.plot(np.arange(train_accs.shape[0]), train_accs, label="train")
    plt.plot(np.arange(test_accs.shape[0]), test_accs, label="test")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(f'{fig_dir}/accs.png')
    plt.figure()
    plt.plot(np.arange(train_loss.shape[0]), train_loss, label="train")
    plt.plot(np.arange(test_loss.shape[0]), test_loss, label="test")
    plt.legend()
    plt.semilogy()
    plt.title("Loss")
    plt.savefig(f'{fig_dir}/loss.png')
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    sns.heatmap(
        confusion_matrix[0],
        annot=True, fmt="d",
        xticklabels=['T neg', 'T pos'],
        yticklabels=['P neg', 'P pos']
    )
    ax.set_title(dataset.get_classes(False)[0])
    ax.set_xlabel("Target")
    ax.set_ylabel("Result")
    ax.invert_yaxis()
    ax.tick_params(axis='y', rotation=0)
    ax = plt.subplot(1, 3, 2)
    sns.heatmap(
        confusion_matrix[1],
        annot=True, fmt="d",
        xticklabels=['T neg', 'T pos'],
        yticklabels=['P neg', 'P pos']
    )
    ax.set_title(dataset.get_classes(False)[1])
    ax.set_xlabel("Target")
    ax.set_ylabel("Result")
    ax.invert_yaxis()
    ax.tick_params(axis='y', rotation=0)
    plt.savefig(f'{fig_dir}/confusion.png')
    
    shutil.copy('live_accuracy.png', f'{model_dir}/live_accuracy.png')
    if export:
        pd.DataFrame({
            'Epoch': np.arange(train_accs.shape[0]),
            'Train accuracy': train_accs * 100,
            'Test accuracy': test_accs * 100,
            'Train loss': train_loss,
            'Test loss': test_loss,
        }).to_csv(f'{model_dir}/train_stats.csv', index=False)
        for n, c in enumerate(dataset.get_classes(False)):
            pd.DataFrame({
                f'Actual #{i}': confusion_matrix[n, :, i] for i in range(num_out)
            }, index=[f'Output #{i}' for i in range(num_out)]).to_csv(f'{model_dir}/test_confusion_{c}.csv')

    plt.show()

    print(f'Done! Check model at {model_dir}/')


if __name__ == '__main__':
    main()

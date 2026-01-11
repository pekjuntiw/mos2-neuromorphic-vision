import json
import os
import shutil
import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from spikingjelly.clock_driven import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import raster_plot
from dataset import ColoredNumber, loader
from model import RateNet
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
        "dataset_type": "3-color",  # "7-color" or "3-color"
        "torch_seed": np.random.randint(0, 2**31 - 1),
        "numpy_seed": np.random.randint(0, 2**31 - 1),

        "train_epoch": 700,
        "snapshots": [i for i in range(0, 400, 20)],
        "batch_size": 28,
        "train_size": 1,
        "lr": 1e-1,
        "decay_step": 200,
        "decay_rate": 0.95,
        "loss_param": {"high": 2, "low": 0.2},
        "l2_reg": 0.,

        "save_model": True,

        "dt": 0.05,
        "T": 3.,
        "tau": 1,  # LIF tau
    }

    export = True

    if 'torch_seed' in FLAGS.keys() and not FLAGS['torch_seed'] is None:
        torch.manual_seed(FLAGS['torch_seed'])
    if 'numpy_seed' in FLAGS.keys() and not FLAGS['numpy_seed'] is None:
        np.random.seed(FLAGS['numpy_seed'])

    dataset_type = FLAGS["dataset_type"]
    assert dataset_type in ["3-color", "7-color"], "dataset_type must be '3-color' or '7-color'"

    save_model = FLAGS["save_model"]
    snapshots = [] if "snapshots" not in FLAGS.keys() else FLAGS["snapshots"]
    model_num = 1
    model_dir = f'{dataset_type}/{model_num}'
    model_output_dir = f'{model_dir}/model'
    log_dir = f'{model_dir}/log'
    fig_dir = f'{model_dir}/fig'
    model_output_name = f'{model_output_dir}/{dataset_type}'  # to be completed below

    # set if resume training from a saved model
    resume_training = False
    resume_from_epoch = 699

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

    batch_size = FLAGS["batch_size"]
    train_epoch = FLAGS["train_epoch"]
    lr = FLAGS["lr"]
    decay_step = FLAGS["decay_step"]
    decay_rate = FLAGS["decay_rate"]
    loss_param = FLAGS["loss_param"]

    num_in = 16 if dataset_type == "3-color" else 48
    num_out = 3 if dataset_type == "3-color" else 7
    l2_reg = FLAGS["l2_reg"]

    dt = FLAGS["dt"]
    T = FLAGS["T"]
    tau = FLAGS["tau"]

    writer = SummaryWriter(log_dir)

    # initialize dataloader
    train_dataset = ColoredNumber(dt=dt, T=T, dataset_type=dataset_type, transform=None)
    train_size = len(train_dataset)
    train_batch_per_epoch = int(train_size / batch_size)
    train_data_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    )

    # initialize model
    net = RateNet(num_in=num_in, num_out=num_out, tau=tau, dt=dt, device=device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

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

    fc1_grad = np.load(f'{model_output_dir}/grad_fc1_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    train_times = len(train_loss_step) if resume_training else 0
    test_times = len(test_loss_step) if resume_training else 0
    max_test_accuracy = max(test_accs) if resume_training else 0
    confusion_matrix = np.zeros([num_out, num_out], dtype=int)

    initial_fc1 = np.load(f'{model_dir}/weights_initial_fc1.npy') if resume_training else net.fc1.weight.data.cpu().numpy()
    np.save(f'{model_dir}/weights_initial_fc1.npy', initial_fc1)
    fc1_stats_step = np.load(f'{model_output_dir}/stats_fc1_ep{resume_from_epoch}.npy').tolist() if resume_training else []

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
        train_dataset.use_transform = True
        # get data in batches
        for i, (pattern, label) in enumerate(
                pbar := tqdm(loader(train_data_loader, device), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                             total=train_batch_per_epoch, desc="Training", ascii=" >>>>=")):
            optimizer.zero_grad()

            # forward pass
            output = net(pattern.float())
            results = torch.mean(output, dim=0) / dt

            loss = F.mse_loss(results, F.one_hot(label, num_out).float() * (loss_param['high'] - loss_param['low']) + loss_param['low'])

            loss.backward()
            optimizer.step()
            scheduler.step()

            # reset SNN state after each forward pass
            functional.reset_net(net)

            results = results.argmax(1)
            is_correct = (results == label).float()
            train_correct_sum += is_correct.sum().item()
            train_sum += label.numel()
            train_batch_accuracy = is_correct.mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs_step.append(train_batch_accuracy)

            train_loss_batches.append(loss.item())
            train_loss_step.append(loss.item())
            writer.add_scalar('train_batch_loss', loss, train_times)

            fc1_grad_norm = torch.linalg.norm(net.fc1.weight.grad).item()
            fc1_grad.append(fc1_grad_norm)

            w1_f = net.fc1.weight.data.cpu().numpy()
            fc1_stats_step.append(
                [w1_f.mean(), w1_f.std(), np.quantile(w1_f, 0.25), np.quantile(w1_f, 0.5), np.quantile(w1_f, 0.75)]
            )

            pbar.set_postfix_str(
                f'    (Step {train_times}: acc={train_batch_accuracy * 100:.4f}, loss={loss.item():.4f})    ({i}a:  {fc1_grad_norm:.6f})'
            )

            train_times += 1

            with torch.no_grad():
                if torch.isnan(net.fc1.weight.data).any():
                    NAN_STOP_FLAG = True
                    print('\nNaN in weights in current batch, stopping current epoch!')
                    break

            if i % 20 == 0:
                fig, ax = plt.subplots(1, 1)
                ax1 = ax.twinx()
                ax1.plot(np.arange(train_times) / train_batch_per_epoch, train_loss_step, color='tab:green', alpha=0.4)
                ax1.set_ylim(0.01, 1.5)
                ax1.set_yscale('log')
                ax1.set_ylabel("train loss")
                ax.plot(np.arange(train_times) / train_batch_per_epoch, train_accs_step, color='tab:blue', alpha=0.4)
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel("train accs")
                ax.set_xlabel("epochs")
                fig.savefig('live_accuracy.png')
                print(f"\n===========Saved live accuracy plot @ epoch {epoch}")
                plt.close(fig)
                fig, ax = plt.subplots(2, 1)
                l = ax[0].plot(np.arange(train_times) / train_batch_per_epoch, [s[0] for s in fc1_stats_step], '-.')
                ax[0].plot(np.arange(train_times) / train_batch_per_epoch, [s[3] for s in fc1_stats_step],
                           color=l[0].get_color())
                ax[0].fill_between(np.arange(train_times) / train_batch_per_epoch, [s[2] for s in fc1_stats_step],
                                   [s[4] for s in fc1_stats_step], alpha=0.4)
                fig.savefig('live_weights.png')
                plt.close(fig)

        train_accuracy = train_correct_sum / train_sum
        train_loss_avg = np.mean(train_loss_batches)
        train_accs.append(train_accuracy)
        train_loss.append(train_loss_avg)

        # test model
        net.eval()
        # reset use transform
        train_dataset.use_transform = False
        with torch.no_grad():
            test_correct_sum = 0
            test_sum = 0
            test_loss_batches = []

            # get data in batches
            for i, (pattern, label) in enumerate(tqdm(loader(train_data_loader, device),
                                           bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", total=train_batch_per_epoch,
                                           desc="Testing", ascii=" >>>>=")):
                # forward pass
                output = net(pattern.float())
                results = torch.mean(output, dim=0) / dt

                loss = F.mse_loss(results, F.one_hot(label, num_out).float() * (loss_param['high'] - loss_param['low']) + loss_param['low'])

                # reset SNN state after each forward pass
                functional.reset_net(net)

                results = results.argmax(1)
                is_correct = (results == label).float()
                test_correct_sum += is_correct.sum().item()
                test_sum += label.numel()
                test_accs_step.append(is_correct.mean().item())

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

            print(f'Saving gradients to {model_output_dir}')
            np.save(f'{model_output_dir}/grad_fc1_ep{epoch}.npy', np.array(fc1_grad))

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

    if export:
        pd.DataFrame({
            'Epoch': np.arange(len(train_accs)),
            'Accuracy': np.array(test_accs) * 100,
            'Loss': np.array(test_loss),
        }).to_csv(f'{model_dir}/training_stats.csv', index=False)

        net.eval()
        train_dataset.use_transform = False
        with torch.no_grad():

            results_all = []
            # get data in batches
            for i, (pattern, label) in enumerate(tqdm(loader(train_data_loader, device),
                                                      bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                                                      total=train_batch_per_epoch,
                                                      desc="Testing", ascii=" >>>>=")):
                # forward pass
                output = net(pattern.float())
                results = torch.mean(output, dim=0) / dt  # [batch, features]

                # reset SNN state after each forward pass
                functional.reset_net(net)

                results = results.cpu().numpy()
                results_all.append(np.sort(results, axis=1)[:, ::-1])  # sort descending
            results_all = np.concatenate(results_all, axis=0)  # [samples, features]

            pd.DataFrame(
                results_all, columns=['Output class']+[f'Other class {i}' for i in range(1, results_all.shape[1])]
            ).to_csv(f'{model_dir}/frequency_sorted.csv', index=False)
            pd.DataFrame({
                'Mean for Output class': [np.mean(results_all[:, 0])],
                'Mean for Other classes': [np.mean(results_all[:, 1:])],
            }).to_csv(f'{model_dir}/frequency_stats.csv', index=False)

    # plot some figures
    net.eval()
    with torch.no_grad():
        pattern, label = train_dataset[2]  # test_dataset[1]
        pattern = torch.unsqueeze(pattern, dim=0)
        pattern = pattern.to(device)

        # forward pass
        net(pattern.float())

        # get model dynamical states
        v_evol_1 = net.v_1.cpu().numpy()
        s_evol_1 = net.spike_1.cpu().numpy()
        x_evol_1 = net.neuron_in_1.cpu().numpy()
        p_evol_1 = net.p_1.cpu().numpy()

        try:
            plt.figure()
        except RuntimeError:
            plt.switch_backend('Agg')
        finally:
            plt.close()

        plt.figure()
        raster_plot(np.arange(pattern.shape[1]), pattern.cpu()[0], show=False, xlim=(-10, pattern.shape[1] + 10))
        plt.title(label)
        plt.figure()
        plt.plot(np.arange(v_evol_1.shape[0]), v_evol_1)
        plt.title("Vmem Out")
        plt.savefig(f'{fig_dir}/vmem-out.png')
        plt.figure()
        plt.plot(np.arange(x_evol_1.shape[0]), x_evol_1)
        plt.title("Input Out")
        plt.savefig(f'{fig_dir}/input-out.png')
        plt.figure()
        plt.plot(np.arange(p_evol_1.shape[0]), p_evol_1)
        plt.title("Power Out")
        plt.savefig(f'{fig_dir}/power-out.png')
        plt.figure()
        raster_plot(np.arange(s_evol_1.shape[0]), s_evol_1, show=False, xlim=(-10, s_evol_1.shape[0] + 10))
        plt.title("Spike Raster Out")
        plt.savefig(f'{fig_dir}/spike-raster-out.png')
        plt.figure()
        sns.heatmap(initial_fc1, square=False, center=0, cmap='vlag')
        plt.gca().invert_yaxis()
        plt.title("Initial weights fc1")
        plt.savefig(f'{fig_dir}/weights_initial.png')
        plt.figure()
        sns.heatmap(net.fc1.weight.data.cpu().numpy(), square=False, center=0, cmap='vlag')
        plt.gca().invert_yaxis()
        plt.title("Weights fc1")
        plt.savefig(f'{fig_dir}/weights.png')
        plt.figure()
        plt.hist(initial_fc1.flatten(), alpha=0.4, label='initial')
        plt.hist(net.fc1.weight.data.cpu().numpy().flatten(), alpha=0.6, label='trained')
        plt.legend()
        plt.savefig(f'{fig_dir}/weights_hist.png')

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
    plt.title("Loss")
    plt.savefig(f'{fig_dir}/loss.png')

    accs = confusion_matrix.trace() / confusion_matrix.sum()
    confusion_matrix_percentage = confusion_matrix / confusion_matrix.sum(axis=0)
    plt.figure()
    sns.heatmap(
        confusion_matrix_percentage,
        annot=True, fmt=("d" if confusion_matrix_percentage.dtype == int else ".2%"), xticklabels=train_dataset.get_classes(False),
        yticklabels=train_dataset.get_classes(False)
    )
    plt.title(f"Test accuracy = {accs:.2%}")
    plt.xlabel("Target")
    plt.ylabel("Result")
    plt.gca().invert_yaxis()
    plt.yticks(rotation=0)
    plt.savefig(f'{fig_dir}/confusion.png')

    plt.figure()
    plt.plot(np.arange(len(fc1_grad)), fc1_grad, label="FC1 grad")
    plt.scatter(np.arange(len(fc1_grad)), fc1_grad)
    plt.yscale('log')
    plt.title('Linear grad')
    plt.legend()
    plt.savefig(f'{fig_dir}/grad.png')

    shutil.copy('live_accuracy.png', f'{model_dir}/live_accuracy.png')
    shutil.copy('live_weights.png', f'{model_dir}/live_weights.png')

    plt.show()

    print(f'Done! Check model at {model_dir}/')


if __name__ == '__main__':
    main()

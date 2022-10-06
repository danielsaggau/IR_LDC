import argparse
import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


from data_aug.data_loader import CustomDataLoader
from model import Model
from trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"



# costum data loader


num_cls = len(test_loader.dataset.classes)
model = Model(base_model=base_model,
              fc_dim=fc_dim,
              k_subs=args.k_subs,
              layer_sizes=args.layer_size,
              use_bn=True,
              dr_rate=0.2).to(args.device)
print(model)
if torch.cuda.device_count() > 1:
    print("We have available", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

if args.stepwise:
    scheduler = MultiStepLR(optimizer, milestones=[120, 240], gamma=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0,
        last_epoch=-1)

trainer = Trainer(model,
                  optimizer,
                  scheduler,
                  temperature,
                  num_cls,
                  epochs,
                  sigma,
                  lmbda,
                  args.device)

# training loop
results = {'train_loss': [],
           'bloss_loss': [],
           'NTXent_loss': [],
           # 'test_acc@1': [],
           # 'test_acc@5': []
           }
save_name_pre = '{}_K{}_{}_{}_{}_{}_{}_{}_{}'.format(
    dataset_name, args.k_subs,
    base_model, lr,
    fc_dim, temperature,
    k_nn, batch_size, epochs)
csv_dir = os.path.join(writer.log_dir, '{}_stats.csv'.format(save_name_pre))
model_dir = os.path.join(writer.log_dir, '{}_model.pth'.format(save_name_pre))
fig_dir = os.path.join(writer.log_dir, '{}_loss_acc.png'.format(save_name_pre))

best_acc = 0.0
for epoch in range(1, epochs + 1):
    train_loss, bloss, NTXent = trainer.train(train_loader, epoch)
    results['train_loss'].append(train_loss)
    results['bloss_loss'].append(bloss)
    results['NTXent_loss'].append(NTXent)
    writer.add_scalar('loss/train', results['train_loss'][-1], epoch)

    # test_acc_1, test_acc_5 = trainer.test(memory_loader, test_loader, k_nn, epoch)
    # results['test_acc@1'].append(test_acc_1)
    # results['test_acc@5'].append(test_acc_5)
    # writer.add_scalar('acc@1/test', results['test_acc@1'][-1], epoch)
    # writer.add_scalar('acc@5/test', results['test_acc@5'][-1], epoch)

    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(csv_dir, index_label='epoch')

    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, model_dir)

    # if test_acc_1 > best_acc:
    #    best_acc = test_acc_1
    #    if isinstance(model, nn.DataParallel):
    #        state_dict = model.module.state_dict()
    #    else:
    #        state_dict = model.state_dict()
    #    torch.save(state_dict, model_dir)

# plotting loss and accuracies
# df = pd.read_csv(csv_dir)
# fig, axes = plt.subplots(1, 3, sharex=True, figsize=(20,5))
# axes[0].set_title('Loss/Train')
# axes[1].set_title('acc@1/test')
# axes[2].set_title('acc@5/test')
# sns.lineplot(ax=axes[0], x="epoch", y="train_loss", data=df)
# sns.lineplot(ax=axes[1], x="epoch", y="test_acc@1", data=df)
# sns.lineplot(ax=axes[2], x="epoch", y="test_acc@5", data=df)

# fig.savefig(fig_dir)

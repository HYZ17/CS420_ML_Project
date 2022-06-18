import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataloader import MyDataset
import random
import numpy as np
from tqdm import tqdm
import datetime
from models.modelzoo import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_r2cnn import SketchR2CNN
from neuralline.rasterize import Raster
from torch.utils.tensorboard import SummaryWriter
import os
#os.environ['CUDA_VISIBLE_DEVICES']= "2,1,0"
dropout = 0.5
cnn_fn = 'resnet50'
intensity_channels = 1
thickness = 1.0
imgsize = CNN_IMAGE_SIZES[cnn_fn]

Batch_size=32
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr=0.0001
weight_decay = 1e-4
EPOCHS=30

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)

def forward_batch(model, points, point_offsets, point_lengths, labels, mode, optimizer, criterion):
    is_train = mode == 'train'

    points = points.to(device)
    points_offset = point_offsets.to(device)
    points_length = point_lengths
    category = labels.to(device)

    if is_train:
        optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        logits, attention, images = model(points, points_offset, points_length)
        loss = criterion(logits, category)
        if is_train:
            loss.backward()
            optimizer.step()

    return logits, loss, category



if __name__ == '__main__':
    set_seed(17)
    print("We are using",device)
    seq_dir="./dataset"
    categories=['cow', 'panda', 'lion', 'tiger', 'raccoon', 'monkey', 'hedgehog', 'zebra', 'horse', 'owl','elephant', 'squirrel', 'sheep', 'dog', 'bear',
                'kangaroo', 'whale', 'crocodile', 'rhinoceros', 'penguin', 'camel', 'flamingo', 'giraffe', 'pig','cat']
    # categories=['cow','panda']
    logdir = './output'
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{17}_{t0}-_loss'
    log_path = os.path.join(logdir, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str('model_loss'))
    train_dataset=MyDataset(seq_dir,categories,"train",200)
    val_dataset=MyDataset(seq_dir,categories,"valid",200)
    test_dataset=MyDataset(seq_dir,categories,"test",200)

    train_dataset_size=train_dataset.__len__()
    val_dataset_size=val_dataset.__len__()
    test_dataset_size=test_dataset.__len__()

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    print("Finish loading data!")

    model = SketchR2CNN(CNN_MODELS[cnn_fn], 3, dropout, imgsize, thickness, 25, intensity_channels=intensity_channels, device = device)

    optimizer=torch.optim.Adam(model.params_to_optimize(weight_decay, ['bias']), lr=lr)
    loss_func=nn.CrossEntropyLoss()
    best_accuracy=-1
    train_step_counter = 0
    val_step_counter = 0
    for epoch in range(EPOCHS):
        total_train_loss=0
        total_train_acc=0
        model.train_mode()
        for i,batch in enumerate(tqdm(train_dataloader)):
            train_step_counter += 1
            point_batch=batch[0].float().to(device)
            point_offset_batch = batch[1].float().to(device)
            length_batch=batch[2].type(torch.int16).cpu()
            label_batch=batch[3].type(torch.LongTensor).to(device)
            logits, loss, gt_category = forward_batch(model, point_batch, point_offset_batch, length_batch, label_batch, 'train', optimizer, loss_func)
            total_train_loss+=loss.data.item()
            writer.add_scalar("models_loss",
                               torch.mean(loss),
                               train_step_counter,
                               )
            train_pred_y=torch.max(logits.cpu(),1)[1].data.numpy()
            train_accuracy=(train_pred_y==label_batch.cpu().data.numpy()).astype(int).sum()
            writer.add_scalar("models_accuracy",
                               train_accuracy.item()/Batch_size,
                               train_step_counter,
                               )
            total_train_acc+=train_accuracy.item()

        print("Epoch:",epoch," Loss:",total_train_loss/train_dataset_size," Accuracy:",total_train_acc/train_dataset_size)
        if (epoch)%1==0:
            total_val_acc=0
            model.eval_mode()
            for i,batch in enumerate(tqdm(val_dataloader)):
                val_step_counter += 1
                point_batch=batch[0].float().to(device)
                point_offset_batch = batch[1].float().to(device)
                length_batch=batch[2].type(torch.int16).cpu()
                label_batch=batch[3].type(torch.LongTensor).to(device)
                logits, loss, gt_category = forward_batch(model, point_batch, point_offset_batch, length_batch, label_batch, 'val', optimizer, loss_func)
                val_pred_y=torch.max(logits.cpu(),1)[1].data.numpy()
                val_accuracy=(val_pred_y==label_batch.cpu().data.numpy()).astype(int).sum()
                writer.add_scalar("models_validation_accuracy",
                                  val_accuracy.item() / Batch_size,
                                  val_step_counter,
                                  )
                total_val_acc+=val_accuracy.item()
            val_acc=total_val_acc/val_dataset_size

            print("In epoch",epoch,",Validation Accuracy:",val_acc)
            if val_acc>best_accuracy or epoch%10==0:
                best_accuracy=val_acc
                filepath = log_path  # 最终参数模型
                model.save(filepath, train_step_counter, val_acc)

    #Test with specific model
    # model_path=r"C:\Users\DELL\Desktop\CS420_ML_Project\checkpoint_model_epoch_1_acc_0.792208.pth"
    # model=My_model(resnet_type="resnet18",num_classes=25).to(device)
    # load_parameters(model,model_path,device)
    # total_test_acc=0
    # model.eval()
    # for i,batch in enumerate(tqdm(test_dataloader)):
    #     png_batch=batch[0].float().to(device)
    #     png_batch=png_batch/ 255.0 * 2.0 - 1.0
    #     label_batch=batch[3].type(torch.LongTensor).to(device)
    #     seq_batch=batch[1].float().to(device)
    #     length_batch=batch[2].type(torch.int16).cpu()
    #     output=model(png_batch,seq_batch,length_batch)
    #     test_pred_y=torch.max(output.cpu(),1)[1].data.numpy()
    #     test_accuracy=(test_pred_y==label_batch.cpu().data.numpy()).astype(int).sum()
    #     total_test_acc+=test_accuracy.item()
    # test_acc=total_test_acc/test_dataset_size
    # model.train()
    # print("Final Test Accuracy:",test_acc)

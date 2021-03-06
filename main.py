import torch
import torch.nn as nn
from dataloader import MyDataset
import random
import numpy as np
from tqdm import tqdm

from Resnet import My_model,save_parameters,load_parameters
import os

Batch_size=128
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr=0.0001
EPOCHS=300
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)



if __name__ == '__main__':
    set_seed(17)
    print("We are using",device)
    data_dir=r"C:\Users\DELL\Desktop\CS420_ML_Project\picture"
    categories=['cow', 'panda', 'lion', 'tiger', 'raccoon', 'monkey', 'hedgehog', 'zebra', 'horse', 'owl','elephant', 'squirrel', 'sheep', 'dog', 'bear',
                'kangaroo', 'whale', 'crocodile', 'rhinoceros', 'penguin', 'camel', 'flamingo', 'giraffe', 'pig','cat']

    train_dataset=MyDataset(data_dir,categories,"train")
    val_dataset=MyDataset(data_dir,categories,"valid")
    test_dataset=MyDataset(data_dir,categories,"test")

    train_dataset_size=train_dataset.__len__()
    val_dataset_size=val_dataset.__len__()
    test_dataset_size=test_dataset.__len__()

    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    print("Finish loading data!")

    model=My_model(resnet_type="resnet18",num_classes=25).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_func=nn.CrossEntropyLoss()
    best_accuracy=-1

    for epoch in range(EPOCHS):
        total_train_loss=0
        total_train_acc=0
        for i,batch in enumerate(tqdm(train_dataloader)):
            data_batch=batch[0].float().to(device)
            data_batch=data_batch / 255.0 * 2.0 - 1.0
            label_batch=batch[1].type(torch.LongTensor).to(device)
            output=model(data_batch)
            loss=loss_func(output,label_batch)
            total_train_loss+=loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pred_y=torch.max(output.cpu(),1)[1].data.numpy()
            train_accuracy=(train_pred_y==label_batch.cpu().data.numpy()).astype(int).sum()
            total_train_acc+=train_accuracy.item()

        print("Epoch:",epoch," Loss:",total_train_loss/train_dataset_size," Accuracy:",total_train_acc/train_dataset_size)

        if (epoch)%1==0:
            total_val_acc=0
            model.eval()
            for i,batch in enumerate(tqdm(val_dataloader)):
                data_batch=batch[0].float().to(device)
                data_batch=data_batch/ 255.0 * 2.0 - 1.0
                label_batch=batch[1].type(torch.LongTensor).to(device)
                output=model(data_batch)
                val_pred_y=torch.max(output.cpu(),1)[1].data.numpy()
                val_accuracy=(val_pred_y==label_batch.cpu().data.numpy()).astype(int).sum()
                total_val_acc+=val_accuracy.item()
            val_acc=total_val_acc/val_dataset_size

            print("In epoch",epoch,",Validation Accuracy:",val_acc)
            if val_acc>best_accuracy or epoch%10==0:
                best_accuracy=val_acc
                filepath=os.path.join('checkpoint_model_epoch_{}_acc_{}.pth'.format(epoch,val_acc))  #??????????????????
                save_parameters(model,filepath)
            model.train()

    #Test with specific model
    # model_path=r"C:\Users\DELL\Desktop\CS420_ML_Project\checkpoint_model_epoch_0_acc_0.757072.pth"
    # model=My_model(resnet_type="resnet18",num_classes=25).to(device)
    # load_parameters(model,model_path,device)
    # total_test_acc=0
    # model.eval()
    # for i,batch in enumerate(tqdm(test_dataloader)):
    #     data_batch=batch[0].float().to(device)
    #     data_batch=data_batch/ 255.0 * 2.0 - 1.0
    #     label_batch=batch[1].type(torch.LongTensor).to(device)
    #     output=model(data_batch)
    #     test_pred_y=torch.max(output.cpu(),1)[1].data.numpy()
    #     test_accuracy=(test_pred_y==label_batch.cpu().data.numpy()).astype(int).sum()
    #     total_test_acc+=test_accuracy.item()
    # test_acc=total_test_acc/test_dataset_size
    # model.train()
    # print("Final Test Accuracy:",test_acc)

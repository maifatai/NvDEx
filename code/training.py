import torch
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from model import ResNet34
import os

batch_size=32
lr=1e-3
epoches=10000

device=torch.device('cuda')
torch.manual_seed(1234)
train_db,val_db=[],[]
for i in range(100):
    train_db.append([torch.randn(1,3,224,224,224),torch.randn(1)])
    val_db.append([torch.randn(1,3,224,224,224),torch.randn(1)])
# train_db=torch.randn(100,3,224,224,224)
# val_db=torch.randn(32,3,224,224,224)

train_load=DataLoader(train_db,batch_size=batch_size,shuffle=True,num_workers=4)
val_load=DataLoader(val_db,batch_size=batch_size,num_workers=2)
viz=visdom.Visdom()
data_len=len(train_load.dataset)
def main():
    model=ResNet34(2)

    checkpoint_save_path = './checkpoint/best.pkl'
    if os.path.exists(checkpoint_save_path):
        print('........load model........')
        model.load_state_dict(torch.load(checkpoint_save_path))

    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()

    best_acc,best_epoch=0,0

    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    y=torch.randn(100)
    for epoch in range(epoches):
        for step,(x,y) in enumerate(train_load):

            x,y=x.to(device),y.to(device)

            model.train()
            logits=model(x)
            loss=criteon(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()],[epoch],win='loss',opts={'title':'train loss'},update='append')
        #validation
        if epoch %1==0:
            model.eval()
            totle_correct,totle_num=0,len(val_load.dataset)

            for x,y in val_load:
                x, y = x.to(device), y.to(device)
                logits=model(x)
                pred=logits.argmax(dim=1)
                totle_correct+=torch.eq(pred,y).float().sum().item()
            val_acc=totle_correct/totle_num
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),checkpoint_save_path)
                viz.line([val_acc], [epoch], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)




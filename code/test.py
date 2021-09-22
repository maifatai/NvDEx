import torch
from torch.utils.data import DataLoader
from model import ResNet34
import os

def evalute():
    batch_size=32
    device = torch.device('cuda')
    val_db = torch.randn(32, 3, 224, 224, 224)
    val_load = DataLoader(val_db, batch_size=batch_size, num_workers=2)

    model = ResNet34(2)

    checkpoint_save_path = './checkpoint/best.pkl'
    if os.path.exists(checkpoint_save_path):
        print('........load model........')
        model.load_state_dict(torch.load(checkpoint_save_path))

    model.eval()
    totle_correct, totle_num = 0, len(val_load.dataset)

    for x, y in val_load:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        totle_correct += torch.eq(pred, y).float().sum().item()
    val_acc = totle_correct / totle_num

    print("test acc",val_acc)




import torch
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from dataset import CSI_Dataset_Exist
from model import PLA

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TWA', help='model')
    # parser.add_argument('--dataset', type=str, default='D:\desk\dataset\\1228\\', help='dataset')

    parser.add_argument('--train',type=str, default='D:\desk\dataset\\11\\',help='train dataset')
    parser.add_argument('--test',type=str, default='D:\desk\dataset\\12\\',help='test dataset')

    parser.add_argument('--batch', type=int, default=16, help='batch size [default: 16]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001]')
    parser.add_argument('--epoch', type=int, default=50, help='number of epoch [default: 50]')
    parser.add_argument('--category', type=int, default=5, help='category [default: 5]')
    parser.add_argument('--amp', type=bool, default=True, help='use amplitude data')
    parser.add_argument('--phase', type=bool, default=False, help='use phase data')
    parser.add_argument('--tx', type=int, default=2, help='use TX number')
    parser.add_argument('--saveWeight',type=bool,default=False,help='save model weight')
    parser.add_argument('--savePath',type=str,default='./weight',help='save model path')

    return parser.parse_args()


args = get_args()

def train_model(args):
    best_acc = 0.0
    save_path = args.savePath

    train_data = args.train
    test_data = args.test
    batch = args.batch

    # train_dataset = CSI_Dataset_Exist(train_data + 'train')
    # test_dataset = CSI_Dataset_Exist(test_data + 'test')

    train_dataset = torch.load('D:\desk\dataset\com\\train.pt')
    test_dataset = torch.load('D:\desk\dataset\com\\test.pt')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=True)

    num_epoch = args.epoch
    learning_rate = args.lr
    model = PLA(num_classes=2,heads=4)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epoch):

        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for batch in train_bar:
            X_train, Y_train = batch
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)

            optimizer.zero_grad()

            outputs = model(X_train)

            loss = criterion(outputs, Y_train)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()  # 将计算的loss累加到train_loss中

            # desc：str类型，作为进度条说明，在进度条右边
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}.".format(epoch + 1, num_epoch, loss)

        # validate
        model.eval()
        val_acc = 0.0
        val_num = 0
        val_bar = tqdm(test_loader, file=sys.stdout)
        with torch.no_grad():
            for batch in val_bar:
                X_test, Y_test = batch
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                #X(batch szie,seq length,channel),Y(batch size,)
                outputs = model(X_test)
                predict = torch.argmax(outputs, dim=1)
                val_acc += torch.eq(predict, Y_test).sum().item()
                val_num += len(X_test)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, num_epoch)

            # 打印epoch数据结果
            val_accurate = val_acc / val_num
            print("[epoch {:.0f}] val_accuracy: {:.3f}".format(epoch + 1, val_accurate))


if __name__ == '__main__':
    train_model(args)
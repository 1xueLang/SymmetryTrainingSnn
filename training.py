import torch
from torch.cuda import amp
import torch.nn.functional as F

import snetx.snn.algorithm as snnalgo


def train_dvs(net, dataloader, optimizer, criterions, criteriona, scaler, device, args):
    s_correct = a_correct = h_correct = running_loss = sumup = 0.0
    net.train()
    for i, (inputs, labels) in enumerate(dataloader):
        s_inputs = inputs.float().to(device, non_blocking=True)
        a_inputs = s_inputs.mean(dim=1)
        labels = labels.to(device, non_blocking=True)
        if scaler == None:
            sout, aout = net(s_inputs, a_inputs)
            loss = criterions(labels, sout) + criteriona(aout, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with amp.autocast():
                sout, aout = net(s_inputs, a_inputs)
                loss = criterions(labels, sout) + criteriona(aout, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item()
        sumup += inputs.shape[0]

        sout = sout.mean(dim=1)
        s_correct += sout.argmax(dim=1).eq(labels).sum().item()
        a_correct += aout.argmax(dim=1).eq(labels).sum().item()
        h_correct += (sout + aout).argmax(dim=1).eq(labels).sum().item()

        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, '
                  f'snn acc: {(s_correct / sumup) * 100:.2f}%, '
                  f'ann acc: {(a_correct / sumup) * 100:.2f}%, '
                  f'hybrid acc: {(h_correct / sumup) * 100:.2f}%')

    return [s_correct, a_correct, h_correct], sumup, float(running_loss / len(dataloader))


def train_static(net, dataloader, optimizer, criterions, criteriona, scaler, device, args):
    s_correct = a_correct = h_correct = running_loss = sumup = 0.0
    net.train()
    for i, (inputs, labels) in enumerate(dataloader):
        a_inputs = inputs.float().to(device, non_blocking=True)
        s_inputs = snnalgo.temporal_repeat(a_inputs, args.T)
        labels = labels.to(device, non_blocking=True)
        if scaler == None:
            sout, aout = net(s_inputs, a_inputs)
            loss = criterions(labels, sout) + criteriona(aout, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with amp.autocast():
                sout, aout = net(s_inputs, a_inputs)
                loss = criterions(labels, sout) + criteriona(aout, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item()
        sumup += inputs.shape[0]

        sout = sout.mean(dim=1)
        s_correct += sout.argmax(dim=1).eq(labels).sum().item()
        a_correct += aout.argmax(dim=1).eq(labels).sum().item()
        h_correct += (sout + aout).argmax(dim=1).eq(labels).sum().item()

        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, '
                  f'snn acc: {(s_correct / sumup) * 100:.2f}%, '
                  f'ann acc: {(a_correct / sumup) * 100:.2f}%, '
                  f'hybrid acc: {(h_correct / sumup) * 100:.2f}%')

    return [s_correct, a_correct, h_correct], sumup, float(running_loss / len(dataloader))


@torch.no_grad()
def validate(net, dataloader, device, args, static=True):
    net.eval()
    h_correct = s_correct = a_correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if static:
            s_inputs = snnalgo.temporal_repeat(inputs, args.T)
            a_inputs = inputs
        else:
            s_inputs = inputs
            a_inputs = inputs.mean(dim=1)

        sout, aout = net(s_inputs, a_inputs)

        sumup += inputs.shape[0]
        sout = sout.mean(dim=1)
        s_correct += sout.argmax(dim=1).eq(labels).sum().item()
        a_correct += aout.argmax(dim=1).eq(labels).sum().item()
        h_correct += (sout + aout).argmax(dim=1).eq(labels).sum().item()

    net.train()
    return [s_correct, a_correct, h_correct], sumup


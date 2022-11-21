from DocDataset import DocDataset
from Model import ResNet18
from tool import print_confusion_matrix, save_confusion_matrix

import os
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

class Trainer():
    def __init__(self, device, classes, img_size=224, grayscale=False, experiments_dir=None, verbose=True):
        self.img_size = img_size
        self.grayscale = grayscale
        self.device = device
        self.classes = np.array(classes)
        self.experiments_dir = experiments_dir
        self.verbose = verbose

        self.trainloader_rdy = False
        self.valloader_rdy = False
        self.model_rdy = False


    def setup_loaders_with_split(self, set_origin, batch_size, validation_split=.2, shuffle=True):
        ds = DocDataset(set_origin, self.classes, self.img_size, self.grayscale)
        if shuffle:
            indices = torch.randperm(len(ds))
        else:
            indices = list(range(len(ds)))

        # split train/eval
        split = int(np.floor(validation_split * len(indices)))
        train_indices, val_indices = indices[split:], indices[:split]
        # sampler
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        # loaders
        self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
        self.val_loader = DataLoader(ds, batch_size=batch_size, sampler=val_sampler)

        self.trainloader_rdy = True
        self.valloader_rdy = True

        if self.verbose:
            print("Train and Validation loaders created")


    def setup_train_loader(self, trainset_origin, batch_size, shuffle=True):
        trainset = DocDataset(trainset_origin, self.classes, self.img_size, self.grayscale)
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        self.trainloader_rdy = True
        if self.verbose:
            print("Train loader created")


    def setup_val_loader(self, valset_origin, batch_size, shuffle=True):
        valset = DocDataset(valset_origin, self.classes, self.img_size, self.grayscale)
        self.val_loader = DataLoader(valset, batch_size=batch_size, shuffle=shuffle)
        self.valloader_rdy = True
        if self.verbose:
            print("Validation loader created")


    def init_model(self, load_state_dict=None):
        self.model = ResNet18(len(self.classes), self.grayscale)
        if self.verbose:
            print("Model initialized")

        if load_state_dict is not None and os.path.exists(load_state_dict):
            # load pretrained weights if given
            self.model.load_state_dict(torch.load(load_state_dict, map_location=self.device))
            if self.verbose:
                print("Model state dict loaded")

        self.model.to(self.device)
        self.model_rdy = True


    def train(self, epochs, eval_each=None, log=False, show_confusion_matrices=False, save_best_model=False):        
        if not self.trainloader_rdy:
            print("Train loader not created")
            return False
        if not self.valloader_rdy and eval_each is not None:
            print("Validation loader not created")
            return False
        if save_best_model and self.experiments_dir is None:
            print("Requested model save, but experiments dir not specified")
            return False
        if log and self.experiments_dir is None:
            print("Requested logging, but experiments dir not specified")
            return False
        if log and not os.path.exists(self.experiments_dir):
            print("Logging requested, but experiments directory does not exists")
            return False
        if not self.model_rdy:
            print("Model not initialized")        
            return False

        print("Training")

        # logging experiment setting
        logdir = None
        if log:
            # log directory init
            now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            logdir_name = 'start_' + now
            logdir = os.path.join(self.experiments_dir, logdir_name)
            os.mkdir(logdir)                    
            print(f" LOGGING EXPERIMENT to: {logdir}")

            # logfile init
            trainstats_file = os.path.join(logdir, "stats.csv")
            with open(trainstats_file, "a") as fd:
                fd.write("epoch,loss,loss_avg,train_acc,eval_acc\n")            
            

        # training settings
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=3e-3)
        
        # stats accumulator
        stats = pd.DataFrame({
            'epoch': [],
            'loss': [],
            'loss_avg': [],
            'train_acc': [],
            'eval_acc': []
        })        
        best_eval_acc = 0

        # TRAIN LOOP
        for epoch in tqdm(range(epochs)):
            epoch_loss, epoch_hits, epoch_samples = 0, 0, 0
            for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                # input
                input, gt, idx = data
                input = input.to(self.device)
                gt = gt.to(self.device)
                                  
                # forward & backward pass & GD
                optimizer.zero_grad()
                output_logits, output_probs = self.model(input)
                    

                loss = F.cross_entropy(output_logits, gt.type(torch.long))                               
                loss.backward()
                optimizer.step()

                # batch train stats
                pred = torch.argmax(output_logits, axis=1)
                hits = (gt == pred).sum()
                epoch_hits += hits.item()
                epoch_samples += pred.shape[0]                
                epoch_loss += loss.item()                                                
            
            # epoch train stats
            epoch_loss_avg = epoch_loss / (len(self.train_loader) * self.train_loader.batch_size)
            train_acc = epoch_hits / epoch_samples * 100

            if self.verbose:
                print(f" Epoch: {epoch}, loss = {epoch_loss}, loss_avg = {epoch_loss_avg}, train accuracy: {train_acc}%")

            # evaluation during training
            eval_acc = -1
            if eval_each is not None and (epoch+1) % eval_each == 0:
                # evaluation
                eval_acc, eval_stats = self._eval(False)
                                
                if show_confusion_matrices:
                    print_confusion_matrix(eval_stats['gt'].tolist(), eval_stats['pred'].tolist(), self.classes)

                # if best so far - log and save state dict
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc

                    if save_best_model:
                        torch.save(self.model.state_dict(), os.path.join(logdir, f"bestmodel.pth"))
                        save_confusion_matrix(eval_stats['gt'].tolist(), eval_stats['pred'].tolist(), self.classes, os.path.join(logdir, 'best_cfm.png'))
                        
                        bestmodel_acc_file = os.path.join(logdir, 'bestmodel_stats.csv')
                        with open(bestmodel_acc_file, "w") as fd:
                            fd.write("epoch,loss,loss_avg,train_acc,eval_acc\n")
                            fd.write(f"{epoch},{epoch_loss},{epoch_loss_avg},{train_acc},{eval_acc}\n")

            batch_stats = pd.DataFrame({
                'epoch': [epoch],
                'loss': [epoch_loss],
                'loss_avg': [epoch_loss_avg],
                'train_acc': [train_acc],
                'eval_acc': [eval_acc]
            })
            stats = pd.concat([stats, batch_stats], ignore_index=True)
            stats.reset_index()

            # epoch stats log
            with open(trainstats_file, "a") as fd:
                log_eval_acc = eval_acc
                if log_eval_acc == -1:
                    log_eval_acc = ""
                fd.write(f"{epoch},{epoch_loss},{epoch_loss_avg},{train_acc},{log_eval_acc}\n")   


        
        # plot epochs average loss evolution
        if log or self.verbose:
            sns.lineplot(data=stats['loss_avg'].tolist(), palette='YlGnBu')
            plt.title('Avg training loss')
            if log:
                loss_logfile = os.path.join(logdir, 'loss_evo.png')
                plt.savefig(loss_logfile)
            if self.verbose:
                plt.show()
            plt.clf()

        # plot epochs evaluation accuracy evolution            
        if log or self.verbose:
            plt.plot(stats['epoch'], stats['train_acc'], marker='.', color='r', label= 'training accuracy')
            
            valid_eval_accs = [eval_acc for eval_acc in stats['eval_acc'].tolist() if eval_acc != -1]
            valid_eval_epochs = [epoch for epoch, eval_acc in zip(stats['epoch'].tolist(), stats['eval_acc'].tolist()) if eval_acc != -1]
            plt.plot(valid_eval_epochs, valid_eval_accs, marker = '+', color = 'g',label = 'evaluation accuracy')

            plt.legend()
            plt.title('Training and evaluation accuracy evolution')
            if log:
                acc_logfile = os.path.join(logdir, 'acc_evo.png')
                plt.savefig(acc_logfile)
            if self.verbose:
                plt.show()            
            plt.clf()
    
    def _eval(self, filestats_stdout=False):
        print("Evaluation")
        # self.model.eval()
            
        # stats accumulators
        hits_count, samples_count = 0, 0
        stats = {
            'filename': [],
            'gt': [],
            'pred': [],
            'certainty': [],
            'error': []
        }

        # EVAL LOOP
        for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            # inputs
            input, gt, idx = data
            input = input.to(self.device)
            gt = gt.to(self.device)
            
            # forward pass
            output_logits, output_probs = self.model(input)

            # predictions & stats
            pred = torch.argmax(output_logits, axis=1)
            certainty = torch.max(output_probs, axis=1).values
            err = (gt != pred)
            hits = (gt == pred).sum()
            
            hits_count += hits.item()
            samples_count += pred.shape[0]

            stats['filename'].extend(self.val_loader.dataset.items[idx.tolist()])
            stats['gt'].extend(self.classes[gt.tolist()])
            stats['pred'].extend(self.classes[pred.tolist()])
            stats['certainty'].extend(certainty.tolist())
            stats['error'].extend(err.tolist())

            if filestats_stdout:
                for i in range(pred.shape[0]):
                    file_error_symbol = '+'
                    if err[i]:
                        file_error_symbol = '-'
                    print(f"{file_error_symbol} | {self.val_loader.dataset.items[idx[i]]}: gt={self.classes[gt[i]]}, pred={self.classes[pred[i]]}, certainty={certainty[i]:.3f}")

        stats = pd.DataFrame(stats)            
                    
        acc = 0
        if samples_count != 0:
            acc = hits_count / samples_count * 100
            
        if self.verbose:
            print(f" Validation accuracy: {acc}%")
    
        return acc, stats

    def evaluate(self, show_confusion_matrix=True, filestats_stdout=False, logdir=None):
        acc, stats = self._eval(filestats_stdout)
        
        if logdir is not None:
            if not os.path.exists(logdir):
                os.mkdir(logdir)            
            
            filestats_path = os.path.join(logdir, 'filestats.csv')
            stats.to_csv(filestats_path)
            
            acc_path = os.path.join(logdir, 'accuracy.txt')
            with open(acc_path, "w") as fd:
                fd.write(f"Accuracy: {acc}%")

            cfm_path = os.path.join(logdir, 'cfm.png')
            save_confusion_matrix(stats['gt'].tolist(), stats['pred'].tolist(), self.classes, cfm_path)

        if show_confusion_matrix:
            print_confusion_matrix(stats['gt'].tolist(), stats['pred'].tolist(), self.classes)

        return acc, stats

    def predict(self, file_or_folder, batch_size=1, print_results=True, results_file=None, plot=True):
        if not self.model_rdy:
            print("Model not initialized")        
            return False
        if not os.path.exists(file_or_folder):
            print("source for prediction not found")

        print("Resolving samples")

        files = []
        if os.path.isfile(file_or_folder):
            files = [file_or_folder]
        else:            
            for (dir_path, dir_names, file_names) in os.walk(file_or_folder):
                files.extend(file_names)
            p = Path(file_or_folder).glob('**/*')
            files = [x.as_posix() for x in p if x.is_file()]
        print(f" {len(files)} items found")    

        self.model.eval()

        # init results file
        if results_file is not None:
            with open(results_file, "w") as fd:
                fd.write(f"filename, pred, certainty\n")
        
        results = []

        print("Prediction")

        batched_files = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        for batch in batched_files:
            images = []
            for file in batch:
                images.append(DocDataset.img_path_to_input(file, self.img_size, self.grayscale))
            input = torch.stack(images, dim=0)

            input = input.to(self.device)
            output_logits, output_probs = self.model(input)
        
            pred = torch.argmax(output_logits, axis=1)
            certainty = torch.max(output_probs, axis=1).values
            class_names = self.classes[pred.tolist()]

            results.extend(zip(batch, class_names))

            # plot 
            if plot:
                plt.figure(figsize=(30,30))
                for i in range(len(images)):
                    plt.subplot(5,5,i+1).set_title(class_names[i] + " (" + str(round(certainty[i].item()*100,2)) + "%)")
                    plt.imshow(images[i].permute((1,2,0)))
                plt.show()

            # print
            if print_results:
                for i in range(len(batch)):
                    filename = os.path.basename(batch[i])
                    print(f"{filename}, {class_names[i]}, {certainty[i]}\n")

            # save results to file
            if results_file is not None:
                with open(results_file, "a") as fd:
                    for i in range(len(batch)):
                        filename = os.path.basename(batch[i])
                        fd.write(f"{filename}, {class_names[i]}, {certainty[i]}\n")

        return results

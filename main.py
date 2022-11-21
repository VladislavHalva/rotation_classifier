import sys
import argparse
import torch
from Trainer import Trainer

def main():
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print("running on GPU")
    else:
        device = torch.device("cpu")
        print("running on CPU")
    
    parser = argparse.ArgumentParser(
        prog = 'Rotation classifier',
        description = 'Documents Rotation classifier',
        epilog = '')
    parser.add_argument('-m', '--mode', choices=['train', 'eval', 'predict'], required=True)
    parser.add_argument('-t', '--trainset')
    parser.add_argument('-e', '--evalset')
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-l', '--load_model')
    parser.add_argument('-r', '--epochs', type=int)
    parser.add_argument('-d', '--logdir')
    parser.add_argument('-f', '--logfile')
    
    args = parser.parse_args()
    if args.mode == 'train':
        # train
        trainer = Trainer(
            device = device, 
            classes = ['up', 'down', 'left', 'right'], 
            img_size=224, 
            grayscale=False, 
            experiments_dir=args.logdir, 
            verbose=True
        )
        trainer.setup_train_loader(
            trainset_origin = args.trainset, 
            batch_size = args.batch_size,
            shuffle = True
        )

        trainer.setup_val_loader(
            valset_origin = args.evalset, 
            batch_size = args.batch_size, 
            shuffle = True
        )
        if args.load_model is not None:
            trainer.init_model(load_state_dict=args.load_model)
        else:
            trainer.init_model()
            
        trainer.train(
            epochs = args.epochs,
            eval_each = 1, 
            log = True, 
            show_confusion_matrices = False,
            save_best_model = True
        )
    elif args.mode == 'eval':
        # eval
        trainer = Trainer(
            device = device, 
            classes = ['up', 'down', 'left', 'right'], 
            img_size=224, 
            grayscale=False, 
            experiments_dir=args.logdir, 
            verbose=True
        )
        trainer.setup_val_loader(
            valset_origin = args.evalset, 
            batch_size = args.batch_size, 
            shuffle = True
        )
        if args.load_model is not None:
            trainer.init_model(load_state_dict=args.load_model)
        else:
            trainer.init_model()
            
        acc, stats = trainer.evaluate(
            show_confusion_matrix=True, 
            filestats_stdout=False, 
            logdir=args.logdir
        )
    else:
        # predict
        trainer = Trainer(
            device = device, 
            classes = ['up', 'down', 'left', 'right'], 
            img_size=224, 
            grayscale=False, 
            experiments_dir=args.logdir, 
            verbose=True
        )
        trainer.setup_val_loader(
            valset_origin = args.evalset, 
            batch_size = args.batch_size, 
            shuffle = True
        )
        if args.load_model is not None:
            trainer.init_model(load_state_dict=args.load_model)
        else:
            trainer.init_model()
            
        trainer.predict(
            file_or_folder = args.evalset, 
            batch_size=args.batch_size, 
            print_results=True, 
            results_file=args.logfile, 
            plot=False
        )
    

if __name__ == "__main__":
    main()
import os
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.graph_reconstruction import Blip2Stage1
from data_provider.stage1_dm import Stage1DM
from data_provider.stage1_kvplm_dm import Stage1KVPLMDM
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler

from pytorch_lightning.loggers import NeptuneLogger
#import dataclass

from transformers import BertTokenizer
from model.gin_model import GNN_Decoder
import torch.nn.functional as F


os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Graph_reconstruction.load_from_checkpoint(args.init_checkpoint, device=args.devices, args=args)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Graph_reconstruction(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # data
    if args.root.find('kv') >= 0:
        dm = Stage1KVPLMDM(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, args)
    else:
        dm = Stage1DM(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, tokenizer,
                      args)
    model.val_match_loader = dm.val_match_loader
    model.test_match_loader = dm.test_match_loader

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="MolCA/all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1))
    
    if len(args.devices.split(',')) > 1:
        strategy = strategies.DDPStrategy(
            find_unused_parameters=True,
            start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'./MolCA/all_checkpoints/{args.filename}/')
    neptune_logger = NeptuneLogger(
        api_key=os.environ.get('NEPTUNE_API_TOKEN'),
        project="chanhui-lee/text-mol",
    )


    trainer_args = {
        'accelerator': args.accelerator,
        'devices': args.devices,
        'precision': args.precision,
        'max_epochs': args.max_epochs,
        'check_val_every_n_epoch': args.check_val_every_n_epoch,
        'callbacks': callbacks,
        'strategy': strategy,
        'logger': [logger, neptune_logger]
    }

    trainer = Trainer(**trainer_args)
    if args.check_dataset_stats:
        total_stats = dm.check_dataset_stats()
        for _logger in trainer.loggers:
            _logger.log_hyperparams(total_stats)

    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49 ## avoid 
        trainer.validate(model, datamodule=dm)
    elif args.mode == 'test':
        output = trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()
    
from dataclasses import dataclass

@dataclass
class Graph_reconstructionOutput:
    atom_type_loss: torch.FloatTensor
    chirality_tag_loss: torch.FloatTensor
    loss: torch.FloatTensor
    atom_type_prob: torch.FloatTensor
    chirality_tag_prob: torch.FloatTensor

class Graph_reconstruction(pl.LightningModule):
    def __init__(self, args):
        super(Graph_reconstruction, self).__init__()
        self.args = args
        # Add your initialization code here
        self.encoder, self.ln_graph = self.init_graph_encoder(args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio)
        self.decoder = GNN_Decoder(
            num_layer=args.gin_num_layers,
            emb_dim=args.gin_hidden_dim,
            gnn_type='gin',
            drop_ratio=args.drop_ratio,
            JK='last',
        )

        self.save_hyperparameters(args)

    @classmethod
    def init_graph_encoder(
        cls, gin_num_layers, gin_hidden_dim, gin_drop_ratio):
        from model.gin_model import GNN
        graph_encoder = GNN(
            num_layer=gin_num_layers,
            emb_dim=gin_hidden_dim,
            gnn_type='gin',
            drop_ratio=gin_drop_ratio,
            JK='last',
        )
        ckpt = torch.load('MolCA/gin_pretrained/graphcl_80.pth', map_location=torch.device('cpu'))
        print('load graph encoder from MolCA/gin_pretrained/graphcl_80.pth')
        missing_keys, unexpected_keys = graph_encoder.load_state_dict(ckpt, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            print(missing_keys)
            print(unexpected_keys)
        
        ln_graph = LayerNorm(graph_encoder.num_features)
            
        return graph_encoder, ln_graph


    def forward(self, batch):
        # Add your forward pass code here
        graph, text, mask = batch
        batch_node, batch_mask = self.encoder(graph)
        pred = self.decoder(batch_node, batch_mask, graph.edge_index, graph.edge_attr)
        # calculate cross entropy loss
        loss_atom = F.cross_entropy(pred.atom_type_prob, graph.x[:, 0])
        loss_chiral = F.cross_entropy(pred.chirality_tag_prob, graph.x[:, 1])
        loss = loss_atom + loss_chiral
        return Graph_reconstructionOutput(
            atom_type_loss=loss_atom,
            chirality_tag_loss=loss_chiral,
            loss=loss,
            atom_type_prob=pred.atom_type_prob,
            chirality_tag_prob=pred.chirality_tag_prob
        )


    def training_step(self, batch, batch_idx):
        # Add your training step code here
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch[-1].size(0)
        output = self.forward(batch)
        self.log('train_loss', output.loss)
        self.log('train_atom_type_loss', output.atom_type_loss)
        self.log('train_chirality_tag_loss', output.chirality_tag_loss)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch[-1].size(0)
        output = self.forward(batch)
        self.log('val_loss', output.loss, batch_size=batch_size, sync_dist=True)
        self.log('val_atom_type_loss', output.atom_type_loss, batch_size=batch_size, sync_dist=True)
        self.log('val_chirality_tag_loss', output.chirality_tag_loss, batch_size=batch_size, sync_dist=True)
        return output

    def test_step(self, batch, batch_idx):
        batch_size = batch[-1].size(0)
        output = self.forward(batch)
        pred_atom = torch.argmax(output.atom_type_prob, dim=-1)
        pred_chiral = torch.argmax(output.chirality_tag_prob, dim=-1)
        true_atom = batch[0].x[:, 0]
        true_chiral = batch[0].x[:, 1]
        self.log('test_atom_type_acc', (pred_atom == true_atom).float().mean(), batch_size=batch_size, sync_dist=True)
        self.log('test_chirality_tag_acc', (pred_chiral == true_chiral).float().mean(), batch_size=batch_size, sync_dist=True)
        
        self.log('test_loss', output.loss, batch_size=batch_size, sync_dist=True)
        self.log('test_atom_type_loss', output.atom_type_loss, batch_size=batch_size, sync_dist=True)
        self.log('test_chirality_tag_loss', output.chirality_tag_loss, batch_size=batch_size, sync_dist=True)
        return output

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer


class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1_test")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--freeze_encoder',  default=True)

    # added args
    parser.add_argument('--check_dataset_stats', action='store_true', default=False)

    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)

    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)


import logging
import time
import os
from argparse import ArgumentParser
import baseline
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, write_json, Average, get_num_gpus_multiworker
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.layers import save_checkpoint, init_distributed
from eight_mile.downloads import EmbeddingDownloader
from eight_mile.pytorch.optz import *
from eight_mile.progress import create_progress_bar
from transformer_utils import (
    MultiFileDatasetReader,
    NextTurnPredictionFileLoader,
    get_lr_decay,
    AllLoss,
)
from mead.utils import index_by_label


logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class MAPModel(nn.Module):

    def __init__(self, encoder, dropout=0.0, weight_std=0.02):
        super().__init__()
        self.embedding_layers = encoder
        self.d_model = encoder.get_dsz()
        self.top_layer = pytorch_linear(self.d_model, self.d_model, bias=True)
        self.dropout = dropout
        self.weight_std = weight_std
        self.apply(self.init_layer_weights)

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def encode_query(self, query):
        encoded_query = self.embedding_layers(query)
        return encoded_query

    def encode_response(self, response):
        encoded_response = self.embedding_layers(response)
        if self.dropout > 0:
            encoded_response = WithDropout(self.top_layer(encoded_response), pdrop=self.dropout) + encoded_response
        else:
            encoded_response = self.top_layer(encoded_response) + encoded_response
        return encoded_response

    def forward(self, query, response):
        encoded_query = self.encode_query(query)
        encoded_response = self.encode_response(response)
        return encoded_query, encoded_response

    def create_loss(self):
        return AllLoss(self, warmup_steps=100)


def run():
    parser = ArgumentParser("Response selection using MAP method from a pretrained encoder")
    parser.add_argument("--basedir", type=str, default='./map')
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--test_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--embeddings", type=str, help="MEAD embedding config file")
    parser.add_argument("--embedding_label", type=str, help="label of the embedding to use")
    parser.add_argument("--vecs", type=str, help="MEAD vecs config file")
    parser.add_argument("--vec_label", type=str, help="label of the vectorizer to use")
    parser.add_argument("--nctx", type=int, default=64, help="Max input length")
    parser.add_argument("--pattern", default='*.txt', help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="The batch size in fine-tuning. The testing metric is the accuracy of ranking the true"
                             "response in top n from batch size number of candidates")
    parser.add_argument("--top_n", type=int, default=1,
                        help="if the true response is ranked top n, it will be considered correct in calculating acc")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0.1, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Num warmup steps")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=4.0e-5, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Num training epochs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, help="write results to a file", default='./results.txt')
    args = parser.parse_args()

    vecs = read_config_stream(args.vecs)
    vecs_set = index_by_label(vecs)
    vecs_config = vecs_set[args.vec_label]
    vectorizer = baseline.vectorizers.create_vectorizer(vectorizer_type=vecs_config['type'], **vecs_config)

    embeddings = read_config_stream(args.embeddings)
    embeddings_set = index_by_label(embeddings)
    embeddings_config = embeddings_set[args.embedding_label]
    embed_file = embeddings_config.get('file')
    unzip_file = embeddings_config.get('unzip', True)
    embed_dsz = embeddings_config['dsz']
    embed_sha1 = embeddings_config.get('sha1')
    embed_file = EmbeddingDownloader(embed_file, embed_dsz, embed_sha1, os.path.expanduser("~/.bl-data"), unzip_file=unzip_file).download()
    preproc_data = baseline.embeddings.load_embeddings('encoder',
                                                       embed_type=embeddings_config['type'],
                                                       embed_file=embed_file,
                                                       **embeddings_config['model'])
    encoder = preproc_data['embeddings']
    vocab = preproc_data['vocab']

    train_set = NextTurnPredictionFileLoader(args.train_file, args.pattern, vocab, vectorizer, args.nctx, distribute=False, shuffle=True)
    test_set = NextTurnPredictionFileLoader(args.test_file, args.pattern, vocab, vectorizer, args.nctx, distribute=False, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    train_steps = len(train_loader) // args.batch_size
    test_steps = len(test_loader) // args.batch_size

    model = MAPModel(encoder, dropout=args.dropout)
    model.to(args.device)
    loss_function = model.create_loss()
    loss_function.to(args.device)
    logger.info("Loaded model and loss")

    lr_decay = get_lr_decay(args.lr_scheduler, args.lr, train_steps, args.epochs, logger,
                            decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay_rate, alpha=args.lr_alpha)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)
    optimizer = OptimizerManager(model, 0, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    model_base = os.path.join(args.basedir, 'checkpoint')
    steps = 0

    # fine-tuning loop
    for epoch in range(args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()
        start = time.time()
        model.train()
        train_itr = iter(train_loader)
        pg = create_progress_bar(train_steps)
        for i in range(train_steps):
            batch = next(train_itr)
            steps += 1
            x, y = batch
            inputs = x.to(args.device)
            labels = y.to(args.device)
            loss = loss_function(inputs, labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            pg.update()
        pg.done()
        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_avg_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_avg_loss
        metrics['lr'] = optimizer.current_lr
        logger.info(metrics)

    # testing
        numerator = 0
        denominator = 0
        metrics = {}
        start = time.time()
        model.eval()
        test_itr = iter(test_loader)
        pg = create_progress_bar(test_steps)
        for i in range(test_steps):
            batch = next(test_itr)
            if batch[0].shape[0] != args.batch_size:
                break
            with torch.no_grad():
                x, y = batch
                inputs = x.to(args.device)
                targets = y.to(args.device)
                query = model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
                response = model.encode_response(targets).unsqueeze(0)  # [1, B, H]
                all_score = nn.CosineSimilarity(dim=-1)(query, response).to('cpu')

                _, indices = torch.topk(all_score, args.top_n, dim=1)
                correct = (indices == torch.arange(args.batch_size).unsqueeze(1).expand(-1, args.top_n)).sum()
                numerator += correct
                logger.debug(f"Selected {correct} correct responses out of {args.batch_size}")
                denominator += args.batch_size
                pg.update()
        pg.done()
        acc = float(numerator)/denominator
        metrics["test_elapsed_min"] = (time.time() - start) / 60
        metrics[f"{args.top_n}@{args.batch_size} acc"] = acc
        logger.info(metrics)

    with open(args.output, 'a') as wf:
        wf.write(f"Encoder: {args.embedding_label}; {args.top_n}@{args.batch_size} accuracy: {acc}\n")


if __name__ == "__main__":
    run()

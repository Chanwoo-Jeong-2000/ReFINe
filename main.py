import torch
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything

from tqdm import tqdm
import argparse
import os
import time

import data_loader, utils

#############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=2024)
parser.add_argument('--gpu_id', type=int, default=4)
parser.add_argument('--dataset', type=str, default='ML-100K')
parser.add_argument('--dataset_augment', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=8192)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--evaluation_step', type=int, default=1)
parser.add_argument('--early_stopping_step', type=int, default=50)
parser.add_argument('--top_k', type=list, default=[5,10,15,20])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--real_neg_samp_prob', type=float, default=1.5, help='real_negative_sampling_probabilities')
parser.add_argument('--p', type=float, default=1.5)
parser.add_argument('--q', type=float, default=0.75)
parser.add_argument('--r', type=int, default=100)
parser.add_argument('--s', type=int, default=5)
parser.add_argument('--t', type=int, default=50)
parser.add_argument('--path_name', type=str, default='nothing')
args = parser.parse_args()
#############################################################################

seed_everything(args.random_seed)
gpu = 'cuda:'+str(args.gpu_id)
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

print('#############################################################################')
print('ramdom_seed:', args.random_seed)
print('gpu_id:', args.gpu_id)
print('dataset:', args.dataset)
print('dataset_augment:', args.dataset_augment)
print('batch_size:', args.batch_size)
print('test_batch_size:', args.test_batch_size)
print('embedding_dim:', args.embedding_dim)
print('layers:', args.layers)
print('learning_rate:', args.learning_rate)
print('evaluation_step:', args.evaluation_step)
print('early_stopping_step:', args.early_stopping_step)
print('top_k:', args.top_k)
print('epochs:', args.epochs)
print('real_negative_sampling_probabilities:', args.real_neg_samp_prob)
print('p:', args.p)
print('q:', args.q)
print('r:', args.r)
print('s:', args.s)
print('t:', args.t)
print('path_name:', args.path_name)
print('#############################################################################\n')

print('#############################################################################')
print('data loading...')
if args.dataset_augment == 0:
    print('train set: train_original')
else:
    print('train set: train_augment')
data, data_neg, data_neutral = data_loader.data_loading(args.dataset, args.dataset_augment, 'val')
num_users, num_items = data['user'].num_nodes, data['item'].num_nodes
data = data.to_homogeneous().to(device)
data_neg = data_neg.to_homogeneous().to(device)
data_neutral = data_neutral.to_homogeneous().to(device)
print('done!')
print('#############################################################################')


batch_size = args.batch_size
test_batch_size = args.test_batch_size

mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size)

mask_neg = data_neg.edge_index[0] < data_neg.edge_index[1]
train_neg_edge_label_index = data_neg.edge_index[:, mask_neg]
'''train_neg_loader = torch.utils.data.DataLoader(
    range(train_neg_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size)'''

mask_neutral = data_neutral.edge_index[0] < data_neutral.edge_index[1]
train_neutral_edge_label_index = data_neutral.edge_index[:, mask_neutral]

model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=args.embedding_dim,
    num_layers=args.layers,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

negative_sampling_probabilities = torch.ones(num_users, num_items, device=device)
negative_sampling_probabilities[train_edge_label_index[0], train_edge_label_index[1]-num_users] = 0.
negative_sampling_probabilities[train_neg_edge_label_index[0], train_neg_edge_label_index[1]-num_users] = args.real_neg_samp_prob
#negative_sampling_probabilities[train_neutral_edge_label_index[0], train_neutral_edge_label_index[1]-num_users] = 1.  #cj neutral


def train(epoch: int):
    total_loss = total_examples = 0
    negative_sampling_probabilities_clone = negative_sampling_probabilities.clone()

    if epoch >= args.t:
        emb = model.get_embedding(data.edge_index)
        item_emb = emb[num_users:]
        item_item = item_emb @ item_emb.t()
        item_item.fill_diagonal_(float('-inf'))

        similar_k = args.s
        _, topk_indices = torch.topk(item_item, k=similar_k, dim=1)
        user_indices = train_edge_label_index[0].repeat_interleave(similar_k)
        item_indices = train_edge_label_index[1] - num_users
        topk_item_indices = topk_indices[item_indices].view(-1)
        sampling_mask = negative_sampling_probabilities_clone[user_indices, topk_item_indices] == 1.
        negative_sampling_probabilities_clone[user_indices[sampling_mask], topk_item_indices[sampling_mask]] = args.q

        neg_similar_k = args.r
        _, neg_topk_indices = torch.topk(item_item, k=neg_similar_k, dim=1)
        user_indices = train_neg_edge_label_index[0].repeat_interleave(neg_similar_k)
        item_indices = train_neg_edge_label_index[1] - num_users
        neg_topk_item_indices = neg_topk_indices[item_indices].view(-1)
        sampling_mask = negative_sampling_probabilities_clone[user_indices, neg_topk_item_indices] == 1.
        negative_sampling_probabilities_clone[user_indices[sampling_mask], neg_topk_item_indices[sampling_mask]] = args.p


    for index in train_loader:  #for index in tqdm(train_loader):
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([pos_edge_label_index[0], torch.multinomial(negative_sampling_probabilities_clone[pos_edge_label_index[0]], num_samples=1).squeeze() + num_users], dim=0)
        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)
        loss = model.recommendation_loss(pos_rank, neg_rank, node_id=edge_label_index.unique())
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()
        

    return total_loss / total_examples


@torch.no_grad()
def test(ks: list):
    emb = model.get_embedding(data.edge_index)
    user_emb, item_emb = emb[:num_users], emb[num_users:]

    results = list()
    for k in ks:
        precision = recall = ndcg = total_examples = 0
        for start in range(0, num_users, test_batch_size):
            end = start + test_batch_size
            logits = user_emb[start:end] @ item_emb.t()

            mask = ((train_edge_label_index[0] >= start) &
                    (train_edge_label_index[0] < end))
            logits[train_edge_label_index[0, mask] - start,
                train_edge_label_index[1, mask] - num_users] = float('-inf')
            mask_neg = ((train_neg_edge_label_index[0] >= start) &
                    (train_neg_edge_label_index[0] < end))
            logits[train_neg_edge_label_index[0, mask_neg] - start,
                train_neg_edge_label_index[1, mask_neg] - num_users] = float('-inf')
            mask_neutral = ((train_neutral_edge_label_index[0] >= start) &
                    (train_neutral_edge_label_index[0] < end))
            logits[train_neutral_edge_label_index[0, mask_neutral] - start,
                train_neutral_edge_label_index[1, mask_neutral] - num_users] = float('-inf')
                
            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            mask = ((data.edge_label_index[0] >= start) &
                    (data.edge_label_index[0] < end))
            ground_truth[data.edge_label_index[0, mask] - start,
                        data.edge_label_index[1, mask] - num_users] = True
            node_count = degree(data.edge_label_index[0, mask] - start,
                                num_nodes=logits.size(0))

            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)

            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

            relevance_scores = ground_truth.float()
            ideal_relevance_scores = relevance_scores.sort(dim=1, descending=True).values[:, :k]
            log2_k = torch.log2(torch.arange(2, k+2, device=logits.device, dtype=torch.float))
            dcg_scores = (relevance_scores.gather(1, topk_index) / log2_k).sum(dim=1)
            ideal_dcg_scores = (ideal_relevance_scores / log2_k).sum(dim=1)
            ndcg += float((dcg_scores / ideal_dcg_scores.clamp(1e-6)).sum())

            total_examples += int((node_count > 0).sum())

        results.append((precision / total_examples, recall / total_examples, ndcg / total_examples))

    return results


if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('result/'+args.dataset):
    os.makedirs('result/'+args.dataset)

path_name = 'result/'+args.dataset+'/'+args.path_name+'.pt'
early_stopping = utils.EarlyStopping(patience=args.early_stopping_step, verbose=True, path=path_name)

topks = args.top_k
start_time = time.time()
for epoch in range(1, args.epochs+1):
    loss = train(epoch=epoch)

    if epoch % args.evaluation_step == 0:
        results = test(ks=topks)

        print(f'\nEpoch: {epoch:03d}, '
            f'Loss: {loss:.4f}')
        for k, (precision, recall, ndcg) in zip(topks, results):
            if k < 10:
                print(f'Precision@{k}: {precision:7.4f}, '
                      f'Recall@{k}: {recall:7.4f}, '
                      f'NDCG@{k}: {ndcg:7.4f}')
            else:
                print(f'Precision@{k}: {precision:.4f}, '
                      f'Recall@{k}: {recall:.4f}, '
                      f'NDCG@{k}: {ndcg:.4f}')

        early_stopping(epoch, results, model)

        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {early_stopping.best_epoch}. Best Validation Results:")
            for k, (precision, recall, ndcg) in zip(topks, early_stopping.best_metrics):
                if k < 10:
                    print(f'Precision@{k}: {precision:7.4f}, '
                          f'Recall@{k}: {recall:7.4f}, '
                          f'NDCG@{k}: {ndcg:7.4f}')
                else:
                    print(f'Precision@{k}: {precision:.4f}, '
                          f'Recall@{k}: {recall:.4f}, '
                          f'NDCG@{k}: {ndcg:.4f}')
            break
end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining completed in {training_time} seconds.")

if not early_stopping.early_stop:
    path_name = 'result/'+args.dataset+'_not_early_stop.pt'
    torch.save(model.state_dict(), path_name)
    print("Early stopping does not trigger.")

else:
    data, _, _ = data_loader.data_loading(args.dataset, args.dataset_augment, 'test')
    num_users, num_items = data['user'].num_nodes, data['item'].num_nodes
    data = data.to_homogeneous().to(device)

    model.load_state_dict(torch.load(path_name, weights_only=True))
    model.eval()
    print('\n#############################################################################')
    print('Final Test Results')
    results = test(ks=topks)
    for k, (precision, recall, ndcg) in zip(topks, results):
            if k < 10:
                print(f'Precision@{k}: {precision:7.4f}, '
                      f'Recall@{k}: {recall:7.4f}, '
                      f'NDCG@{k}: {ndcg:7.4f}')
            else:
                print(f'Precision@{k}: {precision:.4f}, '
                      f'Recall@{k}: {recall:.4f}, '
                      f'NDCG@{k}: {ndcg:.4f}')
    print('#############################################################################')


exit()
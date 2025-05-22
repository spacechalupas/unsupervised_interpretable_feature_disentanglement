import argparse
import csv
from torch.utils.data import DataLoader
import torch 
from attribute_finder_model_rs import AttributeFinder, distributed_sinkhorn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from train_classification import HDF5Dataset, influences#, vocab_levels
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(params):
    num_sub_prototypes = 1

    title = "Animal"
    persona_ds = params.inst_path
    attr_name = params.attr_path

    full_dataset = HDF5Dataset(params.db_path, attr_name, persona_ds)
    # full_dataset = train.HDF5Dataset("../data/attention_outputs_personas.h5", truth, [0])
    batch_size = 64

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    num_heads = params.num_heads
    head_dim = params.head_dim
    mid_dim = params.mid_dim
    num_features = params.num_features
    num_slots = params.num_slots
    threshold = 0.01
    
    num_classes = len(full_dataset.attr)

    attribute_finder = AttributeFinder(head_dim, mid_dim, num_features, num_heads, num_classes, num_slots).to(device)#, num_sub_prototypes).to(device)
    optimizer = optim.Adam(attribute_finder.parameters(), lr=0.001)
    attribute_finder.train()

    epochs = 100
    best_reconstruction_loss = 0
    best_heads = None
    for epoch in range(epochs):
        for _, X_batch, X_attr, y_batch, y_batch2 in train_loader:
            X_batch = X_batch.cuda() if torch.cuda.is_available() else X_batch
            X_attr = X_attr.cuda() if torch.cuda.is_available() else X_attr.to(torch.bfloat16)
            y_batch = y_batch.cuda()if torch.cuda.is_available() else y_batch
            y_batch2 = y_batch2.cuda() if torch.cuda.is_available() else y_batch2
            _, _, head_dim = X_batch.shape

            with torch.no_grad():
                w = attribute_finder.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                attribute_finder.prototypes.weight.copy_(w)

            optimizer.zero_grad()
            

            out, attr_out, pers_prototype_dist, attr_prototype_dist, attn_weight = attribute_finder(X_batch, X_attr)
            
            persona_loss = 0
            total_num = 0
            for i in range(num_slots):
                with torch.no_grad():
                    sinkhorn =  distributed_sinkhorn(pers_prototype_dist[:, i]) #F.softmax(pers_prototype_dist[:, i]/0.7, dim=-1)#
                for j in range(num_slots):
                    if i == j: continue
                    persona_loss -= torch.mean(torch.sum(sinkhorn * F.log_softmax(pers_prototype_dist[:, j], dim=-1), dim=-1))
                    total_num += 1

            loss_attn = 0
            prot_loss = 0
            for slot in range(num_slots):
                loss_attn += pairwise_jsd_loss(attn_weight[:, slot])
                prot_loss += F.cross_entropy(attr_prototype_dist[:, slot], y_batch2.detach())

            div_loss = off_diagonal_cosine_loss(out) + off_diagonal_cosine_loss(attr_out)

            loss_attn /= num_slots
            prot_loss /= num_slots
            persona_loss /= total_num
            div_loss /= num_slots

            print(loss_attn.detach(), prot_loss.detach(), persona_loss.detach())
            total_loss = loss_attn + prot_loss  + persona_loss  + div_loss 
                
            total_loss.backward()
            optimizer.step()

        total_samples = 0
        with torch.no_grad():

            w = attribute_finder.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            attribute_finder.prototypes.weight.copy_(w)

            all_int_attn_weights = []
            reconstruction_loss = 0
            attr_loss = 0
            answers = []
            percentages = []
            outputs = []
            index_str_outputs = []
            true_labels = []
            pred_labels = []
            for ind, X_batch, X_attr, y_batch, y_batch2 in val_loader:
                X_batch = X_batch.cuda()
                X_attr = X_attr.cuda() if torch.cuda.is_available() else X_attr

                y_batch = y_batch.cuda()
                y_batch2 = y_batch2.cuda()

                b_size, _, _ = X_batch.shape

                out, attr_out, pers_prototype_dist, attr_prototype_dist, attn_weight1  = attribute_finder(X_batch, X_attr) #, , out_influence_probe

                pers_prototype_dist = F.softmax(pers_prototype_dist, dim=-1)
                best_guesses1 = torch.argmax(pers_prototype_dist[:, 0], dim=-1)

                pred_labels.append(best_guesses1.squeeze().cpu().numpy())
                true_labels.append(y_batch.squeeze().cpu().numpy())
                reconstruction_loss += (best_guesses1.squeeze() == y_batch.squeeze()).sum().item()
                outputs.append(out.cpu().numpy())
                index_str_outputs.append(full_dataset.attr[y_batch.cpu().numpy()*num_sub_prototypes])
                for x, (i, guess1) in enumerate(zip(ind, best_guesses1)):
                    answers.append([full_dataset.persona_prompt.iloc[i.item()], full_dataset.attr.iloc[guess1.item()], full_dataset.persona_ans.iloc[i.item()]])
                    percentages.append([x.item() for x in pers_prototype_dist[x, 0]])
                    # print(answers)
                all_int_attn_weights.append(attn_weight1)
                total_samples += y_batch.size(0)


        print(reconstruction_loss, total_samples)
        reconstruction_loss /= total_samples
        attr_loss /= total_samples

        all_int_attn_weights = torch.cat(all_int_attn_weights, dim=0)
        if reconstruction_loss > best_reconstruction_loss:
            outputs = np.concatenate(outputs, axis=0)[:, 0, :]
            pred_labels = np.concatenate(pred_labels, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            ari = adjusted_rand_score(true_labels, pred_labels)

            score = 0
            normalized_outputs = normalize(outputs)
            kmeans = KMeans(n_clusters=num_classes, random_state=42)
            labels = kmeans.fit_predict(normalized_outputs)            
            score = silhouette_score(outputs, labels, metric="cosine")
            
            index_str_outputs = np.concatenate(index_str_outputs, axis=0)

            plot_pca(outputs, index_str_outputs, title=f"{title} Clustering")
            plot_tsne(outputs, index_str_outputs, title=f"{title} Clustering")
            best_reconstruction_loss = reconstruction_loss
            best_heads = torch.nonzero(all_int_attn_weights.mean(dim=0) >= threshold, as_tuple=True)#.detach().cpu().numpy()
            print("Best Accuracy", best_reconstruction_loss, attr_loss)
            print(torch.where(all_int_attn_weights.mean(dim=0) >= threshold))
            print(torch.topk(all_int_attn_weights.mean(dim=0), 20, largest=True).indices)
            for slot in range(num_slots):
                plot_slot_head_assignment(all_int_attn_weights[:, slot], f"int_attn_weights_{slot}")
            torch.save(attribute_finder.state_dict(), f"../data/{persona_ds}/attibute_finder.pth")
            with open('model_answers.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(answers)#, np.array2string(best_heads[1].cpu().detach().numpy())])
            with open('percentages.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(percentages)
            with open('slot_heads.txt', 'w') as output:
                output.writelines([np.array2string(best_heads[1].cpu().numpy())])#, np.array2string(best_heads[1].cpu().detach().numpy())])
                output.write(f"\n\n{str(best_reconstruction_loss)}")
                output.write(f"\nARI: {str(ari)}")
                output.write(f"\nSillhoutte: {str(score)}")

        outputs = []
        print(f"Epoch {epoch+1}/{epochs} completed.")
    print("BEST ACCURACY", best_reconstruction_loss)
    print(best_heads)


def off_diagonal_cosine_loss(x):
    # x: [B, n, d]
    x = F.normalize(x, dim=-1)  # normalize for cosine similarity
    B, n, d = x.shape

    # Compute similarity matrix for each sample
    sim_matrix = torch.matmul(x, x.transpose(1, 2))  # [B, n, n]

    # Remove diagonal (self-similarity)
    off_diag = sim_matrix - torch.eye(n, device=x.device)[None]  # [B, n, n]

    # Take absolute value and mean over off-diagonal elements
    loss = off_diag.abs().sum(dim=(1, 2)) / (n * (n - 1))  # mean over pairs per sample
    return loss.mean()  # mean over batch


def plot_slot_head_assignment(slot_attn_weights, title, init=0, end=None):
    """
    Visualizes the slot-to-head assignment as a heatmap.
    slot_attn_weights: Tensor of shape [batch_size, num_slots, num_heads]
    """
    # Average over batch (if batch size > 1)

    avg_weights = slot_attn_weights.mean(dim=0, keepdim=True).detach().cpu().numpy()
    avg_weights = avg_weights.reshape((32, 32), order='F')

    plt.figure(figsize=(6,6))
    
    # sns.heatmap(avg_weights, cmap="viridis", cbar=True, ) #yticklabels=[f"Slot {i}" for i in range(avg_weights.shape[0])])

    fig, ax = plt.subplots()

    cax = ax.imshow(avg_weights, cmap='Blues', interpolation="none", aspect="equal")
    ax.grid(False)

    ax.yaxis.set_ticks_position('left')

    if end is not None:
        plt.xticks(ticks=np.arange(10), labels=np.linspace(init, end, 10, dtype=int))  

    ax.set_xlabel("Layers")
    ax.set_ylabel("Attention Heads")
    cbar = plt.colorbar(cax, ax=ax)
    ticks = cbar.get_ticks()
    cbar.set_ticks([ticks[1], ticks[len(ticks)//2], ticks[-2]])  # Keep only the highest tick
    
    plt.subplots_adjust(top=0.2)
    plt.tight_layout()
    plt.title("Attention Weights")
    plt.savefig(title)
    plt.clf()
    plt.close()
    # plt.show()

def pairwise_jsd_loss(x, eps=1e-8):
    """
    Computes the mean JSD between all pairs in the batch.
    Args:
        x: tensor of shape (batch_size, num_heads), rows must be valid probability distributions.
    Returns:
        Scalar JSD loss.
    """
    B = x.size(0)

    jsd_matrix = torch.zeros(B, B, device=x.device)

    for i in range(B):
        for j in range(i + 1, B):
            p, q = x[i], x[j]
            m = 0.5 * (p + q)
            kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
            kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))
            jsd = 0.5 * (kl_pm + kl_qm)
            jsd_matrix[i, j] = jsd
            jsd_matrix[j, i] = jsd  # symmetric

    # Average over all unique pairs
    loss = jsd_matrix.sum() / (B * (B - 1))
    return loss


def plot_pca(X, labels=None, n_components=2, scale_data=True, title='PCA Plot', figsize=(8, 6),
             point_size=60, alpha=0.8, edge_color='k'):
    """
    Pretty PCA plot with optional labels (for coloring), scaling, and styling.

    Parameters:
    - X: (n_samples, n_features)
    - labels: Optional (n_samples,) array of class labels
    - n_components: 2 or 3
    - scale_data: If True, standardizes the data
    - title: Plot title
    - figsize: Figure size
    - point_size: Size of each scatter point
    - alpha: Transparency
    - edge_color: Outline color for points
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)

    print(labels.shape)
    if len(X) > len(labels):
        labels = np.concatenate([labels, np.full((len(X) - len(labels), ), "Prototypes")], axis=0)

    if scale_data:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    plt.grid(False)
    if n_components == 2:
        plt.figure(figsize=figsize)
        if labels is not None:
            unique_labels = np.unique(labels)
            palette = sns.color_palette("husl", len(unique_labels))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                            label=label, 
                            s=point_size, alpha=alpha, 
                            edgecolor=edge_color, 
                            color=palette[i])
            plt.legend(title="True Label", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], s=point_size, alpha=alpha, edgecolor=edge_color)
        
        plt.grid(False)
        plt.xlabel(f'PC 1')
        plt.ylabel(f'PC 2')
        plt.title(title)
        plt.tight_layout()
        # plt.show()
        plt.savefig("pca_elements.png")
    
    else:
        raise NotImplementedError("Only 2D PCA plotting is supported in this version.")
    
def plot_tsne(X, labels, label_names=None, title="Clustering"):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')  # You can try 'euclidean' too
    X_embedded = reducer.fit_transform(X)  # shape: [num_points, 2]

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 7), dpi=120)

    if len(X) > len(labels):
        labels = np.concatenate([labels, np.full((len(X) - len(labels), ), "Prototypes")], axis=0)

    unique_labels = np.unique(labels)


    palette = sns.color_palette("husl", n_colors=len(unique_labels))


    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names is not None else label
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=name, s=60, alpha=0.8, edgecolor='k', color=palette[i])  # if you have labels
    
    plt.grid(False)
    plt.title(title)
    plt.xlabel('PC-1')
    plt.ylabel('PC-2')
    plt.legend(title='True Labels', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    plt.savefig("tsne_elements.png")

def parse_model_params():
    parser = argparse.ArgumentParser(description="Model parameter input")

    parser.add_argument("--num_heads", type=int, default=1024, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Dimension of each head")
    parser.add_argument("--num_features", type=int, default=128, help="Number of features")
    parser.add_argument("--num_slots", type=int, default=2, help="Number of slots")
    parser.add_argument("--mid_dim", type=int, default=256, help="Intermediate dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--attr_name", type=str, required=True, help="Name of the Attribute dataset")
    parser.add_argument("--inst_name", type=str, required=True, help="Name of the Instance dataset")
    parser.add_argument("--db_path", type=str, default="../data/attention_outputs_personas.h5", help="Path of the database")

    args = parser.parse_args()

    return {
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "num_features": args.num_features,
        "num_slots": args.num_slots,
        "mid_dim": args.mid_dim,
        "batch_size": args.batch_size,
        "attr_name": args.attr_name,
        "inst_name": args.inst_name
    }


if __name__ == "__main__":
    params = parse_model_params()
    main(params)
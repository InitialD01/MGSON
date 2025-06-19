import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pywt
from sklearn import preprocessing


# ����������������������������
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ��������
def open_data(base_path, key_num):
    path = os.path.join(base_path, f"{key_num}.mat")
    print(f"Attempting to open file: {path}")
    key = f"X{key_num:03d}_DE_time"
    data = scio.loadmat(path)
    if key not in data:
        raise KeyError(f"Key {key} not found in the .mat file.")
    return data[key]


# ����������������������������������������
def deal_data(data, length, label, snr_db):
    data = np.reshape(data, (-1,))
    num = len(data) // length
    data = data[:num * length]
    data = np.reshape(data, (num, length))

    # ������������������
    fft_result = np.fft.fft(data, axis=1)
    psd = np.abs(fft_result)**2 / length
    avg_psd = np.mean(psd, axis=0)

    # ����SNR������������
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(avg_psd)
    noise_power = signal_power / snr_linear

    # ����������������������������
    noise_fft = np.sqrt(noise_power / 2) * (np.random.normal(size=fft_result.shape) + 1j * np.random.normal(size=fft_result.shape))
    noise = np.fft.ifft(noise_fft, axis=1).real

    # ��������
    noisy_data = data + noise

    # MaxAbsScaler ������
    maxabs_scaler = preprocessing.MaxAbsScaler()
    noisy_data = np.transpose(noisy_data, [1, 0])
    noisy_data = maxabs_scaler.fit_transform(noisy_data)
    noisy_data = np.transpose(noisy_data, [1, 0])

    labels = np.ones((num, 1)) * label
    return np.column_stack((noisy_data, labels))


# ����������
def split_data(data, split_rate):
    train_size = split_rate[0]
    eval_size = split_rate[1]

    train_data, temp_data = train_test_split(data, train_size=train_size, shuffle=True)
    eval_data, test_data = train_test_split(temp_data, test_size=1 - eval_size / (1 - train_size), shuffle=True)
    return train_data, eval_data, test_data


# ����CWRU������
def load_cwru_data(num, length, hp, fault_diameter, split_rate):
    bath_path1 = r"/root/Desktop/Mine/Normal Baseline"
    bath_path2 = r"/root/Desktop/Mine/12k Drive End Bearing Fault Data"

    data_list = []
    label = 0

    # ��������
    normal_data = open_data(bath_path1, 97)
    data = deal_data(normal_data, length, label=label)
    data_list.append(data)

    # ��������
    for i in hp:
        for j in fault_diameter:
            if j == 0.007:
                inner_num = 105
                ball_num = 118
                outer_num = 130
            elif j == 0.014:
                inner_num = 169
                ball_num = 185
                outer_num = 197
            else:
                inner_num = 209
                ball_num = 222
                outer_num = 234

            for fault_type, num in zip(['inner', 'ball', 'outer'], [inner_num, ball_num, outer_num]):
                fault_data = open_data(bath_path2, num + i)
                fault_data = deal_data(fault_data, length, label + (['inner', 'ball', 'outer'].index(fault_type) + 1))
                data_list.append(fault_data)

        label += 3

    # ����������
    min_num = min(len(d) for d in data_list)
    min_num = min(num, min_num)

    train, eval, test = [], [], []
    for data in data_list:
        data = data[:min_num]
        t, e, te = split_data(data, split_rate)
        train.append(t)
        eval.append(e)
        test.append(te)

    train = np.vstack(train)
    eval = np.vstack(eval)
    test = np.vstack(test)

    train = train[random.sample(range(len(train)), len(train))]
    eval = eval[random.sample(range(len(eval)), len(eval))]
    test = test[random.sample(range(len(test)), len(test))]

    train_data, train_label = train[:, :length], torch.zeros(len(train), 10).scatter_(1, torch.LongTensor(
        train[:, length].astype(int)).unsqueeze(1), 1)
    eval_data, eval_label = eval[:, :length], torch.zeros(len(eval), 10).scatter_(1, torch.LongTensor(
        eval[:, length].astype(int)).unsqueeze(1), 1)
    test_data, test_label = test[:, :length], torch.zeros(len(test), 10).scatter_(1, torch.LongTensor(
        test[:, length].astype(int)).unsqueeze(1), 1)

    return train_data, train_label, test_data, test_label


# DWT ��������
def apply_dwt(data):

    return features


# MWKCNN-BiGRU-Attention ��������
class MWKCNNBiGRUAttention(nn.Module):
    def __init__(self, num_classes, num_heads=4):

    def forward(self, x):

        return x


# SGAN������
class SGAN_Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):

    def forward(self, z):

        return output


# SGAN������
class SGAN_Discriminator(nn.Module):
    def __init__(self, input_dim):


    def forward(self, x):
        return self.model(x)


# DDS������
class DDS_Sampler:
    def __init__(self, generator, device):

    def sample(self, num_samples):

        return generated_samples


# DDPG Actor����
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):


    def forward(self, state):
        return self.model(state)


# DDPG Critic����
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):


    def forward(self, state, action):

        return self.model(x)


# ��������������
class ReplayBuffer:
    def __init__(self, capacity):

    def push(self, state, action, reward, next_state, done):


    def sample(self, batch_size):

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# OU����������
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):


    def reset(self):

    def noise(self):

        return self.state


# DDPG��������
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0002)
        self.replay_buffer = ReplayBuffer(10000)
        self.ou_noise = OUNoise(action_dim)
        self.gamma = 0.99
        self.tau = 0.001

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # ����Critic
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        expected_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, expected_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ����Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ��������������
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)


# ��������������
def train_and_evaluate_mwkcnn_bigru(batch_size, initial_learning_rate, loss_threshold, num_epochs=50):
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_data, train_label, test_data, test_label = load_cwru_data()

    # DWT��������
    train_data = np.array([apply_dwt(sample) for sample in train_data])
    test_data = np.array([apply_dwt(sample) for sample in test_data])

    # ����������
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    train_data, test_data = map(lambda x: torch.FloatTensor(x).to(device), [train_data, test_data])
    train_label, test_label = map(lambda x: x.to(device), [train_label, test_label])

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MWKCNNBiGRUAttention(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # SGAN
    noise_dim = 100
    output_dim = train_data.shape[1]
    generator = SGAN_Generator(noise_dim, output_dim).to(device)
    discriminator = SGAN_Discriminator(output_dim).to(device)
    mse_loss = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

    # DDS
    dds_sampler = DDS_Sampler(generator, device)

    # DDPG
    state_dim = 5
    action_dim = 1
    ddpg = DDPG(state_dim, action_dim)

    early_stopping = False
    best_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}] Train Loss: {epoch_loss:.4f}")
        scheduler.step()

        # SGAN����
        generator.train()
        discriminator.train()
        gan_batch_size = batch_size
        indices = torch.randperm(train_data.size(0))[:gan_batch_size]
        real_samples = train_data[indices]
        noise = torch.randn(gan_batch_size, noise_dim).to(device)
        fake_samples = generator(noise)

        optimizer_D.zero_grad()
        real_preds = discriminator(real_samples)
        fake_preds = discriminator(fake_samples.detach())
        loss_D_real = mse_loss(real_preds, torch.full_like(real_preds, 0.5))
        loss_D_fake = mse_loss(fake_preds, torch.full_like(fake_preds, 0.5))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        fake_preds = discriminator(generator(noise))
        loss_G = mse_loss(fake_preds, torch.full_like(fake_preds, 0.5))
        loss_G.backward()
        optimizer_G.step()
        print(f"                     D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

        # DDS����
        dds_samples = dds_sampler.sample(gan_batch_size)

        # ��������
        augmented_train_data = torch.cat((train_data, fake_samples, dds_samples), dim=0)
        augmented_train_labels = torch.cat((train_label, train_label[indices], train_label[indices]), dim=0)
        augmented_train_dataset = TensorDataset(augmented_train_data, augmented_train_labels)
        augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)

        # ��������
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data)
            test_loss = criterion(test_outputs, test_label.argmax(dim=1)).item()
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_predictions == test_label.argmax(dim=1)).float().mean().item()
            f1 = f1_score(test_label.argmax(dim=1).cpu(), test_predictions.cpu(), average='macro', zero_division=0)
        print(
            f"                                          Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {f1:.4f}")

        # DDPG����
        state = np.array([epoch_loss, test_loss, optimizer.param_groups[0]['lr'],
                          np.linalg.norm(optimizer.state_dict()['state'][0]['exp_avg'].cpu().numpy()), test_accuracy])
        action = ddpg.select_action(state) + ddpg.ou_noise.noise()
        action = np.clip(action, -1, 1)
        new_lr = initial_learning_rate * (1 + action[0])
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        reward = -epoch_loss + test_accuracy
        if len(ddpg.replay_buffer.buffer) > 0:
            avg_reward = np.mean([exp[2] for exp in ddpg.replay_buffer.buffer])
        else:
            avg_reward = 0
        centered_reward = reward - avg_reward
        next_state = np.array(
            [epoch_loss, test_loss, new_lr, np.linalg.norm(optimizer.state_dict()['state'][0]['exp_avg'].cpu().numpy() ),
             test_accuracy])
        ddpg.replay_buffer.push(state, action, centered_reward, next_state, False)
        ddpg.update(batch_size=64)

        # ��������
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                early_stopping = True
                break

        if epoch_loss < loss_threshold:
            print("Loss threshold reached.")
            break

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        test_predictions = torch.argmax(test_outputs, dim=1)
        final_accuracy = (test_predictions == test_label.argmax(dim=1)).float().mean().item()
        print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # ��������������
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("deep", n_colors=10)
    scatter = sns.scatterplot(x=test_data.cpu().numpy()[:, 0], y=test_data.cpu().numpy()[:, 1],
                              hue=test_label.argmax(dim=1).cpu(), palette=palette)
    plt.title("Original Feature Distribution", fontsize=16)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
                          markerfacecolor=palette[i]) for i in range(10)]
    plt.legend(handles=handles, title="Classes", fontsize=12)
    plt.savefig("/root/PycharmProjects/CWRU/confusion_matrix.png")
    plt.close()

    # t-SNE������
    feature_representation = test_outputs.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=3000, init='pca',
                early_exaggeration=20, metric='cosine')
    tsne_results = tsne.fit_transform(feature_representation)
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1],
                              hue=test_predictions.cpu(), palette=palette)
    plt.title("MWKCNN-BiGRU Features", fontsize=16)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
                          markerfacecolor=palette[i]) for i in range(10)]
    plt.legend(handles=handles, title="Classes", fontsize=12)
    plt.savefig("/root/PycharmProjects/CWRU/Features.png")
    plt.close()

    # ��������
    cm = confusion_matrix(test_label.argmax(dim=1).cpu(), test_predictions.cpu())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.title("Confusion Matrix", fontsize=16)
    plt.savefig("/root/PycharmProjects/CWRU/Features2.png")
    plt.close()


train_and_evaluate_mwkcnn_bigru()
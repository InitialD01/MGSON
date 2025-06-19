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
import scipy.io
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pywt

# Set random seed for reproducibility
def set_random_seed(seed=):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load UConn data
def load_uconn_data(data_per_class, split_rate):
    file_path_timedomain = r'/root/Desktop/Mine/DataForClassification_TimeDomain.mat'
    time_domain_data = scipy.io.loadmat(file_path_timedomain)
    if 'AccTimeDomain' not in time_domain_data:
        raise KeyError("'AccTimeDomain' not found in the dataset")
    time_domain_matrix = time_domain_data['AccTimeDomain']
    df_time_domain = pd.DataFrame(time_domain_matrix)
    labels = np.repeat(np.arange(9), 104)
    X = df_time_domain.T
    y = pd.Series(labels)
    random.seed()
    shuffled_indices = np.random.permutation(len(X))
    X = X.iloc[shuffled_indices].reset_index(drop=True)
    y = y.iloc[shuffled_indices].reset_index(drop=True)

    train_data, test_data = [], []
    for label in range(9):
        class_indices = y[y == label].index[:data_per_class]
        X_class = X.loc[class_indices]
        y_class = y.loc[class_indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X_class, y_class, test_size=split_rate[1], random_state=42, stratify=y_class
        )
        train_data.append((X_train, y_train))
        test_data.append((X_test, y_test))
    X_train_final = pd.concat([data[0] for data in train_data], ignore_index=True)
    y_train_final = pd.concat([data[1] for data in train_data], ignore_index=True)
    X_test_final = pd.concat([data[0] for data in test_data], ignore_index=True)
    y_test_final = pd.concat([data[1] for data in test_data], ignore_index=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    # Apply DWT for feature extraction
    def apply_dwt(data):

        return features


    return train_data_tensor, train_label_tensor, test_data_tensor, test_label_tensor, X_train_final, X_test_final, y_train_final, y_test_final

# Define MWKCNN-BiGRU-Attention model
class MWKCNNBiGRUAttention(nn.Module):
    def __init__(self, num_classes, num_heads=4):


    def forward(self, x):

        return x

# Define SGAN Generator
class SGAN_Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):


    def forward(self, z):

        return output

# Define SGAN Discriminator
class SGAN_Discriminator(nn.Module):
    def __init__(self, input_dim):

    def forward(self, x):
        return self.model(x)

# Define DDS Sampler
class DDS_Sampler:
    def __init__(self, generator, device):

    def sample(self, num_samples):

# Define DDPG Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):


    def forward(self, state):
        return self.model(state)

# Define DDPG Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):

    def forward(self, state, action):

        return self.model(x)

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):


    def push(self, state, action, reward, next_state, done):

    def sample(self, batch_size):

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Define OU Noise
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Define DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
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

        # Update Critic
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        expected_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, expected_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

# Train and evaluate MWKCNN-BiGRU with SGAN, DDS, and DDPG
def train_and_evaluate_mwkcnn_bigru(batch_size, initial_learning_rate, loss_threshold, num_epochs=50):
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_data, train_label, test_data, test_label, X_train_final, X_test_final, y_train_final, y_test_final = load_uconn_data()
    train_data, test_data = map(lambda x: x.to(device), [train_data, test_data])
    train_label, test_label = map(lambda x: x.to(device), [train_label, test_label])
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MWKCNNBiGRUAttention(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)


    noise_dim = 100
    output_dim = train_data.shape[1]
    generator = SGAN_Generator(noise_dim, output_dim).to(device)
    discriminator = SGAN_Discriminator(output_dim).to(device)
    mse_loss = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    # Initialize DDS
    dds_sampler = DDS_Sampler(generator, device)

    # Initialize DDPG
    state_dim = 5
    action_dim = 1
    ddpg = DDPG(state_dim, action_dim)

    early_stopping = False
    best_loss = float('inf')
    patience = 10
    counter = 0

    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops = 0
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            in_channels, out_channels, kernel_size = module.in_channels, module.out_channels, module.kernel_size[0]
            total_flops += in_channels * out_channels * kernel_size * (
                module.input_size[1] if hasattr(module, 'input_size') else 1)
        elif isinstance(module, nn.Linear):
            in_features, out_features = module.in_features, module.out_features
            total_flops += in_features * out_features
    total_flops /= 1e9

    f1_scores = []

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

        # Train SGAN
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

        # Apply DDS
        dds_samples = dds_sampler.sample(gan_batch_size)

        # Create augmented dataset
        augmented_train_data = torch.cat((train_data, fake_samples, dds_samples), dim=0)
        augmented_train_labels = torch.cat((train_label, train_label[indices], train_label[indices]), dim=0)
        augmented_train_dataset = TensorDataset(augmented_train_data, augmented_train_labels)
        augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data)
            test_loss = criterion(test_outputs, test_label.argmax(dim=1)).item()
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_predictions == test_label.argmax(dim=1)).float().mean().item()
            f1 = f1_score(test_label.argmax(dim=1).cpu(), test_predictions.cpu(), average='macro', zero_division=0)
            f1_scores.append(f1)
        print(f"                                          Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {f1:.4f}")

        # Update DDPG
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
            [epoch_loss, test_loss, new_lr, np.linalg.norm(optimizer.state_dict()['state'][0]['exp_avg'].cpu().numpy()),
             test_accuracy])
        ddpg.replay_buffer.push(state, action, centered_reward, next_state, False)
        ddpg.update(batch_size=64)

        # Early stopping
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
            print("Early stopping due to low loss.")
            break

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        test_predictions = torch.argmax(test_outputs, dim=1)
        final_accuracy = (test_predictions == test_label.argmax(dim=1)).float().mean().item()
        print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Feature visualization with improved aesthetics
    feature_representation = test_outputs.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=3000, init='pca',
                early_exaggeration=20, metric='cosine')
    tsne_results = tsne.fit_transform(feature_representation)

    # High-resolution visualization with better color palette
    plt.figure(figsize=(8, 8), dpi=900)
    palette = sns.color_palette("husl", n_colors=9)
    scatter = sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1],
                              hue=test_predictions.cpu(), palette=palette, alpha=0.8, s=100, linewidth=0.5,
                              edgecolor='w')

    # ????????????????????????
    plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    # ????????????????????????
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(0.95, 0.7), fontsize=24, markerscale=2)

    plt.title("", fontsize=20, fontweight='bold')
    plt.xlabel("Dimension 1", fontsize=24)
    plt.ylabel("Dimension 2", fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("/root/PycharmProjects/UConn/Features.png")
    plt.close()

    # Confusion matrix visualization
    cm = confusion_matrix(test_label.argmax(dim=1).cpu(), test_predictions.cpu())
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=np.arange(9), yticklabels=np.arange(9),
                annot_kws={"size": 14})
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.title("Confusion Matrix", fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig("/root/PycharmProjects/UConn/Features2.png")
    plt.close()


train_and_evaluate_mwkcnn_bigru()
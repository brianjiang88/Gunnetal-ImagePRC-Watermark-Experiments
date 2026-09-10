import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

# create an enum for covariance types
class CovarianceType(Enum):
    ISOTROPIC = 1
    HETEROSCEDASTIC = 2
    PSD = 3


class NoisePredictor(nn.Module):
    def __init__(
        self,
        dimension,
        num_diffusion_steps,
        time_embedding_dim=32,
        hidden_dim=128,
    ):
        super().__init__()

        # Includes timestep indices 0, ..., T
        self.time_embedding = nn.Embedding(
            num_diffusion_steps + 1,
            time_embedding_dim,
        )

        self.network = nn.Sequential(
            nn.Linear(dimension + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dimension),
        )

    def forward(self, x_t, t):
        time_embedding = self.time_embedding(t)
        model_input = torch.cat([x_t, time_embedding], dim=-1)
        return self.network(model_input)

class GMMDiffusion:
    def __init__(self, dimension=1, num_components=1, num_diffusion_steps=1000, cov_type=CovarianceType.ISOTROPIC):
        """
        Initializes a Gaussian Mixture Model (GMM) with specified parameters.
        """
        self.num_components = num_components
        self.dimension = dimension
        self.dimension = dimension
        self.num_components = num_components
        self.num_diffusion_steps = num_diffusion_steps

        # Shared diffusion schedule
        self.betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)

        self.alphas = 1.0 - self.betas

        # Index 0 represents clean data, so alpha_bar[0] = 1.
        self.alpha_bars = torch.cat([
            torch.ones(1),
            torch.cumprod(self.alphas, dim=0),
        ])
        # create random means, covariances, and weights for the GMM components
        random_weights = np.random.rand(num_components)
        self.weights = random_weights / np.sum(random_weights)
        self.means = 10 * np.random.randn(num_components, dimension)
        if cov_type == CovarianceType.ISOTROPIC:
            sigma = np.random.rand() + 0.5  # ensure positive variance
            self.covariances = np.array([sigma ** 2 * np.eye(dimension) for _ in range(num_components)])
        elif cov_type == CovarianceType.HETEROSCEDASTIC:
            self.covariances = np.array([np.diag(np.random.rand(dimension)) for _ in range(num_components)])
        elif cov_type == CovarianceType.PSD:
            self.grams = np.array([np.random.rand(dimension, dimension) for _ in range(num_components)]) # construct gram matrices
            self.covariances = np.array([gram @ gram.T for gram in self.grams]) # convert to covariance matrices

    # training time: generate samples from the GMM to train the diffusion model (forward process)
    def sample_gmm(self, num_samples=1000):
        """
        Generates samples from the Gaussian Mixture Model (GMM).
        """
        samples = []
        for _ in range(num_samples):
            # choose a component based on the weights
            component = np.random.choice(self.num_components, p=self.weights)
            mean = self.means[component]
            covariance = self.covariances[component]
            # sample from the chosen Gaussian component
            sample = np.random.multivariate_normal(mean, covariance)
            samples.append(sample)
        return np.array(samples)

    # training time: run forward diffusion process on the GMM samples to generate noisy samples
    def forward_diffusion(self, samples, timesteps=None):
        samples = torch.as_tensor(samples, dtype=torch.float32)

        batch_size = samples.shape[0]

        if timesteps is None:
            timesteps = torch.randint(1, self.num_diffusion_steps + 1, (batch_size,))

        epsilon = torch.randn_like(samples)
        alpha_bar_t = self.alpha_bars[timesteps].unsqueeze(1)

        noisy_samples = torch.sqrt(alpha_bar_t) * samples + torch.sqrt(1 - alpha_bar_t) * epsilon
        return noisy_samples, epsilon, timesteps

    def train_reverse_diffusion(
        self,
        clean_samples,
        num_epochs=1000,
        learning_rate=1e-3,
        batch_size=128,
        device="cpu",
    ):
        clean_samples = torch.as_tensor(
            clean_samples,
            dtype=torch.float32,
            device=device,
        )

        model = NoisePredictor(
            dimension=self.dimension,
            num_diffusion_steps=self.num_diffusion_steps,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        num_samples = clean_samples.shape[0]

        for epoch in range(num_epochs):
            model.train()

            permutation = torch.randperm(
                num_samples,
                device=device,
            )

            total_loss = 0.0

            for start in range(0, num_samples, batch_size):
                indices = permutation[start : start + batch_size]
                x_0 = clean_samples[indices]
                current_batch_size = x_0.shape[0]

                # Sample a separate timestep for every data point.
                t = torch.randint(
                    low=1,
                    high=self.num_diffusion_steps + 1,
                    size=(current_batch_size,),
                    device=device,
                )

                # Sample independent Gaussian noise.
                epsilon = torch.randn_like(x_0)

                # Shape: [batch_size, 1], for broadcasting across dimensions.
                alpha_bar_t = self.alpha_bars[t].to(device).unsqueeze(1)

                # Closed-form forward diffusion.
                x_t = (
                    torch.sqrt(alpha_bar_t) * x_0
                    + torch.sqrt(1 - alpha_bar_t) * epsilon
                )

                # Predict the noise that was used to construct x_t.
                epsilon_pred = model(x_t, t)

                loss = F.mse_loss(epsilon_pred, epsilon)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * current_batch_size

            if epoch % 100 == 0:
                average_loss = total_loss / num_samples

                print(
                    f"Epoch [{epoch}/{num_epochs}], "
                    f"Loss: {average_loss:.4f}"
                )

        return model

    @torch.no_grad()
    def recover(
        self,
        x_0,
        model,
        num_inference_steps=100,
        device="cpu",
    ):
        """
        Approximately recover the initial noise x_T from a generated sample x_0
        using deterministic DDIM inversion.
        """
        model.eval()

        x = torch.as_tensor(
            x_0,
            dtype=torch.float32,
            device=device,
        )

        if x.ndim == 1:
            x = x.unsqueeze(0)

        # alpha_bars should have indices 0,...,T, with alpha_bars[0] = 1.
        total_steps = len(self.alpha_bars) - 1

        timesteps = torch.linspace(
            0,
            total_steps,
            num_inference_steps + 1,
            device=device,
        ).round().long()

        for current_t, next_t in zip(timesteps[:-1], timesteps[1:]):
            alpha_bar_t = self.alpha_bars[current_t].to(device)
            alpha_bar_next = self.alpha_bars[next_t].to(device)

            t_batch = torch.full(
                (x.shape[0],),
                int(current_t.item()),
                dtype=torch.long,
                device=device,
            )

            # Model predicts the noise at the current location and time.
            epsilon_pred = model(x, t_batch)

            # Estimate the clean sample corresponding to the current point.
            predicted_x_0 = (
                x
                - torch.sqrt(1 - alpha_bar_t) * epsilon_pred
            ) / torch.sqrt(alpha_bar_t)

            # Move toward the next, noisier timestep.
            x = (
                torch.sqrt(alpha_bar_next) * predicted_x_0
                + torch.sqrt(1 - alpha_bar_next) * epsilon_pred
            )

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        initial_noise,
        model,
        num_inference_steps=100,
        device=None,
    ):
        """
        Generate samples using deterministic DDIM sampling.

        Parameters
        ----------
        initial_noise:
            Tensor or array with shape [batch_size, dimension], sampled from N(0, I).

        model:
            Trained noise-prediction model epsilon_theta(x_t, t).

        num_inference_steps:
            Number of DDIM updates. Must not exceed the number of training steps.

        device:
            PyTorch device. If omitted, use the model's device.

        Returns
        -------
        Tensor with shape [batch_size, dimension].
        """
        model.eval()

        if device is None:
            device = next(model.parameters()).device

        x_t = torch.as_tensor(
            initial_noise,
            dtype=torch.float32,
            device=device,
        ).clone()

        # Convert a single sample [dimension] to [1, dimension].
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)

        if x_t.ndim != 2:
            raise ValueError(
                "initial_noise must have shape "
                "[dimension] or [batch_size, dimension]"
            )

        if x_t.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"but received {x_t.shape[1]}"
            )

        if num_inference_steps > self.num_diffusion_steps:
            raise ValueError(
                "num_inference_steps cannot exceed "
                "num_diffusion_steps"
            )

        # Increasing grid: [0, ..., T].
        # We iterate over it in reverse.
        timesteps = torch.linspace(
            0,
            self.num_diffusion_steps,
            num_inference_steps + 1,
        ).round().long()

        for index in range(len(timesteps) - 1, 0, -1):
            current_t = int(timesteps[index].item())
            previous_t = int(timesteps[index - 1].item())

            alpha_bar_t = self.alpha_bars[current_t].to(device)
            alpha_bar_previous = self.alpha_bars[previous_t].to(device)

            # Every sample in the batch is currently at the same timestep.
            t_batch = torch.full(
                size=(x_t.shape[0],),
                fill_value=current_t,
                dtype=torch.long,
                device=device,
            )

            # Predict the noise contained in x_t.
            epsilon_pred = model(x_t, t_batch)

            # Estimate the underlying clean sample.
            predicted_x_0 = (
                x_t
                - torch.sqrt(1.0 - alpha_bar_t) * epsilon_pred
            ) / torch.sqrt(alpha_bar_t)

            # Deterministic DDIM update from current_t to previous_t.
            x_t = (
                torch.sqrt(alpha_bar_previous) * predicted_x_0
                + torch.sqrt(1.0 - alpha_bar_previous) * epsilon_pred
            )

        # At the end of the loop, x_t is at timestep 0.
        return x_t

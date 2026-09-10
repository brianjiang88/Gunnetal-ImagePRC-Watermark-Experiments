import gmm_diffusion
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

def experiment(dimensions=10, 
               num_components=3, 
               num_diffusion_steps=1000, 
               num_inference_steps=100, 
               num_training_samples=10000, 
               num_trials=1000, 
               num_experiment_repetitions=10,
               cov_type=gmm_diffusion.CovarianceType.ISOTROPIC, 
               path="results"):

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    # dimension = 10
    # num_components = 3
    # num_diffusion_steps = 1000
    # num_inference_steps = 100
    # num_training_samples = 10000
    # num_trials = 1000
    # num_experiment_repetitions = 10

    if not os.path.exists(path):
        os.makedirs(path)

    diffusion_model_statistics = []
    for i in range(num_experiment_repetitions):

        # ---------------------------------------------------------
        # Step 1: Construct the target GMM and diffusion schedule
        # ---------------------------------------------------------

        diffusion = gmm_diffusion.GMMDiffusion(
            dimension=dimensions,
            num_components=num_components,
            num_diffusion_steps=num_diffusion_steps,
            cov_type=cov_type,
        )


        # ---------------------------------------------------------
        # Step 2: Sample clean training data from the target GMM
        # ---------------------------------------------------------

        gmm_samples = diffusion.sample_gmm(
            num_samples=num_training_samples
        )


        # ---------------------------------------------------------
        # Step 3: Train epsilon_theta(x_t, t)
        # ---------------------------------------------------------

        model = diffusion.train_reverse_diffusion(
            clean_samples=gmm_samples,
            num_epochs=1000,
            learning_rate=1e-5,
        )


        # ---------------------------------------------------------
        # Step 4: Sample initial Gaussian noise
        # ---------------------------------------------------------

        original_noise = torch.randn(
            num_trials,
            dimensions,
        )


        # ---------------------------------------------------------
        # Step 5: Generate GMM samples with deterministic DDIM
        # ---------------------------------------------------------

        with torch.no_grad():
            generated_samples = diffusion.ddim_sample(
                initial_noise=original_noise,
                model=model,
                num_inference_steps=num_inference_steps,
            )


        # ---------------------------------------------------------
        # Step 6: Recover the initial noise with DDIM inversion
        # ---------------------------------------------------------

        with torch.no_grad():
            recovered_noise = diffusion.recover(
                x_0=generated_samples,
                model=model,
                num_inference_steps=num_inference_steps,
            )


        # Convert to NumPy for analysis
        original_noise = original_noise.cpu().numpy()
        generated_samples = generated_samples.cpu().numpy()
        recovered_noise = recovered_noise.cpu().numpy()

        recovery_errors = recovered_noise - original_noise

        recovery_error_covariance = np.cov(recovery_errors, rowvar=False)

        print("Mean recovery error:")
        print(recovery_errors.mean(axis=0))

        print("\nRecovery-error covariance:")
        print(recovery_error_covariance)

        # print the eigenvalues of the recovery-error covariance matrix, operator norm, condition number
        eigenvalues = np.linalg.eigvalsh(recovery_error_covariance)
        condition_number = eigenvalues.max() / eigenvalues.min()
        operator_norm = np.linalg.norm(recovery_error_covariance, ord=2)

        print("\nRecovery-error covariance matrix:")
        print(recovery_error_covariance)

        print("\nEigenvalues:", eigenvalues)
        print("Condition number:", condition_number)
        print("Operator norm:", operator_norm)

        # heatmap of recovery-error covariance
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(
            recovery_error_covariance,
            cmap="viridis",
        )

        fig.colorbar(image, ax=ax)

        ax.set_title("Recovery-error covariance")
        ax.set_xlabel("Coordinate")
        ax.set_ylabel("Coordinate")

        fig.tight_layout()
        fig.savefig(
            os.path.join(path, f"recovery_error_covariance_{i}.png"),
            dpi=150,
        )
        plt.close(fig)

        print("\nMean squared recovery error:")
        print(np.mean(recovery_errors**2))

        sigma_squared_hat = np.mean(recovery_errors**2)
        sigma_hat = np.sqrt(sigma_squared_hat)

        print("\nEstimated sigma:", sigma_hat)

        fig, axes = plt.subplots(
            1,
            dimensions,
            figsize=(5 * dimensions, 4),
        )

        if dimensions == 1:
            axes = [axes]

        for coordinate in range(dimensions):
            errors = recovery_errors[:, coordinate]

            axes[coordinate].hist(
                errors,
                bins=50,
                density=True,
                alpha=0.7,
                label="Empirical error",
            )

            mean = errors.mean()
            std = errors.std(ddof=1)

            grid = np.linspace(
                errors.min(),
                errors.max(),
                300,
            )

            axes[coordinate].plot(
                grid,
                stats.norm.pdf(grid, mean, std),
                color="red",
                label="Fitted Gaussian",
            )

            axes[coordinate].set_title(
                f"Coordinate {coordinate}"
            )
            axes[coordinate].set_xlabel(r"$z'_i-z_i$")
            axes[coordinate].legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(path, f"recovery_errors_{i}.png"),
            dpi=150,
        )

        plt.close(fig)

        print("Target GMM means:")
        print(diffusion.means)

        print("\nTarget GMM covariances:")
        print(diffusion.covariances)

        print("\nTarget GMM weights:")
        print(diffusion.weights)

        diffusion_model_statistics.append({
            # "means": diffusion.means,
            # "covariances": diffusion.covariances,
            # "weights": diffusion.weights,
            # "recovery_error_mean": recovery_errors.mean(axis=0),
            # "recovery_error_covariance": recovery_error_covariance,
            # "eigenvalues": eigenvalues,
            "condition_number": condition_number,
            "operator_norm": operator_norm,
            "mean_squared_recovery_error": np.mean(recovery_errors**2),
            # "estimated_sigma": sigma_hat,
        })



        # plt.scatter(
        #     generated_samples[:, 0],
        #     generated_samples[:, 1],
        #     s=10,
        #     alpha=0.5,
        # )

        # plt.title("Samples generated by DDIM")
        # plt.xlabel("Coordinate 0")
        # plt.ylabel("Coordinate 1")
        # plt.savefig(f"graphs/generated_samples_{i}.png")

    # save diffusion_model_statistics to a CSV file
    # create pandas dataframe from diffusion_model_statistics
    diffusion_model_statistics_df = pd.DataFrame(diffusion_model_statistics)
    diffusion_model_statistics_df.to_csv(os.path.join(path, "diffusion_model_statistics.csv"), index=False)

if __name__ == "__main__":
    # allow user to specify all these parameters from the command line, with default values
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--num_components", type=int, default=3)
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--num_training_samples", type=int, default=10000)
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--num_experiment_repetitions", type=int, default=10)
    parser.add_argument("--cov_type", type=str, default="ISOTROPIC", choices=["ISOTROPIC", "HETEROSCEDASTIC", "PSD"])  
    parser.add_argument("--path", type=str, default="results")
    args = parser.parse_args()
    experiment(
        dimensions=args.dimensions,
        num_components=args.num_components,
        num_diffusion_steps=args.num_diffusion_steps,
        num_training_samples=args.num_training_samples,
        num_trials=args.num_trials,
        num_experiment_repetitions=args.num_experiment_repetitions,
        cov_type=gmm_diffusion.CovarianceType[args.cov_type],
        path=args.path
    )
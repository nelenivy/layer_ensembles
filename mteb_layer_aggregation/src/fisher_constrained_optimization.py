import os

import hashlib
import numpy as np
import scipy.optimize
from scipy.linalg import null_space
import warnings
import logging
import mteb
from typing import Dict, List, Tuple, Optional
from src.aggregated_encoder import AggregatedEncoder


# ============================================================================
# EFFICIENT CACHED WRAPPER FOR OBJECTIVE/GRADIENT/HESSIAN
# ============================================================================

class CachedFisherObjective:
    """
    Efficient wrapper that caches Fisher gradient/Hessian computations.

    Problem: scipy calls objective(), jac(), and hess() separately, but all
    three need the same expensive Fisher computation at point w.

    Solution: Cache the last evaluation. If called again at same point,
    return cached results instead of recomputing.

    This reduces 3× redundant computations to 1×.
    """

    def __init__(
        self,
        model_name: str,
        dataset_resolver,
        n_layers: int,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        hessian_cache_dir: str = "./hessiancache",
        use_emb_cache: bool = True,
        emb_cache_dir: str = "./embs_cache",
        perturbationsize: float = 0.1,
        num_samples: int = 1000,
        verbose: int = 1
    ):
        self.model_name = model_name
        self.dataset_resolver = dataset_resolver
        self.n_layers = n_layers
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = device
        self.hessian_cache_dir = hessian_cache_dir
        self.use_emb_cache = use_emb_cache
        self.emb_cache_dir = emb_cache_dir
        self.perturbationsize = perturbationsize
        self.num_samples = num_samples
        self.verbose = verbose

        # Cache for last evaluation
        self._cache_w = None
        self._cache_quality = None
        self._cache_gradient = None
        self._cache_hessian = None

        # History tracking
        self.history_weights = []
        self.history_quality = []
        self.n_eval = 0

    def _compute_if_needed(self, w: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Fisher gradient/Hessian only if point changed.

        Args:
            w: Weight vector

        Returns:
            quality, gradient, hessian
        """
        # Check if this is the same point as last time
        if self._cache_w is not None and np.allclose(w, self._cache_w, atol=1e-12):
            if self.verbose >= 2:
                print(f"  [Cache hit! Reusing computation]")
            return self._cache_quality, self._cache_gradient, self._cache_hessian

        # New point - need to compute
        if self.verbose >= 2:
            print(f"  [Cache miss - computing Fisher at new point]")

        quality, gradient, hessian = self._compute_fisher_gradient_hessian(w)

        # Update cache
        self._cache_w = w.copy()
        self._cache_quality = quality
        self._cache_gradient = gradient
        self._cache_hessian = hessian

        # Update history (only once per unique point)
        self.n_eval += 1
        self.history_weights.append(w.copy())
        self.history_quality.append(quality)

        if self.verbose >= 1:
            print(f"\nEvaluation {self.n_eval}: quality = {quality:.6f}")

        return quality, gradient, hessian

    def _compute_fisher_gradient_hessian(self, weights: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute gradient and Hessian using Fisher Information Matrix.

        This is the expensive operation we want to avoid repeating.
        """
        predictionbasedir = os.path.join(self.hessian_cache_dir, "fisher_predictions_trustregion")
        os.makedirs(predictionbasedir, exist_ok=True)

        # Evaluate at center
        encoder_center = AggregatedEncoder(
            model_name=self.model_name,
            pooling=self.pooling,
            batch_size=self.batch_size,
            device=self.device,
            aggregation_weights=weights,
            use_cache=self.use_emb_cache,
            cache_dir=self.emb_cache_dir
        )

        weights_hash = hashlib.md5(weights.tobytes()).hexdigest()[:8]
        predictiondir_center = os.path.join(predictionbasedir, f"center_{weights_hash}")
        os.makedirs(predictiondir_center, exist_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results_center = mteb.evaluate(
                model=encoder_center,
                tasks=[self.dataset_resolver.task],
                encode_kwargs={"batch_size": self.batch_size},
                prediction_folder=predictiondir_center,
                show_progress_bar=False,
                overwrite_strategy="always"
            )

        quality = results_center[0].scores[self.dataset_resolver.val_name][0].get("main_score", 0.0)

        from src.one_layer_eval import load_per_sample_scores_from_predictions
        persample_scores_center = load_per_sample_scores_from_predictions(
            predictiondir_center, self.dataset_resolver
        )

        # Subsample if needed
        indices = None
        if len(persample_scores_center) > self.num_samples:
            np.random.seed(42)
            indices = np.random.choice(len(persample_scores_center), self.num_samples, replace=False)
            persample_scores_center = persample_scores_center[indices]

        persample_log_L_center = np.log(np.clip(persample_scores_center, 1e-10, 1.0))

        # Perturb toward each layer
        persample_gradients = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loggers = ["mteb", "datasets", "transformers", "httpx", "urllib3", "filelock", "huggingface_hub"]
            orig_levels = {name: logging.getLogger(name).level for name in loggers}
            for name in loggers:
                logging.getLogger(name).setLevel(logging.ERROR)

            for i in range(self.n_layers):
                target = np.zeros(self.n_layers)
                target[i] = 1.0
                perturbed = self.perturbationsize * target + weights #(1 - self.perturbationsize) * weights 

                encoder_i = AggregatedEncoder(
                    model_name=self.model_name,
                    pooling=self.pooling,
                    batch_size=self.batch_size,
                    device=self.device,
                    aggregation_weights=perturbed,
                    use_cache=self.use_emb_cache,
                    cache_dir=self.emb_cache_dir
                )

                predictiondir_i = os.path.join(predictionbasedir, f"layer{i}_{weights_hash}")
                os.makedirs(predictiondir_i, exist_ok=True)

                results_i = mteb.evaluate(
                    model=encoder_i,
                    tasks=[self.dataset_resolver.task],
                    encode_kwargs={"batch_size": self.batch_size},
                    prediction_folder=predictiondir_i,
                    show_progress_bar=False,
                    overwrite_strategy="always"
                )

                persample_scores_i = load_per_sample_scores_from_predictions(
                    predictiondir_i, self.dataset_resolver
                )

                if indices is not None:
                    persample_scores_i = persample_scores_i[indices]

                persample_log_L_i = np.log(np.clip(persample_scores_i, 1e-10, 1.0))
                gradient_i = (persample_log_L_i - persample_log_L_center) / self.perturbationsize
                persample_gradients.append(gradient_i)

            for name, level in orig_levels.items():
                logging.getLogger(name).setLevel(level)

        persample_gradients = np.array(persample_gradients)
        F = np.cov(persample_gradients)
        print(F)
        gradient = persample_gradients.mean(axis=1)#.mean(axis=1)
        print(gradient)
        gradient_corrrection = np.outer(gradient, gradient)
        print(gradient_corrrection)
        H = -F + gradient
        print(H)
        H = (H + H.T) / 2

        return quality, gradient, H

    def objective(self, w: np.ndarray) -> float:
        """Objective function (for minimization)."""
        quality, _, _ = self._compute_if_needed(w)
        return -quality  # Minimize negative = maximize quality

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Gradient of objective."""
        _, gradient, _ = self._compute_if_needed(w)
        return -gradient  # Gradient of negative quality

    def hessian(self, w: np.ndarray) -> np.ndarray:
        """Hessian of objective."""
        _, _, hessian = self._compute_if_needed(w)
        return -hessian  # Hessian of negative quality


# ============================================================================
# EFFICIENT TRUST-REGION METHOD USING CACHED WRAPPER
# ============================================================================

def fisher_gradient_descent_trust_region_efficient(
    model_name: str,
    dataset_resolver,
    n_layers: int,
    layer_qualities: np.ndarray,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    hessian_cache_dir: str = ".hessiancache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = ".embscache",
    verbose: int = 1,
    perturbationsize: float = 0.1,
    num_samples: int = 1000,
    max_steps: int = 20
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """
    Trust-region constrained optimization with EFFICIENT caching.

    Key improvement: Uses CachedFisherObjective to avoid recomputing
    gradient/Hessian multiple times for the same point.

    scipy.optimize.minimize calls objective(), jac(), and hess() separately
    for the same point. Without caching, this means 3× redundant Fisher
    computations. With caching, we compute only once per unique point.

    Args:
        [Same as before...]

    Returns:
        final_weights, history_quality, history_weights
    """
    # Initial point
    bestlayeridx = np.argmax(layer_qualities)
    w0 = np.zeros(n_layers)
    w0[bestlayeridx] = 1.0

    # Create cached objective wrapper
    cached_obj = CachedFisherObjective(
        model_name=model_name,
        dataset_resolver=dataset_resolver,
        n_layers=n_layers,
        pooling=pooling,
        batch_size=batch_size,
        device=device,
        hessian_cache_dir=hessian_cache_dir,
        use_emb_cache=use_emb_cache,
        emb_cache_dir=emb_cache_dir,
        perturbationsize=perturbationsize,
        num_samples=num_samples,
        verbose=verbose
    )

    # Constraints
    constraints = [
        # Equality: sum(w) = 1
        scipy.optimize.LinearConstraint(
            np.ones((1, n_layers)), 
            lb=1.0, ub=1.0
        ),
        # Inequality: w >= 0
        scipy.optimize.LinearConstraint(
            np.eye(n_layers),
            lb=0, ub=np.inf
        )
    ]

    if verbose >= 1:
        print("="*70)
        print("Fisher Second-Order Optimization: EFFICIENT TRUST-REGION")
        print("="*70)
        print("Using scipy.optimize.minimize with trust-constr")
        print("With CACHED objective/gradient/Hessian (3x speedup!)")
        print(f"Starting at best layer: {bestlayeridx}")
        print("="*70)

    # Run optimization
    result = scipy.optimize.minimize(
        cached_obj.objective,
        w0,
        method='trust-constr',
        jac=cached_obj.gradient,
        hess=cached_obj.hessian,
        constraints=constraints,

        options={
            'initial_tr_radius': 0.5,
            #'max_tr_radius': 1.0,
            'maxiter': max_steps,
            'verbose': 2 if verbose >= 2 else 0
        }
    )

    if verbose >= 1:
        print("\n" + "="*70)
        print("Optimization Complete!")
        print("="*70)
        print(f"Status: {result.message}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Unique evaluations: {cached_obj.n_eval}")
        print(f"Function calls by scipy: {result.nfev}")
        print(f"Cache efficiency: {result.nfev / max(1, cached_obj.n_eval):.1f}x redundancy avoided")
        print(f"Final quality: {-result.fun:.6f}")
        print("="*70)

        print(cached_obj.history_quality, cached_obj.history_weights)

    return result.x, cached_obj.history_quality, cached_obj.history_weights


# ============================================================================
# ALSO UPDATE NULL SPACE AND PROJECTED NEWTON WITH INTERNAL CACHING
# ============================================================================

def fisher_gradient_descent_null_space_efficient(
    model_name: str,
    dataset_resolver,
    n_layers: int,
    layer_qualities: np.ndarray,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    hessian_cache_dir: str = ".hessiancache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = ".embscache",
    verbose: int = 1,
    perturbationsize: float = 0.1,
    num_samples: int = 1000,
    stepsize: float = 0.1,
    max_steps: int = 20,
    epsilon: float = 1e-4,
    regularization: float = 1e-5
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """
    Null space method with efficient Fisher computation.

    Note: This method doesn't have the same redundancy issue as trust-region
    because we control when gradient/Hessian are computed. But we still use
    the cached wrapper for consistency and cleaner code.
    """
    # Initialize
    bestlayeridx = np.argmax(layer_qualities)
    currentweights = np.zeros(n_layers)
    currentweights[bestlayeridx] = 1.0

    history_weights = [currentweights.copy()]
    history_quality = []

    # Null space basis
    A = np.ones((1, n_layers))
    Z = null_space(A)

    # Create cached objective (even though we won't call it redundantly)
    cached_obj = CachedFisherObjective(
        model_name=model_name,
        dataset_resolver=dataset_resolver,
        n_layers=n_layers,
        pooling=pooling,
        batch_size=batch_size,
        device=device,
        hessian_cache_dir=hessian_cache_dir,
        use_emb_cache=use_emb_cache,
        emb_cache_dir=emb_cache_dir,
        perturbationsize=perturbationsize,
        num_samples=num_samples,
        verbose=verbose
    )

    if verbose >= 1:
        print("="*70)
        print("Fisher Second-Order Gradient Descent: NULL SPACE METHOD")
        print("="*70)
        print(f"Working in reduced {n_layers-1}-dimensional space")
        print(f"Starting at best layer: {bestlayeridx}")
        print("="*70)

    def evaluate_at_weights(weights):
        encoder = AggregatedEncoder(
            model_name=model_name, pooling=pooling, batch_size=batch_size,
            device=device, aggregation_weights=weights,
            use_cache=use_emb_cache, cache_dir=emb_cache_dir
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = mteb.evaluate(
                model=encoder, tasks=[dataset_resolver.task],
                encode_kwargs={"batch_size": batch_size},
                show_progress_bar=False, overwrite_strategy="always"
            )
        return results[0].scores[valname][0].get("main_score", 0.0)

    def project_to_simplex_if_needed(weights):
        if np.any(weights < -1e-10):
            weights = np.maximum(weights, 0)
            weights_sum = weights.sum()
            if weights_sum > 1e-10:
                weights = weights / weights_sum
            else:
                weights = np.ones(n_layers) / n_layers
        return weights

    # Main loop
    for step in range(max_steps):
        if verbose >= 1:
            print(f"\nStep {step + 1}/{max_steps}")
            print("-" * 70)

        # Compute using cached wrapper
        quality, gradient_full, hessian_full = cached_obj._compute_if_needed(currentweights)
        history_quality.append(quality)

        # Project to null space
        gradient_reduced = Z.T @ gradient_full
        hessian_reduced = Z.T @ hessian_full @ Z

        grad_norm = np.linalg.norm(gradient_reduced)

        if verbose >= 1:
            print(f"Quality: {quality:.6f}")
            print(f"Reduced gradient norm: {grad_norm:.6f}")

        if grad_norm < epsilon:
            if verbose >= 1:
                print(f"\nCONVERGED: Reduced gradient norm {grad_norm:.6f} < epsilon {epsilon}")
            break

        # Solve in reduced space
        hessian_reduced_reg = hessian_reduced - regularization * np.eye(n_layers - 1)

        try:
            delta_alpha = np.linalg.solve(hessian_reduced_reg, gradient_reduced)
        except np.linalg.LinAlgError:
            if verbose >= 1:
                print("  Warning: Reduced Hessian singular, using pseudo-inverse")
            delta_alpha = np.linalg.pinv(hessian_reduced_reg) @ gradient_reduced

        # Map to full space and update
        delta_weights = Z @ delta_alpha
        new_weights = currentweights + stepsize * delta_weights

        # Safety checks
        weight_sum = new_weights.sum()
        if abs(weight_sum - 1.0) > 1e-6:
            new_weights = new_weights / weight_sum

        if np.any(new_weights < 0):
            if verbose >= 1:
                print(f"  Warning: {(new_weights < 0).sum()} negative weights, projecting")
            new_weights = project_to_simplex_if_needed(new_weights)

        weight_change = np.linalg.norm(new_weights - currentweights)

        if verbose >= 1:
            print(f"Weight change: {weight_change:.6f}")

        if weight_change < 1e-8:
            break

        currentweights = new_weights
        history_weights.append(currentweights.copy())

    # Final evaluation
    final_quality = evaluate_at_weights(currentweights)
    if len(history_quality) == 0 or abs(final_quality - history_quality[-1]) > 1e-6:
        history_quality.append(final_quality)

    if verbose >= 1:
        print("\n" + "="*70)
        print("Optimization Complete!")
        print("="*70)
        print(f"Final quality: {final_quality:.6f}")
        print(f"Fisher evaluations: {cached_obj.n_eval}")
        print("="*70)

    return currentweights, history_quality, history_weights


print("Efficient cached implementations created!")


def fisher_gradient_descent_trust_region(
    model_name: str,
    dataset_resolver,
    n_layers: int,
    layer_qualities: np.ndarray,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    hessian_cache_dir: str = ".hessiancache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = "./embs_cache",
    verbose: int = 1,
    perturbationsize: float = 0.1,
    num_samples: int = 1000,
    max_steps: int = 20
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """
    Second-order optimization using scipy's trust-region constrained optimizer.

    KEY INSIGHT: Use professional implementation (trust-constr) that handles
    constraints properly with sequential quadratic programming.

    This wraps our Fisher Information calculation in scipy's optimizer,
    which handles:
    - Equality constraints (sum = 1)
    - Inequality constraints (w_i >= 0)
    - Trust region for stability
    - Automatic convergence detection

    Most robust option but requires finite evaluations for Hessian callback.

    Args:
        [Same as before, but no stepsize/epsilon - scipy handles this]

    Returns:
        final_weights, history_quality, history_weights
    """
    # Track history during optimization
    history_weights = []
    history_quality = []
    n_eval = [0]  # Mutable counter

    # Cache for gradient/hessian
    cache = {}

    def evaluate_at_weights(weights):
        encoder = AggregatedEncoder(
            model_name=model_name, pooling=pooling, batch_size=batch_size,
            device=device, aggregation_weights=weights,
            use_cache=use_emb_cache, cache_dir=emb_cache_dir
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = mteb.evaluate(
                model=encoder, tasks=[dataset_resolver.task],
                encode_kwargs={"batch_size": batch_size},
                show_progress_bar=False, overwrite_strategy="always"
            )
        return results[0].scores[valname][0].get("main_score", 0.0)

    def compute_fisher_gradient_hessian(weights):
        # Check cache
        weights_tuple = tuple(weights)
        if weights_tuple in cache:
            return cache[weights_tuple]

        # [Same Fisher computation as before]
        predictionbasedir = os.path.join(hessian_cache_dir, "fisher_predictions_trustregion")
        os.makedirs(predictionbasedir, exist_ok=True)

        encoder_center = AggregatedEncoder(
            model_name=model_name, pooling=pooling, batch_size=batch_size,
            device=device, aggregation_weights=weights,
            use_cache=use_emb_cache, cache_dir=emb_cache_dir
        )

        weights_hash = hashlib.md5(weights.tobytes()).hexdigest()[:8]
        predictiondir_center = os.path.join(predictionbasedir, f"center_{weights_hash}")
        os.makedirs(predictiondir_center, exist_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results_center = mteb.evaluate(
                model=encoder_center, tasks=[dataset_resolver.task],
                encode_kwargs={"batch_size": batch_size},
                prediction_folder=predictiondir_center,
                show_progress_bar=False, overwrite_strategy="always"
            )

        quality = results_center[0].scores[valname][0].get("main_score", 0.0)

        from src.one_layer_eval import load_per_sample_scores_from_predictions
        persample_scores_center = load_per_sample_scores_from_predictions(
            predictiondir_center, dataset_resolver
        )

        indices = None
        if len(persample_scores_center) > num_samples:
            np.random.seed(42)
            indices = np.random.choice(len(persample_scores_center), num_samples, replace=False)
            persample_scores_center = persample_scores_center[indices]

        persample_log_L_center = np.log(np.clip(persample_scores_center, 1e-10, 1.0))
        persample_gradients = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loggers = ["mteb", "datasets", "transformers", "httpx", "urllib3", "filelock", "huggingface_hub"]
            orig_levels = {name: logging.getLogger(name).level for name in loggers}
            for name in loggers:
                logging.getLogger(name).setLevel(logging.ERROR)

            for i in range(n_layers):
                target = np.zeros(n_layers)
                target[i] = 1.0
                perturbed = (1 - perturbationsize) * weights + perturbationsize * target

                encoder_i = AggregatedEncoder(
                    model_name=model_name, pooling=pooling, batch_size=batch_size,
                    device=device, aggregation_weights=perturbed,
                    use_cache=use_emb_cache, cache_dir=emb_cache_dir
                )

                predictiondir_i = os.path.join(predictionbasedir, f"layer{i}_{weights_hash}")
                os.makedirs(predictiondir_i, exist_ok=True)

                results_i = mteb.evaluate(
                    model=encoder_i, tasks=[dataset_resolver.task],
                    encode_kwargs={"batch_size": batch_size},
                    prediction_folder=predictiondir_i,
                    show_progress_bar=False, overwrite_strategy="always"
                )

                persample_scores_i = load_per_sample_scores_from_predictions(
                    predictiondir_i, dataset_resolver
                )

                if indices is not None:
                    persample_scores_i = persample_scores_i[indices]

                persample_log_L_i = np.log(np.clip(persample_scores_i, 1e-10, 1.0))
                gradient_i = (persample_log_L_i - persample_log_L_center) / persample_log_L_center.shape[0]/ perturbationsize
                persample_gradients.append(gradient_i)

            for name, level in orig_levels.items():
                logging.getLogger(name).setLevel(level)

        #print(persample_gradients)
        persample_gradients = np.array(persample_gradients)

        F = np.cov(persample_gradients)
        print(F)
        gradient = persample_gradients.mean(axis=1)
        print(gradient)
        H = -F
        H = (H + H.T) / 2

        result = (quality, gradient, H)
        cache[weights_tuple] = result
        return result

    # Objective function (minimize negative quality)
    def objective(w):
        n_eval[0] += 1
        history_weights.append(w.copy())

        quality, gradient, hessian = compute_fisher_gradient_hessian(w)
        history_quality.append(quality)

        if verbose >= 1:
            print(f"\nEvaluation {n_eval[0]}: quality = {quality:.6f}")

        return -quality  # Minimize negative = maximize quality

    def gradient_func(w):
        _, gradient, _ = compute_fisher_gradient_hessian(w)
        return -gradient  # Gradient of negative quality

    def hessian_func(w):
        _, _, hessian = compute_fisher_gradient_hessian(w)
        return -hessian  # Hessian of negative quality

    # Initial point
    bestlayeridx = np.argmax(layer_qualities)
    w0 = np.zeros(n_layers)
    w0[bestlayeridx] = 1.0

    # Constraints
    constraints = [
        # Equality: sum(w) = 1
        scipy.optimize.LinearConstraint(
            np.ones((1, n_layers)), 
            lb=1.0, ub=1.0
        ),
        # Inequality: w >= 0
        scipy.optimize.LinearConstraint(
            np.eye(n_layers),
            lb=0, ub=np.inf
        )
    ]

    if verbose >= 1:
        print("="*70)
        print("Fisher Second-Order Optimization: TRUST-REGION METHOD")
        print("="*70)
        print("Using scipy.optimize.minimize with trust-constr")
        print(f"Starting at best layer: {bestlayeridx}")
        print("="*70)

    # Run optimization
    result = scipy.optimize.minimize(
        objective,
        w0,
        method='trust-constr',
        jac=gradient_func,
        hess=hessian_func,
        constraints=constraints,
        options={
            'maxiter': max_steps,
            'verbose': 2 if verbose >= 2 else 0
        }
    )

    if verbose >= 1:
        print("\n" + "="*70)
        print("Optimization Complete!")
        print("="*70)
        print(f"Status: {result.message}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final quality: {-result.fun:.6f}")
        print("="*70)
        print(history_weights)
        print(history_quality)
    return result.x, history_quality, history_weights


print("Three constrained optimization approaches created!")

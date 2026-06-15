"""Activation steering using PCA of response activations.

Extract activations from a forward pass on the response, compute PCA,
and steer the model using principal components to improve prediction.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from contextlib import contextmanager


class ActivationSteering:
    """Extract activations and steer model using PCA."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """Initialize activation steering.

        Args:
            model: HuggingFace causal LM
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Determine number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            self.num_layers = len(model.transformer.h)
        else:
            raise ValueError("Could not determine model architecture")

        # Storage for extracted activations
        self.activations = None
        self.hook_handle = None

        # PCA results
        self.principal_components = None  # [num_components, hidden_dim]
        self.singular_values = None
        self.mean_activation = None

    def _get_layer(self, layer_idx: int):
        """Get the layer module at given index."""
        if hasattr(self.model, 'model') and hasattr(self.model.model,
                                                    'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(
                self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError("Could not access model layers")

    def extract_activations(self,
                            context: str,
                            response: str,
                            layer_idx: Optional[int] = None) -> torch.Tensor:
        """Extract activations for response tokens at a specific layer.

        Args:
            context: Context string
            response: Response string
            layer_idx: Layer to extract from (default: middle layer)

        Returns:
            Activations tensor [num_response_tokens, hidden_dim]
        """
        if layer_idx is None:
            layer_idx = self.num_layers // 2

        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response,
                                             add_special_tokens=False)
        full_ids = context_ids + response_ids
        input_ids = torch.tensor([full_ids], device=self.device)

        response_start = len(context_ids)

        # Hook to capture activations
        captured_activations = []

        def capture_hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured_activations.append(hidden.detach().clone())
            return output

        # Register hook
        layer = self._get_layer(layer_idx)
        handle = layer.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        # Extract response positions
        activations = captured_activations[0][
            0, response_start:, :]  # [num_response_tokens, hidden_dim]

        return activations

    def compute_pca(
        self,
        activations: torch.Tensor,
        num_components: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PCA of activations.

        Args:
            activations: [num_tokens, hidden_dim]
            num_components: Number of principal components to compute

        Returns:
            Tuple of (principal_components, singular_values, mean)
            - principal_components: [num_components, hidden_dim]
            - singular_values: [num_components]
            - mean: [hidden_dim]
        """
        # Center the activations
        mean = activations.mean(dim=0)
        centered = activations - mean

        # SVD (activations = U @ diag(S) @ V^T)
        # V columns are principal components
        U, S, Vh = torch.linalg.svd(centered.float(), full_matrices=False)

        # Take top components
        num_components = min(num_components, len(S))
        principal_components = Vh[:
                                  num_components, :]  # [num_components, hidden_dim]
        singular_values = S[:num_components]

        # Store for later use
        self.principal_components = principal_components
        self.singular_values = singular_values
        self.mean_activation = mean

        return principal_components, singular_values, mean

    @contextmanager
    def steering_context(self,
                         layer_idx: int,
                         steering_vector: torch.Tensor,
                         alpha: float = 1.0):
        """Context manager that applies steering during forward passes.

        Args:
            layer_idx: Layer to steer
            steering_vector: Direction to steer [hidden_dim]
            alpha: Steering strength (can be negative)
        """
        # Get model dtype to match steering vector
        model_dtype = next(self.model.parameters()).dtype
        steering_vec = steering_vector.to(device=self.device,
                                          dtype=model_dtype)
        if steering_vec.dim() == 1:
            steering_vec = steering_vec.unsqueeze(0).unsqueeze(
                0)  # [1, 1, hidden_dim]

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # Add steering to all positions
                steered = hidden + alpha * steering_vec
                return (steered, ) + output[1:]
            else:
                return output + alpha * steering_vec

        layer = self._get_layer(layer_idx)
        handle = layer.register_forward_hook(steering_hook)

        try:
            yield
        finally:
            handle.remove()

    def compute_ce_with_steering(
            self,
            context: str,
            response: str,
            layer_idx: int,
            steering_vector: torch.Tensor,
            alpha: float = 1.0) -> Tuple[float, List[float]]:
        """Compute cross-entropy of response with steering applied.

        Args:
            context: Context string
            response: Response string
            layer_idx: Layer to steer
            steering_vector: Direction to steer [hidden_dim]
            alpha: Steering strength

        Returns:
            Tuple of (total_ce, per_token_probs)
        """
        context_ids = self.tokenizer.encode(context, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response,
                                             add_special_tokens=False)
        full_ids = context_ids + response_ids
        input_ids = torch.tensor([full_ids], device=self.device)

        response_start = len(context_ids)

        with self.steering_context(layer_idx, steering_vector, alpha):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

        # Compute CE for response tokens
        total_ce = 0.0
        per_token_probs = []

        for i, target_token in enumerate(response_ids):
            pred_pos = response_start + i - 1
            if pred_pos < 0:
                continue

            probs = F.softmax(logits[0, pred_pos, :].float(), dim=-1)
            prob = probs[target_token].item()
            ce = -torch.log(probs[target_token] + 1e-10).item()

            per_token_probs.append(prob)
            total_ce += ce

        return total_ce, per_token_probs


def run_pca_steering_experiment(model,
                                tokenizer,
                                context: str,
                                response: str,
                                layer_idx: Optional[int] = None,
                                num_pcs: int = 1,
                                alpha_values: List[float] = None,
                                device: str = "cuda",
                                verbose: bool = True) -> Dict:
    """Run PCA steering experiment on a single example.

    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context string
        response: Response string
        layer_idx: Layer to use (default: middle)
        num_pcs: Number of principal components to try
        alpha_values: List of alpha values to try (default: [0.1, 0.5, 1.0, 2.0])
        device: Device
        verbose: Print detailed output for each alpha (default: True)

    Returns:
        Dict with experiment results
    """
    if alpha_values is None:
        alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    steerer = ActivationSteering(model, tokenizer, device)

    if layer_idx is None:
        layer_idx = steerer.num_layers // 2

    if verbose:
        print(f"    Using layer {layer_idx}/{steerer.num_layers}")
        print(f"    Extracting activations...")

    activations = steerer.extract_activations(context, response, layer_idx)

    if verbose:
        print(f"    Activations shape: {activations.shape}")
        print(f"    Computing PCA...")

    pcs, singular_values, mean = steerer.compute_pca(activations,
                                                     num_components=num_pcs)
    variance_explained = (singular_values**2) / (singular_values**2).sum()

    if verbose:
        print(f"    Variance explained by PC1: {variance_explained[0]:.2%}")
        print(f"    Computing baseline CE...")

    baseline_ce, baseline_probs = steerer.compute_ce_with_steering(context,
                                                                   response,
                                                                   layer_idx,
                                                                   pcs[0],
                                                                   alpha=0.0)

    if verbose:
        print(f"    Baseline CE: {baseline_ce:.4f}")

    # Try different alpha values for each PC
    results = {
        "layer_idx": layer_idx,
        "num_layers": steerer.num_layers,
        "activations_shape": list(activations.shape),
        "variance_explained": variance_explained.tolist(),
        "baseline_ce": baseline_ce,
        "baseline_avg_prob": float(torch.tensor(baseline_probs).mean()),
        "steering_results": {}
    }

    for pc_idx in range(num_pcs):
        pc = pcs[pc_idx]
        pc_results = []

        if verbose:
            print(
                f"    Testing PC{pc_idx + 1} (var explained: {variance_explained[pc_idx]:.2%}):"
            )

        for alpha in alpha_values:
            ce, probs = steerer.compute_ce_with_steering(context,
                                                         response,
                                                         layer_idx,
                                                         pc,
                                                         alpha=alpha)
            ce_change = ce - baseline_ce
            ce_pct_change = (ce_change / baseline_ce *
                             100) if baseline_ce > 0 else 0

            if verbose:
                print(
                    f"      alpha={alpha:+.1f}: CE={ce:.4f} ({ce_pct_change:+.2f}%)"
                )

            pc_results.append({
                "alpha": alpha,
                "ce": ce,
                "ce_change": ce_change,
                "ce_pct_change": ce_pct_change,
                "avg_prob": float(torch.tensor(probs).mean())
            })

        # Also try negative alpha
        for alpha in alpha_values:
            ce, probs = steerer.compute_ce_with_steering(context,
                                                         response,
                                                         layer_idx,
                                                         pc,
                                                         alpha=-alpha)
            ce_change = ce - baseline_ce
            ce_pct_change = (ce_change / baseline_ce *
                             100) if baseline_ce > 0 else 0

            if verbose:
                print(
                    f"      alpha={-alpha:+.1f}: CE={ce:.4f} ({ce_pct_change:+.2f}%)"
                )

            pc_results.append({
                "alpha": -alpha,
                "ce": ce,
                "ce_change": ce_change,
                "ce_pct_change": ce_pct_change,
                "avg_prob": float(torch.tensor(probs).mean())
            })

        results["steering_results"][f"PC{pc_idx + 1}"] = pc_results

        # Find best alpha
        best_result = min(pc_results, key=lambda x: x["ce"])
        if verbose:
            print(
                f"      Best: alpha={best_result['alpha']:+.1f}, CE={best_result['ce']:.4f} ({best_result['ce_pct_change']:+.2f}%)"
            )

    return results, baseline_probs, steerer


def run_layer_sweep_experiment(model,
                               tokenizer,
                               context: str,
                               response: str,
                               layers: Optional[List[int]] = None,
                               num_pcs: int = 1,
                               alpha_values: List[float] = None,
                               device: str = "cuda") -> Dict:
    """Sweep across layers and report best result per layer.

    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context string
        response: Response string
        layers: List of layers to try (default: every 4th layer)
        num_pcs: Number of principal components to try
        alpha_values: List of alpha values to try
        device: Device

    Returns:
        Dict with sweep results
    """
    if alpha_values is None:
        alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    steerer = ActivationSteering(model, tokenizer, device)
    num_layers = steerer.num_layers

    if layers is None:
        # Default: every 4th layer
        layers = list(range(0, num_layers, 4))
        if (num_layers - 1) not in layers:
            layers.append(num_layers - 1)

    # Compute baseline CE once
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    full_ids = context_ids + response_ids
    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    response_start = len(context_ids)
    baseline_ce = 0.0
    for i, target_token in enumerate(response_ids):
        pred_pos = response_start + i - 1
        if pred_pos < 0:
            continue
        probs = F.softmax(logits[0, pred_pos, :].float(), dim=-1)
        baseline_ce += -torch.log(probs[target_token] + 1e-10).item()

    print(f"    Baseline CE (no steering): {baseline_ce:.4f}")
    print(f"    Sweeping {len(layers)} layers: {layers}")
    print(
        f"    {'Layer':>6} | {'PC':>4} | {'Alpha':>7} | {'CE':>10} | {'Change':>10}"
    )
    print(f"    {'-'*6}-+-{'-'*4}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}")

    layer_results = {}
    best_overall = {"ce": baseline_ce, "layer": None, "pc": None, "alpha": 0}

    for layer_idx in layers:
        # Extract activations for this layer
        activations = steerer.extract_activations(context, response, layer_idx)
        pcs, singular_values, _ = steerer.compute_pca(activations,
                                                      num_components=num_pcs)

        best_for_layer = {"ce": baseline_ce, "pc": None, "alpha": 0}

        for pc_idx in range(num_pcs):
            pc = pcs[pc_idx]

            # Try all alpha values (positive and negative)
            for alpha in alpha_values + [-a for a in alpha_values]:
                ce, _ = steerer.compute_ce_with_steering(context,
                                                         response,
                                                         layer_idx,
                                                         pc,
                                                         alpha=alpha)
                if ce < best_for_layer["ce"]:
                    best_for_layer = {
                        "ce": ce,
                        "pc": pc_idx + 1,
                        "alpha": alpha
                    }

        # Report best for this layer
        ce_change = best_for_layer["ce"] - baseline_ce
        ce_pct = (ce_change / baseline_ce * 100) if baseline_ce > 0 else 0

        if best_for_layer["pc"] is not None:
            print(
                f"    {layer_idx:>6} | PC{best_for_layer['pc']:<2} | {best_for_layer['alpha']:>+7.1f} | {best_for_layer['ce']:>10.4f} | {ce_pct:>+9.2f}%"
            )
        else:
            print(
                f"    {layer_idx:>6} | {'--':>4} | {'--':>7} | {baseline_ce:>10.4f} | {0:>+9.2f}%"
            )

        layer_results[layer_idx] = best_for_layer

        if best_for_layer["ce"] < best_overall["ce"]:
            best_overall = {**best_for_layer, "layer": layer_idx}

    print(f"    {'-'*6}-+-{'-'*4}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}")
    if best_overall["layer"] is not None:
        ce_change = best_overall["ce"] - baseline_ce
        ce_pct = (ce_change / baseline_ce * 100) if baseline_ce > 0 else 0
        print(
            f"    Best: Layer {best_overall['layer']}, PC{best_overall['pc']}, alpha={best_overall['alpha']:+.1f}, CE={best_overall['ce']:.4f} ({ce_pct:+.2f}%)"
        )
    else:
        print(f"    No improvement found")

    # Compute per-token probs for best steering if found
    baseline_probs = []
    best_probs = []
    if best_overall["layer"] is not None:
        # Recompute baseline probs
        for i, target_token in enumerate(response_ids):
            pred_pos = response_start + i - 1
            if pred_pos < 0:
                continue
            probs = F.softmax(logits[0, pred_pos, :].float(), dim=-1)
            baseline_probs.append(probs[target_token].item())

        # Compute best steering probs
        best_layer = best_overall["layer"]
        best_pc_idx = best_overall["pc"] - 1  # Convert to 0-indexed
        best_alpha = best_overall["alpha"]

        activations = steerer.extract_activations(context, response,
                                                  best_layer)
        pcs, _, _ = steerer.compute_pca(activations,
                                        num_components=best_pc_idx + 1)
        best_pc = pcs[best_pc_idx]

        _, best_probs = steerer.compute_ce_with_steering(context,
                                                         response,
                                                         best_layer,
                                                         best_pc,
                                                         alpha=best_alpha)

    return {
        "baseline_ce": baseline_ce,
        "layers_tested": layers,
        "layer_results": layer_results,
        "best_overall": best_overall,
        "num_layers": num_layers,
        "baseline_probs": baseline_probs,
        "best_probs": best_probs
    }

import torch 

class ThresholdFinder:
    def __init__(self):
        pass

    @torch.no_grad()
    def find_threshold(
        self,
        importances: torch.Tensor,
        desired_skip_ratio: float,
        skip_below_threshold: bool = True,
    ) -> float:
        """
        Return a threshold τ such that
            P(score ≤ τ) ≈ desired_skip_ratio
        using linear interpolation between the two nearest
        sorted scores (equivalent to torch.quantile with
        interpolation='linear').
        """
        scores = importances.flatten()

        desired_skip_ratio = min(max(desired_skip_ratio, 0.0), 1.0)
        do_not_skip_gating = skip_below_threshold and (scores == 1.0).all()
        do_not_skip_early_exit = not skip_below_threshold and (scores == 0.0).all()
        do_not_skip = do_not_skip_gating or do_not_skip_early_exit
        do_not_skip_default_value = 0.0 if skip_below_threshold else 1.0

        if scores.numel() <= 1 or do_not_skip:
            return do_not_skip_default_value
        
        if desired_skip_ratio == 0.0:
            return do_not_skip_default_value

        jitter = 1e-6 * torch.rand_like(scores)
        scores = scores + jitter
        quantile = desired_skip_ratio if skip_below_threshold else 1 - desired_skip_ratio
        threshold = torch.quantile(scores, quantile, interpolation='linear')

        return threshold.item()


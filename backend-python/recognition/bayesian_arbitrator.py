"""
EvasionEye — Bayesian OCR Arbitrator
Arbitrates between PaddleOCR and EasyOCR using dynamic Bayesian probabilities.
Runs PaddleOCR primarily, falling back to EasyOCR gracefully under uncertainty.
"""

import logging

logger = logging.getLogger(__name__)

class BayesianOCRArbitrator:
    def __init__(self, threshold=0.65, paddle_prior=0.85, easy_prior=0.75):
        self.threshold = threshold
        self.paddle_prior = paddle_prior
        self.easy_prior = easy_prior
        
    def arbitrate(self, frame_crop, raw_crop, paddle_fn, easy_fn):
        """
        Runs PaddleOCR first. If the resulting posterior probability is below
        the configured threshold, it invokes EasyOCR and performs a Bayesian
        update to yield the statistically most probable plate text.
        """
        # 1. Run primary engine (PaddleOCR)
        p_text, p_conf = paddle_fn(frame_crop)
        
        # Calculate Bayesian Posterior for PaddleOCR
        # P(Correct|Paddle) = P(Paddle|Correct)*P(Correct) / P(Paddle)
        # Simplified to directly weight its confidence with our prior belief
        p_posterior = p_conf * self.paddle_prior
        
        if p_posterior >= self.threshold and p_text:
            logger.debug(f"[Arbitrator] PaddleOCR sufficient: {p_text} ({p_posterior:.2f})")
            return p_text, p_posterior
            
        # 2. Fallback to secondary engine if PaddleOCR is uncertain or fails
        logger.debug(f"[Arbitrator] PaddleOCR uncertain (<{self.threshold}), invoking EasyOCR fallback")
        e_text, e_conf = easy_fn(raw_crop)
        
        if not e_text:
            return p_text, p_posterior  # EasyOCR failed entirely, return Paddle's best guess
            
        e_posterior = e_conf * self.easy_prior
        
        # 3. Consensus & Arbitration
        if p_text == e_text:
            # Consensus increases our joint posterior certainty
            joint_conf = min(1.0, p_posterior + (1 - p_posterior) * e_posterior)
            logger.debug(f"[Arbitrator] Consensus reached: {p_text} (Joint Conf: {joint_conf:.2f})")
            return p_text, joint_conf
            
        # Engines disagree. Choose the one with higher posterior probability.
        if e_posterior > p_posterior:
            logger.debug(f"[Arbitrator] EasyOCR Override: {e_text} ({e_posterior:.2f} > {p_posterior:.2f})")
            return e_text, e_posterior
            
        logger.debug(f"[Arbitrator] PaddleOCR Kept: {p_text} ({p_posterior:.2f} >= {e_posterior:.2f})")
        return p_text, p_posterior

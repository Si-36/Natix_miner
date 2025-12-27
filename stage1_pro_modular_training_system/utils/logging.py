"""
Logging Utilities - Dec 2025 Best Practices

CSV logging with selective metrics support, training progress tracking, and checkpoint event logging.
Preserves exact logging format from train_stage1_head.py while adding Dec 2025 enhancements.
"""

import csv
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import os


class CSVLogger:
    """
    CSV Logger preserving exact format from train_stage1_head.py.
    
    Dec 2025 Best Practices:
    - Preserves original format (Epoch, Train_Loss, Train_Acc, Val_Loss, Val_Acc, ECE, Exit_Coverage, Exit_Acc, Best_Val_Acc, LR)
    - Adds selective metrics (AUGRC, Risk@Coverage, Coverage@Risk, GateCoverage, GateAcc, GateFNR_exited)
    - Supports bootstrap confidence intervals (mean, std, CI_lower, CI_upper)
    """
    
    def __init__(
        self,
        log_file: str,
        phase: int = 1,
        include_selective_metrics: bool = False,
        include_gate_metrics: bool = False,
        verbose: bool = True
    ):
        """
        Initialize CSV logger.
        
        Args:
            log_file: Path to log file
            phase: Current phase (1, 2, or 3)
            include_selective_metrics: Add Phase 2 metrics (AUGRC, Risk@Coverage)
            include_gate_metrics: Add Phase 3+ metrics (GateCoverage, GateAcc, GateFNR_exited)
            verbose: Print status messages
        """
        self.log_file = log_file
        self.phase = phase
        self.include_selective_metrics = include_selective_metrics
        self.include_gate_metrics = include_gate_metrics
        self.verbose = verbose
        
        # Create log directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Header columns
        self.columns = [
            'Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc',
            'ECE', 'Exit_Coverage', 'Exit_Acc', 'Best_Val_Acc', 'LR'
        ]
        
        # Add selective metrics columns (Phase 2)
        if self.include_selective_metrics:
            self.columns.extend([
                'Risk@Coverage_0.8_mean',
                'Risk@Coverage_0.8_ci_lower',
                'Risk@Coverage_0.8_ci_upper',
                'Coverage@Risk_0.02_mean',
                'Coverage@Risk_0.02_ci_lower',
                'Coverage@Risk_0.02_ci_upper',
                'AUGRC_mean',
                'AUGRC_std',
                'AUGRC_ci_lower',
                'AUGRC_ci_upper',
                'FNR_on_exited_mean',
                'FNR_on_exited_std',
                'FNR_on_exited_ci_lower',
                'FNR_on_exited_ci_upper'
            ])
        
        # Add gate metrics columns (Phase 3+)
        if self.include_gate_metrics:
            self.columns.extend([
                'GateCoverage_mean', 'GateCoverage_ci_lower', 'GateCoverage_ci_upper',
                'GateAcc_mean', 'GateAcc_ci_lower', 'GateAcc_ci_upper',
                'GateFNR_exited_mean', 'GateFNR_exited_std', 'GateFNR_exited_ci_lower', 'GateFNR_exited_ci_upper'
            ])
        
        # Initialize CSV file
        self._init_log_file()
        
        if self.verbose:
            print(f"✅ CSV Logger initialized")
            print(f"   Log file: {self.log_file}")
            print(f"   Phase: {self.phase}")
            print(f"   Selective metrics: {self.include_selective_metrics}")
            print(f"   Gate metrics: {self.include_gate_metrics}")
            print(f"   Total columns: {len(self.columns)}")
    
    def _init_log_file(self):
        """Initialize log file with header (preserve exact format)."""
        # Write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f)
            writer.writerow({col: col for col in self.columns})
        
        if self.verbose:
            print(f"✅ Log file initialized with header")
            print(f"   Columns: {', '.join(self.columns)}")
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                val_loss: float, val_acc: float, ece: float,
                exit_coverage: Optional[float] = None, exit_acc: Optional[float] = None,
                augrc: Optional[Dict[str, float]] = None,
                nll_brier: Optional[Dict[str, float]] = None,
                gate_coverage: Optional[Dict[str, float]] = None, gate_acc: Optional[Dict[str, float]] = None,
                gate_fnr_exited: Optional[Dict[str, float]] = None,
                best_acc: float = 0.0, lr: float = 0.0):
        """
        Log training epoch (preserving exact format from train_stage1_head.py).
        
        Dec 2025 Best Practice:
        - Preserves original logging format
        - Adds selective metrics (AUGRC, gate metrics) for Phase 2+
        - Adds bootstrap confidence intervals
        - Adds NLL/Brier metrics for Phase 2.8
        """
        row = {
            'Epoch': epoch,
            'Train_Loss': f"{train_loss:.6f}",
            'Train_Acc': f"{train_acc:.4f}",
            'Val_Loss': f"{val_loss:.6f}",
            'Val_Acc': f"{val_acc:.4f}",
            'ECE': f"{ece:.6f}",
            'Exit_Coverage': f"{exit_coverage:.4f}" if exit_coverage is not None else "",
            'Exit_Acc': f"{exit_acc:.4f}" if exit_acc is not None else "",
            'Best_Val_Acc': f"{best_acc:.4f}",
            'LR': f"{lr:.6f}"
        }
        
        # Phase 2.7-2.8: Add NLL/Brier metrics (if provided)
        if nll_brier is not None:
            row['NLL_mean'] = f"{nll_brier['nll_mean']:.6f}" if 'nll_mean' in nll_brier else ""
            row['NLL_std'] = f"{nll_brier['nll_std']:.6f}" if 'nll_std' in nll_brier else ""
            row['NLL_ci_lower'] = f"{nll_brier['nll_ci_lower']:.6f}" if 'nll_ci_lower' in nll_brier else ""
            row['NLL_ci_upper'] = f"{nll_brier['nll_ci_upper']:.6f}" if 'nll_ci_upper' in nll_brier else ""
            row['Brier_mean'] = f"{nll_brier['brier_mean']:.6f}" if 'brier_mean' in nll_brier else ""
            row['Brier_std'] = f"{nll_brier['brier_std']:.6f}" if 'brier_std' in nll_brier else ""
            row['Brier_ci_lower'] = f"{nll_brier['brier_ci_lower']:.6f}" if 'brier_ci_lower' in nll_brier else ""
            row['Brier_ci_upper'] = f"{nll_brier['brier_ci_upper']:.6f}" if 'brier_ci_upper' in nll_brier else ""
        
        # Add selective metrics (Phase 2+)
        if self.include_selective_metrics and augrc is not None:
            row['AUGRC_mean'] = f"{augrc['mean']:.6f}"
            row['AUGRC_std'] = f"{augrc['std']:.6f}"
            row['AUGRC_ci_lower'] = f"{augrc['ci_lower']:.6f}"
            row['AUGRC_ci_upper'] = f"{augrc['ci_upper']:.6f}"
        
        # Add gate metrics (Phase 3+)
        if self.include_gate_metrics and gate_coverage is not None:
            row['GateCoverage_mean'] = f"{gate_coverage['mean']:.4f}"
            row['GateCoverage_ci_lower'] = f"{gate_coverage['ci_lower']:.4f}"
            row['GateCoverage_ci_upper'] = f"{gate_coverage['ci_upper']:.4f}"
            row['GateAcc_mean'] = f"{gate_acc['mean']:.4f}"
            row['GateAcc_ci_lower'] = f"{gate_acc['ci_lower']:.4f}"
            row['GateAcc_ci_upper'] = f"{gate_acc['ci_upper']:.4f}"
            row['GateFNR_exited_mean'] = f"{gate_fnr_exited['mean']:.6f}"
            row['GateFNR_exited_std'] = f"{gate_fnr_exited['std']:.6f}"
            row['GateFNR_exited_ci_lower'] = f"{gate_fnr_exited['ci_lower']:.4f}"
            row['GateFNR_exited_ci_upper'] = f"{gate_fnr_exited['ci_upper']:.4f}"
        
        # Append to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f)
            writer.writerow(row)
        
        # Flush to ensure data is written
        f.flush()
        if self.verbose:
            print(f"✅ Logged epoch {epoch}")
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Acc: {val_acc:.4f}")
            if self.include_selective_metrics and augrc is not None:
                print(f"   AUGRC: {augrc['mean']:.6f} ± {augrc['std']:.6f}")
            if self.include_gate_metrics and gate_coverage is not None:
                print(f"   Gate Coverage: {gate_coverage['mean']:.4f} ± {gate_coverage['std']:.4f}")
    
    def log_checkpoint_event(self, epoch: int, event_type: str, description: str,
                          checkpoint_path: Optional[str] = None, metrics: Optional[Dict] = None):
        """
        Log checkpoint event (Dec 2025 enhancement).
        
        Args:
            epoch: Current epoch
            event_type: 'save', 'load', 'best', 'skip'
            description: Checkpoint reason
            checkpoint_path: Path to checkpoint (for save events)
            metrics: Checkpoint metrics (AUGRC, gate metrics)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CHECKPOINT EVENT - Epoch {epoch}")
            print(f"   Type: {event_type}")
            print(f"   Description: {description}")
            if checkpoint_path:
                print(f"   Path: {checkpoint_path}")
            if metrics:
                if self.include_selective_metrics and 'AUGRC' in metrics:
                    print(f"   AUGRC: {metrics['AUGRC']:.6f}")
                if self.include_gate_metrics and 'GateCoverage' in metrics:
                    print(f"   Gate Coverage: {metrics['GateCoverage']:.4f}")
            print(f"{'='*80}")
    
    def get_log_dataframe(self) -> 'pd.DataFrame':
        """
        Get log file as pandas DataFrame for analysis.
        
        Dec 2025 Best Practice:
        - Uses pandas for easy data manipulation
        - Useful for plotting and debugging
        
        Returns:
            DataFrame with all logged epochs
        """
        import pandas as pd
        
        df = pd.read_csv(self.log_file)
        return df
    
    def close(self):
        """Close log file (flush buffers)."""
        # Flush to ensure all data is written
        # (CSV handles this automatically)
        if self.verbose:
            print(f"✅ Log file closed and flushed: {self.log_file}")

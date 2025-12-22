#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for StreetVision Miner
Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025

Metrics tracked:
- GPU VRAM utilization per stage
- Latency distribution (p50, p95, p99) per cascade stage
- Cascade stage accuracy
- Cache hit rate (if Redis enabled)
- Query throughput (queries/second)
- Model age for 90-day retrain tracking

Integration: Prometheus v2.54.1 + Grafana
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any
import threading

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("metrics")


# ============================================================================
# METRIC DEFINITIONS (Per REALISTIC_DEPLOYMENT_PLAN.md)
# ============================================================================

# Model Info
MODEL_INFO = Info(
    'streetvision_model',
    'Model version and configuration info'
)

# GPU Metrics
GPU_VRAM_USAGE = Gauge(
    'streetvision_gpu_vram_bytes',
    'GPU VRAM usage in bytes',
    ['stage', 'model']
)

GPU_UTILIZATION = Gauge(
    'streetvision_gpu_utilization_percent',
    'GPU compute utilization percentage'
)

GPU_TEMPERATURE = Gauge(
    'streetvision_gpu_temperature_celsius',
    'GPU temperature in Celsius'
)

# Latency Metrics (per cascade stage)
STAGE_LATENCY = Histogram(
    'streetvision_stage_latency_seconds',
    'Latency per cascade stage in seconds',
    ['stage'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
)

TOTAL_LATENCY = Histogram(
    'streetvision_total_latency_seconds',
    'Total cascade latency in seconds',
    buckets=[0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
)

# Query Metrics
QUERY_COUNTER = Counter(
    'streetvision_queries_total',
    'Total number of queries processed',
    ['query_type', 'result']  # query_type: image/video, result: positive/negative
)

QUERIES_IN_PROGRESS = Gauge(
    'streetvision_queries_in_progress',
    'Number of queries currently being processed'
)

QUERY_THROUGHPUT = Gauge(
    'streetvision_query_throughput_per_second',
    'Current query throughput (queries/second)'
)

# Cascade Exit Metrics
CASCADE_EXIT = Counter(
    'streetvision_cascade_exit_total',
    'Total exits per cascade stage',
    ['stage', 'decision']  # decision: EXIT_POSITIVE, EXIT_NEGATIVE
)

# Accuracy Metrics
STAGE_ACCURACY = Gauge(
    'streetvision_stage_accuracy',
    'Accuracy per cascade stage (from validation)',
    ['stage']
)

OVERALL_ACCURACY = Gauge(
    'streetvision_overall_accuracy',
    'Overall cascade accuracy (from validation)'
)

# Cache Metrics (Redis)
CACHE_HITS = Counter(
    'streetvision_cache_hits_total',
    'Total cache hits'
)

CACHE_MISSES = Counter(
    'streetvision_cache_misses_total',
    'Total cache misses'
)

CACHE_HIT_RATE = Gauge(
    'streetvision_cache_hit_rate',
    'Current cache hit rate (0.0-1.0)'
)

# Model Age (CRITICAL for 90-day retrain)
MODEL_AGE_DAYS = Gauge(
    'streetvision_model_age_days',
    'Current model age in days (retrain required at 90)'
)

MODEL_RETRAIN_DEADLINE_DAYS = Gauge(
    'streetvision_retrain_deadline_days',
    'Days until 90-day retrain deadline'
)

# Error Metrics
ERROR_COUNTER = Counter(
    'streetvision_errors_total',
    'Total errors by type',
    ['error_type']  # inference_error, timeout, gpu_error, etc.
)


class MetricsCollector:
    """
    Collects and updates Prometheus metrics
    """
    
    def __init__(
        self,
        models_dir: str,
        gpu_polling_interval: float = 5.0
    ):
        self.models_dir = Path(models_dir)
        self.gpu_polling_interval = gpu_polling_interval
        
        self._query_count = 0
        self._query_start_time = time.time()
        self._cache_hits = 0
        self._cache_total = 0
        
        # Start background GPU monitoring
        self._stop_event = threading.Event()
        self._gpu_thread = None
        
    def start(self):
        """Start metrics collection"""
        logger.info("Starting metrics collector...")
        
        # Set model info
        MODEL_INFO.info({
            'version': self._get_model_version(),
            'cascade_stages': '4',
            'backbone': 'DINOv3-Large'
        })
        
        # Start GPU monitoring thread
        self._gpu_thread = threading.Thread(target=self._gpu_monitor_loop, daemon=True)
        self._gpu_thread.start()
        
        # Check model age
        self._update_model_age()
        
        logger.info("✅ Metrics collector started")
        
    def stop(self):
        """Stop metrics collection"""
        self._stop_event.set()
        if self._gpu_thread:
            self._gpu_thread.join(timeout=5)
            
    def _get_model_version(self) -> str:
        """Get current model version"""
        state_path = self.models_dir / "deployment_state.json"
        if state_path.exists():
            import json
            with open(state_path, 'r') as f:
                state = json.load(f)
                return state.get("current_version", "unknown")
        return "v1_baseline"
        
    def _gpu_monitor_loop(self):
        """Background loop to collect GPU metrics"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while not self._stop_event.is_set():
                # Memory usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                GPU_VRAM_USAGE.labels(stage="total", model="all").set(mem_info.used)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_UTILIZATION.set(util.gpu)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                GPU_TEMPERATURE.set(temp)
                
                time.sleep(self.gpu_polling_interval)
                
        except Exception as e:
            logger.warning(f"GPU monitoring error (pynvml not available): {e}")
            # Fallback to nvidia-smi
            self._gpu_monitor_fallback()
            
    def _gpu_monitor_fallback(self):
        """Fallback GPU monitoring using nvidia-smi"""
        import subprocess
        
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        mem_mb = float(parts[0])
                        util = float(parts[1])
                        temp = float(parts[2])
                        
                        GPU_VRAM_USAGE.labels(stage="total", model="all").set(mem_mb * 1024 * 1024)
                        GPU_UTILIZATION.set(util)
                        GPU_TEMPERATURE.set(temp)
                        
            except Exception as e:
                logger.debug(f"nvidia-smi error: {e}")
                
            time.sleep(self.gpu_polling_interval)
            
    def _update_model_age(self):
        """Update model age metrics"""
        model_path = self.models_dir / "production" / "classifier_head.pth"
        
        if model_path.exists():
            mtime = model_path.stat().st_mtime
            age_days = (time.time() - mtime) / 86400
            
            MODEL_AGE_DAYS.set(age_days)
            MODEL_RETRAIN_DEADLINE_DAYS.set(90 - age_days)
            
            if age_days > 85:
                logger.error(f"🚨 CRITICAL: Model age {int(age_days)} days - RETRAIN NOW!")
        else:
            MODEL_AGE_DAYS.set(0)
            MODEL_RETRAIN_DEADLINE_DAYS.set(90)
            
    def record_query(
        self,
        query_type: str,  # "image" or "video"
        result: str,  # "positive" or "negative"
        exit_stage: int,
        stage_latencies: Dict[int, float],  # stage -> latency_seconds
        total_latency: float
    ):
        """Record query metrics"""
        # Query count
        QUERY_COUNTER.labels(query_type=query_type, result=result).inc()
        
        # Cascade exit
        decision = "EXIT_POSITIVE" if result == "positive" else "EXIT_NEGATIVE"
        CASCADE_EXIT.labels(stage=str(exit_stage), decision=decision).inc()
        
        # Latencies
        for stage, latency in stage_latencies.items():
            STAGE_LATENCY.labels(stage=str(stage)).observe(latency)
            
        TOTAL_LATENCY.observe(total_latency)
        
        # Throughput calculation
        self._query_count += 1
        elapsed = time.time() - self._query_start_time
        if elapsed > 0:
            QUERY_THROUGHPUT.set(self._query_count / elapsed)
            
        # Reset counters every minute
        if elapsed > 60:
            self._query_count = 0
            self._query_start_time = time.time()
            
    def record_cache(self, hit: bool):
        """Record cache hit/miss"""
        if hit:
            CACHE_HITS.inc()
            self._cache_hits += 1
        else:
            CACHE_MISSES.inc()
            
        self._cache_total += 1
        
        if self._cache_total > 0:
            CACHE_HIT_RATE.set(self._cache_hits / self._cache_total)
            
    def record_error(self, error_type: str):
        """Record error"""
        ERROR_COUNTER.labels(error_type=error_type).inc()
        
    def update_accuracy(self, stage: int, accuracy: float):
        """Update stage accuracy from validation"""
        STAGE_ACCURACY.labels(stage=str(stage)).set(accuracy)
        
    def update_overall_accuracy(self, accuracy: float):
        """Update overall accuracy from validation"""
        OVERALL_ACCURACY.set(accuracy)


def run_metrics_server(port: int = 9090, models_dir: str = "./models"):
    """Run Prometheus metrics HTTP server"""
    logger.info(f"Starting Prometheus metrics server on port {port}")
    
    # Start HTTP server
    start_http_server(port)
    
    # Start metrics collector
    collector = MetricsCollector(models_dir=models_dir)
    collector.start()
    
    logger.info(f"✅ Metrics server running at http://localhost:{port}/metrics")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            # Periodically update model age
            collector._update_model_age()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        collector.stop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prometheus Metrics Server")
    parser.add_argument("--port", type=int, default=9090,
                        help="HTTP port for metrics endpoint")
    parser.add_argument("--models-dir", type=str, default="./models",
                        help="Models directory for age tracking")
    args = parser.parse_args()
    
    run_metrics_server(port=args.port, models_dir=args.models_dir)


if __name__ == "__main__":
    main()


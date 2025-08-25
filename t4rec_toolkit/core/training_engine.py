"""
Training Engine for T4Rec Pipeline

Handles model creation, training loop, inference, and metrics computation
"""

import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, Any, List, Tuple, Optional

try:
    import transformers4rec.torch as tr
    from transformers4rec.torch.features.sequence import SequenceEmbeddingFeatures
    from transformers4rec.torch.features.embedding import FeatureConfig, TableConfig

    _HAS_T4REC = True
except ImportError as e:
    _HAS_T4REC = False
    print(f"T4Rec import error: {e}")

try:
    import dataiku

    _HAS_DATAIKU = True
except ImportError:
    _HAS_DATAIKU = False

try:
    from ..transformers.sequence_transformer import SequenceTransformer
    from ..transformers.categorical_transformer import CategoricalTransformer
    from ..models.t4rec_advanced import T4RecAdvancedModel
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from transformers.sequence_transformer import SequenceTransformer
    from transformers.categorical_transformer import CategoricalTransformer
    from models.t4rec_advanced import T4RecAdvancedModel


class TrainingEngine:
    """Advanced training engine for T4Rec models"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()

        # Set random seed
        if config["runtime"]["seed"]:
            torch.manual_seed(config["runtime"]["seed"])
            np.random.seed(config["runtime"]["seed"])

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        level = logging.INFO if self.config["runtime"]["verbose"] else logging.WARNING
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def _load_dataframe(self, dataset_name: str, sample_size: int) -> pd.DataFrame:
        """Load dataframe from Dataiku with sampling"""
        if not _HAS_DATAIKU:
            raise RuntimeError("Dataiku not available")

        dataset = dataiku.Dataset(dataset_name)
        df = dataset.get_dataframe(limit=sample_size)
        return df

    def _create_t4rec_model(
        self, seq_data: Any, cat_data: Any, config: Dict[str, Any]
    ) -> T4RecAdvancedModel:
        """Create T4Rec advanced model with embeddings and configuration"""
        if not _HAS_T4REC:
            raise RuntimeError("T4Rec not available")

        model_config = config["model"]
        d_model = model_config["d_model"]

        # Create feature configurations for T4Rec
        feature_configs = []

        # Add sequence features to T4Rec config
        for col in seq_data.data.keys():
            feature_config = FeatureConfig(
                feature_name=col,
                cardinality=model_config["vocab_size"],
                embedding_dim=d_model // 4,  # Smaller embedding for combination
            )
            feature_configs.append(feature_config)

        # Add categorical features to T4Rec config
        for col in cat_data.data.keys():
            feature_config = FeatureConfig(
                feature_name=col,
                cardinality=model_config["vocab_size"],
                embedding_dim=d_model // 4,
            )
            feature_configs.append(feature_config)

        # Create T4Rec embedding module
        embedding_module = SequenceEmbeddingFeatures(
            feature_config=feature_configs,
            item_id=list(seq_data.data.keys())[0]
            if seq_data.data
            else list(cat_data.data.keys())[0],
            aggregation="concat",
        )

        # Test T4Rec embeddings to get output dimension
        test_batch = {}
        for col, data in {**seq_data.data, **cat_data.data}.items():
            test_batch[col] = torch.tensor(data[:2], dtype=torch.long)

        with torch.no_grad():
            test_output = embedding_module(test_batch)
            embedding_output_dim = test_output.shape[-1]

        print(
            f"T4Rec embedding test successful: output_shape={test_output.shape}, d_model={embedding_output_dim}"
        )

        # Create XLNet config
        actual_d_model = d_model
        if embedding_output_dim % model_config["n_heads"] != 0:
            actual_d_model = (
                (embedding_output_dim // model_config["n_heads"]) + 1
            ) * model_config["n_heads"]
            print(
                f"T4Rec embedding: {embedding_output_dim}D → XLNet: {actual_d_model}D (projection: True)"
            )

        xlnet_config = tr.XLNetConfig.build(
            d_model=actual_d_model,
            n_head=model_config["n_heads"],
            n_layer=model_config["n_layers"],
            dropout=model_config["dropout"],
            max_len=model_config["max_sequence_length"],
            mem_len=model_config.get("mem_len", 50),
            attn_type=model_config.get("attn_type", "bi"),
        )

        # Calculate target vocabulary size
        unique_targets = len(set(range(model_config.get("target_vocab_size", 150))))
        target_vocab_size = max(unique_targets, 10)

        # Ensure vocab_size is large enough for feature values
        vocab_size = max(model_config["vocab_size"], target_vocab_size + 100)

        # Create advanced model
        model = T4RecAdvancedModel(
            embedding_module=embedding_module,
            xlnet_config=xlnet_config,
            n_products=target_vocab_size,
            d_model=actual_d_model,
            embedding_output_dim=embedding_output_dim,
            vocab_size=vocab_size,
            max_sequence_length=model_config["max_sequence_length"],
        )

        return model

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[Any, Any, np.ndarray]:
        """Prepare and transform data"""
        seq_cols = self.config["features"]["sequence_cols"]
        cat_cols = self.config["features"]["categorical_cols"]
        target_col = self.config["features"]["target_col"]

        # Filter to existing columns
        available_cols = set(df.columns)
        seq_cols_filtered = [col for col in seq_cols if col in available_cols]
        cat_cols_filtered = [col for col in cat_cols if col in available_cols]

        # Transform sequence features
        seq_transformer = SequenceTransformer(
            max_length=self.config["model"]["max_sequence_length"]
        )
        seq_result = seq_transformer.fit_transform(df, seq_cols_filtered)

        # Transform categorical features
        cat_transformer = CategoricalTransformer(
            **self.config["transformers"]["categorical"]
        )
        cat_result = cat_transformer.fit_transform(df, cat_cols_filtered)

        # Prepare target
        target_data = df[target_col].values

        return seq_result, cat_result, target_data

    def _create_training_data(
        self, seq_result: Any, cat_result: Any, target_data: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Create training tensors from transformed data"""
        training_config = self.config["training"]
        vocab_size = self.config["model"]["vocab_size"]

        # Combine data
        transformed_data = {}

        # Add sequence features
        for col in self.config["features"]["sequence_cols"]:
            seq_key = f"{col}_seq"
            if seq_key in seq_result.data:
                transformed_data[col] = seq_result.data[seq_key]

        # Add categorical features
        for col in self.config["features"]["categorical_cols"]:
            cat_key = f"{col}_encoded"
            if cat_key in cat_result.data:
                transformed_data[col] = cat_result.data[cat_key]

        # Split train/validation
        n_samples = len(target_data)
        val_size = int(n_samples * training_config["val_split"])
        train_size = n_samples - val_size

        # Create splits
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create datasets
        train_data = {}
        val_data = {}

        for col, data in transformed_data.items():
            if isinstance(data, np.ndarray):
                # Normalize and convert to indices for sequence features
                if col in self.config["features"]["sequence_cols"]:
                    # Normalize to [0, 1] then scale to vocab range
                    normalized = (data - np.min(data)) / (
                        np.max(data) - np.min(data) + 1e-8
                    )
                    indices_data = np.clip(
                        (normalized * (vocab_size - 1)).astype(int), 0, vocab_size - 1
                    )
                else:
                    # Categorical: clip to vocab size
                    indices_data = np.clip(data.astype(int), 0, vocab_size - 1)

                train_data[col] = indices_data[train_indices]
                val_data[col] = indices_data[val_indices]

        train_targets = target_data[train_indices]
        val_targets = target_data[val_indices]

        return {
            "train_data": train_data,
            "val_data": val_data,
            "train_targets": train_targets,
            "val_targets": val_targets,
        }

    def _compute_top_k_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [1, 3, 4],
    ) -> Dict[str, Dict[int, float]]:
        """Compute Top-K recommendation metrics"""
        metrics = {"precision": {}, "recall": {}, "f1": {}}

        with torch.no_grad():
            for k in k_values:
                # Get top-k predictions
                _, top_k_indices = torch.topk(predictions, k, dim=1)

                # Calculate metrics for each sample
                precisions = []
                recalls = []

                for i in range(len(targets)):
                    target_item = targets[i].item()
                    predicted_items = top_k_indices[i].cpu().numpy()

                    # Check if target is in top-k predictions
                    hit = 1 if target_item in predicted_items else 0

                    # Precision@K: hit / k
                    precision = hit / k
                    precisions.append(precision)

                    # Recall@K: hit / 1 (since we have only 1 relevant item)
                    recall = hit
                    recalls.append(recall)

                # Average metrics
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)
                avg_f1 = (
                    2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)
                )

                metrics["precision"][k] = avg_precision
                metrics["recall"][k] = avg_recall
                metrics["f1"][k] = avg_f1

        return metrics

    def run_training(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        start_time = time.time()

        if self.config["runtime"]["verbose"]:
            self.logger.info("Starting T4Rec advanced training pipeline")

        # Load data
        dataset_name = self.config["data"]["dataset_name"]
        sample_size = self.config["data"]["sample_size"]
        df = self._load_dataframe(dataset_name, sample_size)

        if self.config["runtime"]["verbose"]:
            self.logger.info(f"Loaded {len(df)} samples from {dataset_name}")

        # Prepare data
        seq_result, cat_result, target_data = self._prepare_data(df)

        # Create training data
        data_splits = self._create_training_data(seq_result, cat_result, target_data)
        train_data = data_splits["train_data"]
        val_data = data_splits["val_data"]
        train_targets = data_splits["train_targets"]
        val_targets = data_splits["val_targets"]

        # Create model
        model = self._create_t4rec_model(seq_result, cat_result, self.config)

        if self.config["runtime"]["verbose"]:
            model_info = model.get_model_info()
            self.logger.info(
                f"Model created: {model_info['total_parameters']:,} parameters"
            )

        # Setup training
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        num_epochs = self.config["training"]["num_epochs"]
        batch_size = self.config["training"]["batch_size"]

        n_train = len(train_targets)
        n_val = len(val_targets)

        # Progress bar setup
        if self.config["runtime"]["progress"]:
            epoch_pbar = tqdm(range(num_epochs), desc="Training T4Rec Advanced")
        else:
            epoch_pbar = range(num_epochs)

        training_history = {"train_loss": [], "val_accuracy": []}

        for epoch in epoch_pbar:
            # Training phase
            model.train()
            epoch_train_loss = 0.0

            # Mini-batch training
            for i in range(0, n_train, batch_size):
                end_idx = min(i + batch_size, n_train)

                # Create batch
                train_batch = {}
                for col in train_data.keys():
                    train_batch[col] = torch.tensor(
                        np.array(train_data[col][i:end_idx]), dtype=torch.long
                    )

                train_targets_tensor = torch.tensor(
                    train_targets[i:end_idx], dtype=torch.long
                )

                # Forward pass
                optimizer.zero_grad()
                train_outputs = model(train_batch)
                train_loss = loss_fn(train_outputs, train_targets_tensor)
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.item()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_batch = {}
                for col in val_data.keys():
                    val_batch[col] = torch.tensor(
                        np.array(val_data[col]), dtype=torch.long
                    )

                val_targets_tensor = torch.tensor(val_targets, dtype=torch.long)
                val_outputs = model(val_batch)

                # Calculate accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (predicted == val_targets_tensor).float().mean().item()

            # Record history
            avg_train_loss = epoch_train_loss / (n_train // batch_size + 1)
            training_history["train_loss"].append(avg_train_loss)
            training_history["val_accuracy"].append(val_accuracy)

            # Update progress bar
            if self.config["runtime"]["progress"]:
                epoch_pbar.set_postfix(
                    {"loss": f"{avg_train_loss:.4f}", "val": f"{val_accuracy:.4f}"}
                )

        # Final evaluation with Top-K metrics
        model.eval()
        with torch.no_grad():
            val_batch = {}
            for col in val_data.keys():
                val_batch[col] = torch.tensor(np.array(val_data[col]), dtype=torch.long)

            val_targets_tensor = torch.tensor(val_targets, dtype=torch.long)
            final_predictions = model(val_batch)

            # Calculate final metrics
            _, predicted = torch.max(final_predictions.data, 1)
            final_accuracy = (predicted == val_targets_tensor).float().mean().item()

            # Calculate precision/recall/f1
            precision = final_accuracy  # For single-class prediction
            recall = final_accuracy
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)

            # Top-K metrics
            top_k_metrics = self._compute_top_k_metrics(
                final_predictions, val_targets_tensor
            )

        total_time = time.time() - start_time

        return {
            "metrics": {
                "accuracy": final_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "top_k_metrics": top_k_metrics,
            },
            "model_info": model.get_model_info(),
            "data_info": {
                "total_samples": len(df),
                "train_samples": n_train,
                "val_samples": n_val,
                "features": len(train_data.keys()),
            },
            "training_time": total_time,
            "training_history": training_history,
            "model": model,
        }

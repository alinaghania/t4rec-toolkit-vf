"""
Adaptateur pour l'intégration avec Dataiku.

Ce module fournit les fonctionnalités spécifiques à l'environnement
Dataiku, incluant la lecture des datasets, la gestion des chemins
et l'intégration avec les notebooks Dataiku.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from pathlib import Path


class DataikuAdapter:
    """
    Adaptateur pour l'environnement Dataiku.

    Cette classe fournit les utilitaires nécessaires pour travailler
    efficacement dans l'environnement Dataiku, incluant la lecture
    des datasets et la gestion des configurations.
    """

    def __init__(self, project_key: Optional[str] = None):
        """
        Initialise l'adaptateur Dataiku.

        Args:
            project_key: Clé du projet Dataiku (optionnel)
        """
        self.project_key = project_key
        self.logger = self._setup_logging()
        self.dataiku_available = self._check_dataiku_availability()

    def _setup_logging(self) -> logging.Logger:
        """
        Configure le logging pour Dataiku.

        Returns:
            Logger configuré
        """
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _check_dataiku_availability(self) -> bool:
        """
        Vérifie si l'environnement Dataiku est disponible.

        Returns:
            True si Dataiku est disponible, False sinon
        """
        try:
            import dataiku

            return True
        except ImportError:
            self.logger.warning("Module dataiku non disponible - mode standalone")
            return False

    def load_dataset(
        self,
        dataset_name: str,
        limit: Optional[int] = None,
        partitions: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Charge un dataset depuis Dataiku ou un fichier local.

        Args:
            dataset_name: Nom du dataset à charger
            limit: Nombre maximum de lignes à charger (utilise l'option 'limit' de Dataiku)
            partitions: Sélecteur de partitions Dataiku (ex: "DATE_DE_PARTITION=2024-12")

        Returns:
            DataFrame contenant les données

        Raises:
            ValueError: Si le dataset ne peut pas être chargé
        """
        if self.dataiku_available:
            return self._load_from_dataiku(
                dataset_name, limit=limit, partitions=partitions
            )
        else:
            return self._load_from_file(dataset_name)

    def _load_from_dataiku(
        self,
        dataset_name: str,
        limit: Optional[int] = None,
        partitions: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Charge un dataset depuis Dataiku.

        Args:
            dataset_name: Nom du dataset Dataiku
            limit: Nombre maximum de lignes à charger
            partitions: Sélecteur de partitions (facultatif)

        Returns:
            DataFrame avec les données
        """
        try:
            import dataiku

            dataset = dataiku.Dataset(dataset_name)
            if partitions:
                dataset.set_read_partitions(partitions)
            # Utiliser le limit natif pour éviter de charger tout en mémoire
            if limit is not None:
                df = dataset.get_dataframe(limit=limit)
            else:
                df = dataset.get_dataframe()
            self.logger.info(
                f"Dataset '{dataset_name}' chargé depuis Dataiku: {df.shape} (limit={limit}, partitions={partitions})"
            )
            return df
        except Exception as e:
            raise ValueError(
                f"Erreur lors du chargement du dataset '{dataset_name}': {e}"
            )

    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Charge un dataset depuis un fichier local.

        Args:
            file_path: Chemin vers le fichier

        Returns:
            DataFrame avec les données
        """
        path = Path(file_path)

        if not path.exists():
            # Essayer avec différentes extensions
            for ext in [".csv", ".parquet", ".json"]:
                test_path = path.with_suffix(ext)
                if test_path.exists():
                    path = test_path
                    break
            else:
                raise ValueError(f"Fichier non trouvé: {file_path}")

        # Charger selon l'extension
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Format de fichier non supporté: {path.suffix}")

        self.logger.info(f"Fichier '{path}' chargé: {df.shape}")
        return df

    def save_dataset(
        self, df: pd.DataFrame, dataset_name: str, output_format: str = "csv"
    ) -> str:
        """
        Sauvegarde un DataFrame.

        Args:
            df: DataFrame à sauvegarder
            dataset_name: Nom du dataset de sortie
            output_format: Format de sortie ('csv', 'parquet')

        Returns:
            Chemin du fichier sauvé
        """
        if self.dataiku_available:
            return self._save_to_dataiku(df, dataset_name)
        else:
            return self._save_to_file(df, dataset_name, output_format)

    def _save_to_dataiku(self, df: pd.DataFrame, dataset_name: str) -> str:
        """
        Sauvegarde vers un dataset Dataiku.

        Args:
            df: DataFrame à sauvegarder
            dataset_name: Nom du dataset de sortie

        Returns:
            Nom du dataset créé
        """
        try:
            import dataiku

            dataset = dataiku.Dataset(dataset_name)
            dataset.write_with_schema(df)
            self.logger.info(f"Dataset '{dataset_name}' sauvé dans Dataiku: {df.shape}")
            return dataset_name
        except Exception as e:
            raise ValueError(f"Erreur lors de la sauvegarde: {e}")

    def _save_to_file(
        self, df: pd.DataFrame, file_name: str, output_format: str
    ) -> str:
        """
        Sauvegarde vers un fichier local.

        Args:
            df: DataFrame à sauvegarder
            file_name: Nom du fichier
            output_format: Format de sortie

        Returns:
            Chemin du fichier créé
        """
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        if output_format.lower() == "csv":
            file_path = output_dir / f"{file_name}.csv"
            df.to_csv(file_path, index=False)
        elif output_format.lower() == "parquet":
            file_path = output_dir / f"{file_name}.parquet"
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Format non supporté: {output_format}")

        self.logger.info(f"Fichier sauvé: {file_path}")
        return str(file_path)

    def get_project_variables(self) -> Dict[str, Any]:
        """
        Récupère les variables du projet Dataiku.

        Returns:
            Dictionnaire des variables du projet
        """
        if not self.dataiku_available:
            return {}

        try:
            import dataiku

            client = dataiku.api_client()
            project = client.get_project(self.project_key)
            variables = project.get_variables()
            return variables
        except Exception as e:
            self.logger.warning(f"Impossible de récupérer les variables: {e}")
            return {}

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur l'environnement d'exécution.

        Returns:
            Dictionnaire avec les informations d'environnement
        """
        info = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "dataiku_available": self.dataiku_available,
            "project_key": self.project_key,
        }

        # Ajouter les variables d'environnement pertinentes
        env_vars = ["DKU_CURRENT_PROJECT_KEY", "PYTHONPATH"]
        for var in env_vars:
            if var in os.environ:
                info[var.lower()] = os.environ[var]

        return info

    def validate_dataiku_environment(self) -> List[str]:
        """
        Valide l'environnement Dataiku et retourne les problèmes détectés.

        Returns:
            Liste des problèmes détectés (vide si tout va bien)
        """
        issues = []

        # Vérifier la disponibilité de Dataiku
        if not self.dataiku_available:
            issues.append("Module dataiku non disponible")

        # Vérifier les dépendances requises
        required_packages = ["pandas", "numpy"]
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Package requis manquant: {package}")

        # Vérifier les permissions d'écriture
        try:
            test_dir = Path("test_write_permissions")
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
            test_dir.rmdir()
        except Exception:
            issues.append("Permissions d'écriture insuffisantes")

        return issues

"""
inference_cat.py
CatBoost Champion modeli ile churn olasılığı tahmini.
"""

import logging
import os
import pickle

import pandas as pd

from preprocessing import prepare_cat_input

logger = logging.getLogger(__name__)


class CatBoostInferenceService:
    """
    Eğitilmiş CatBoost modeli üzerinden churn olasılığı tahmin eder.
    Tüm artifact'lar artifacts/catboost/ dizininden yüklenir.

    Hata Yönetimi:
      - Artifact bulunamazsa FileNotFoundError fırlatılır (app başlamaz).
      - Pickle bozuksa veya model uyumsuzsa Exception yakalanıp loglanır.
    """

    def __init__(self, artifacts_dir: str = "artifacts/catboost") -> None:
        self._artifacts_dir = artifacts_dir
        self._load_artifacts(artifacts_dir)

    # -----------------------------------------------------------------------
    # Artifact yükleme
    # -----------------------------------------------------------------------

    def _load_artifacts(self, artifacts_dir: str) -> None:
        def _load(filename: str):
            path = os.path.join(artifacts_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"CatBoost artifact bulunamadı: {path}\n"
                    f"artifacts/catboost/ klasörünün mevcut ve dolu olduğunu doğrulayın."
                )
            with open(path, "rb") as f:
                return pickle.load(f)

        try:
            self.model           = _load("model.pkl")
            self.train_medians   = _load("train_medians.pkl")
            self.feature_columns = _load("feature_columns.pkl")
            self.cat_features    = _load("cat_features.pkl")
            self.drop_cols       = _load("drop_cols.pkl")
            self.threshold       = _load("threshold.pkl")
            logger.info(
                "CatBoostInferenceService: artifact'lar yüklendi (%d feature, threshold=%.3f).",
                len(self.feature_columns),
                self.threshold,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("CatBoost artifact yükleme hatası: %s", exc)
            raise

    # -----------------------------------------------------------------------
    # Tahmin
    # -----------------------------------------------------------------------

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ham DataFrame alır; churn_proba ve predicted_churn sütunlarını
        eklenmiş DataFrame döner.

        train_medians preprocessing'e iletilerek HighRiskProfile /
        ShortTenure_HighCharge eşiklerinde veri sızıntısı engellenir.
        """
        X_ready = prepare_cat_input(
            raw_df=raw_df,
            train_medians=self.train_medians,
            feature_columns=self.feature_columns,
            drop_cols=self.drop_cols,
        )

        churn_proba     = self.model.predict_proba(X_ready)[:, 1]
        predicted_churn = (churn_proba >= self.threshold).astype(int)

        result = X_ready.copy()
        result["model_name"]      = "Champion (CatBoost)"
        result["churn_proba"]     = churn_proba
        result["predicted_churn"] = predicted_churn
        result["threshold_used"]  = self.threshold

        logger.info(
            "CatBoost tahmini tamamlandı: %d müşteri, "
            "ortalama churn_proba=%.3f, churn tahmin sayısı=%d.",
            len(result),
            float(churn_proba.mean()),
            int(predicted_churn.sum()),
        )
        return result

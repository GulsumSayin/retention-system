"""
model_router.py
Model seçim ve yönlendirme katmanı.

Mimari Notlar:
  Champion (XGBoost) varsayılan modeldir. CatBoost servisi Challenger olarak
  yedekte tutulur; predict_both() ile Champion/Challenger karşılaştırması
  yapılabilir.

  SOLID-D (Dependency Inversion):
    ModelRouter doğrudan InferenceService sınıflarını örneklemek yerine,
    constructor'a enjekte edilen servisleri kullanır. Bu yaklaşım:
      - Birim testlerde mock servis geçmeyi mümkün kılar.
      - Yeni model servislerini mevcut koda dokunmadan ekler.
      - app.py'de @st.cache_resource ile beraber kullanımı kolaylaştırır.
"""

import logging

import pandas as pd

from inference_cat import CatBoostInferenceService
from inference_xgb import XGBInferenceService

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Tek sorumluluk: predict() çağrısını doğru InferenceService'e yönlendirmek.

    Parametreler
    ------------
    cat_service : CatBoostInferenceService (None ise varsayılan olarak oluşturulur)
    xgb_service : XGBInferenceService      (None ise varsayılan olarak oluşturulur)
    """

    def __init__(
        self,
        cat_service: CatBoostInferenceService | None = None,
        xgb_service: XGBInferenceService | None = None,
    ) -> None:
        self.cat_service = cat_service or CatBoostInferenceService()
        self.xgb_service = xgb_service or XGBInferenceService()
        logger.info("ModelRouter hazır (Champion: XGBoost, Challenger: CatBoost).")

    # -----------------------------------------------------------------------
    # Tek model tahmini
    # -----------------------------------------------------------------------

    def predict(self, raw_df: pd.DataFrame, model: str = "xgboost") -> pd.DataFrame:
        """
        Ham müşteri verisini belirtilen modele yönlendirir.

        Parametreler
        ------------
        raw_df : Ham DataFrame (IBM Telco formatı)
        model  : "xgboost" (varsayılan, Champion) | "catboost" (Challenger)

        Döndürür
        --------
        pd.DataFrame : churn_proba ve predicted_churn eklenmiş DataFrame
        """
        if model == "catboost":
            logger.info("ModelRouter → CatBoost (Challenger), %d satır", len(raw_df))
            return self.cat_service.predict(raw_df)

        logger.info("ModelRouter → XGBoost (Champion), %d satır", len(raw_df))
        return self.xgb_service.predict(raw_df)

    # -----------------------------------------------------------------------
    # Champion / Challenger karşılaştırması
    # -----------------------------------------------------------------------

    def predict_both(self, raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Her iki modelden tahmin üretir.

        Kullanım: tez değerlendirme sekansında Champion/Challenger kıyası.

        Döndürür
        --------
        {"catboost": df_cat, "xgboost": df_xgb}
        """
        logger.info(
            "ModelRouter.predict_both: %d satır üzerinde Champion/Challenger karşılaştırması.",
            len(raw_df),
        )
        return {
            "catboost": self.cat_service.predict(raw_df),
            "xgboost":  self.xgb_service.predict(raw_df),
        }

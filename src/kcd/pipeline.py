"""
KCD 예측 파이프라인

전체 흐름:
1. 사고내용 텍스트 입력
2. NER 모델로 Feature 추출 (BODY, SIDE, DIS_MAIN, SYMPTOM, CAUSE, TIME, TEST, TREATMENT, ACT, NEG, DIS_HIST)
3. 메타 정보 결합 (나이, 성별, 접수경로, 진료과목, EDI)
4. KCD 예측 모델로 최종 코드 예측

사용법:
    pipeline = KCDPredictionPipeline(
        ner_model_path="./ner_output",
        kcd_model_path="./kcd_output"
    )

    result = pipeline.predict(
        text="환자가 좌측 무릎 골절로 수술을 받았습니다.",
        age=45,
        gender="M",
        department="정형외과"
    )
"""

from dataclasses import dataclass
from typing import Optional

from src.ner.model import NERModel, load_model as load_ner_model
from src.kcd.model import KCDPredictionModel, load_model as load_kcd_model
from src.kcd.data_format import NERFeatures, MetaFeatures
from src.kcd.kcd_dictionary import get_kcd_dictionary, KCDCode


@dataclass
class PredictionResult:
    """예측 결과"""
    text: str                                # 원본 텍스트
    ner_features: NERFeatures               # NER 추출 결과
    meta_features: MetaFeatures             # 메타 정보
    predictions: list[dict]                 # KCD 예측 결과 리스트
    top_prediction: dict                    # 최상위 예측

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "ner_features": self.ner_features.to_dict(),
            "meta_features": self.meta_features.to_dict(),
            "predictions": self.predictions,
            "top_prediction": self.top_prediction,
        }

    def __str__(self) -> str:
        lines = [
            f"입력: {self.text}",
            f"NER: {self.ner_features.to_text()}",
            f"메타: {self.meta_features.to_text()}",
            f"예측: {self.top_prediction['code']} - {self.top_prediction['name']} "
            f"(확률: {self.top_prediction['score']:.2%})",
        ]
        return "\n".join(lines)


class KCDPredictionPipeline:
    """
    KCD 예측 파이프라인

    NER 모델 → Feature 추출 → KCD 예측 모델 → 최종 KCD 코드
    """

    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        kcd_model_path: Optional[str] = None,
        ner_model: Optional[NERModel] = None,
        kcd_model: Optional[KCDPredictionModel] = None,
    ):
        """
        Args:
            ner_model_path: NER 모델 경로
            kcd_model_path: KCD 예측 모델 경로
            ner_model: NER 모델 인스턴스 (직접 전달 시)
            kcd_model: KCD 예측 모델 인스턴스 (직접 전달 시)
        """
        # NER 모델 로드
        if ner_model:
            self.ner_model = ner_model
        elif ner_model_path:
            print(f"NER 모델 로드: {ner_model_path}")
            self.ner_model = load_ner_model(ner_model_path)
        else:
            self.ner_model = None
            print("경고: NER 모델이 없습니다. NER Feature를 직접 제공해야 합니다.")

        # KCD 예측 모델 로드
        if kcd_model:
            self.kcd_model = kcd_model
        elif kcd_model_path:
            print(f"KCD 모델 로드: {kcd_model_path}")
            self.kcd_model = load_kcd_model(kcd_model_path)
        else:
            raise ValueError("KCD 모델 경로 또는 인스턴스가 필요합니다.")

        self.kcd_dict = get_kcd_dictionary()

    def extract_ner_features(self, text: str) -> NERFeatures:
        """
        텍스트에서 NER Feature 추출

        Args:
            text: 입력 텍스트

        Returns:
            NERFeatures 객체
        """
        if self.ner_model is None:
            return NERFeatures()

        # NER 모델로 Feature 추출
        features_dict = self.ner_model.extract_features(text)

        return NERFeatures(
            body=features_dict.get("BODY", []),
            side=features_dict.get("SIDE", []),
            dis_main=features_dict.get("DIS_MAIN", []),
            symptom=features_dict.get("SYMPTOM", []),
            cause=features_dict.get("CAUSE", []),
            time=features_dict.get("TIME", []),
            test=features_dict.get("TEST", []),
            treatment=features_dict.get("TREATMENT", []),
            act=features_dict.get("ACT", []),
            neg=features_dict.get("NEG", []),
            dis_hist=features_dict.get("DIS_HIST", []),
        )

    def predict(
        self,
        text: str,
        ner_features: Optional[NERFeatures] = None,
        age: Optional[int] = None,
        gender: str = "U",
        reception_route: str = "기타",
        department: str = "",
        edi_code: str = "",
        has_edi: bool = False,
        top_k: int = 3,
    ) -> PredictionResult:
        """
        KCD 코드 예측

        Args:
            text: 사고내용 텍스트
            ner_features: NER Feature (None이면 자동 추출)
            age: 나이
            gender: 성별 (M/F/U)
            reception_route: 접수경로
            department: 진료과목
            edi_code: EDI 코드
            has_edi: EDI 존재 여부
            top_k: 상위 K개 예측 반환

        Returns:
            PredictionResult 객체
        """
        # NER Feature 추출 (제공되지 않은 경우)
        if ner_features is None:
            ner_features = self.extract_ner_features(text)

        # 메타 정보 생성
        meta_features = MetaFeatures(
            age=age,
            gender=gender,
            reception_route=reception_route,
            department=department,
            edi_code=edi_code,
            has_edi=has_edi,
        )

        # KCD 예측
        predictions = self.kcd_model.predict(
            text=text,
            ner_features=ner_features,
            meta_features=meta_features,
            top_k=top_k,
        )

        return PredictionResult(
            text=text,
            ner_features=ner_features,
            meta_features=meta_features,
            predictions=predictions,
            top_prediction=predictions[0] if predictions else {},
        )

    def predict_batch(
        self,
        texts: list[str],
        meta_list: Optional[list[dict]] = None,
        top_k: int = 3,
    ) -> list[PredictionResult]:
        """
        배치 예측

        Args:
            texts: 텍스트 리스트
            meta_list: 메타 정보 리스트 (선택)
            top_k: 상위 K개 예측 반환

        Returns:
            PredictionResult 리스트
        """
        results = []
        meta_list = meta_list or [{}] * len(texts)

        for text, meta in zip(texts, meta_list):
            result = self.predict(
                text=text,
                age=meta.get("age"),
                gender=meta.get("gender", "U"),
                reception_route=meta.get("reception_route", "기타"),
                department=meta.get("department", ""),
                edi_code=meta.get("edi_code", ""),
                has_edi=meta.get("has_edi", False),
                top_k=top_k,
            )
            results.append(result)

        return results

    def get_code_info(self, code: str) -> Optional[KCDCode]:
        """KCD 코드 정보 조회"""
        return self.kcd_dict.get_code(code)


def create_pipeline(
    ner_model_path: str = "./ner_output",
    kcd_model_path: str = "./kcd_output",
) -> KCDPredictionPipeline:
    """파이프라인 생성 (편의 함수)"""
    return KCDPredictionPipeline(
        ner_model_path=ner_model_path,
        kcd_model_path=kcd_model_path,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("KCD 예측 파이프라인 테스트")
    print("=" * 60)
    print("\n이 스크립트는 학습된 모델이 필요합니다.")
    print("\n사용 예시:")
    print("""
    from src.kcd.pipeline import KCDPredictionPipeline

    # 파이프라인 생성
    pipeline = KCDPredictionPipeline(
        ner_model_path="./ner_output",
        kcd_model_path="./kcd_output"
    )

    # 예측
    result = pipeline.predict(
        text="환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하였습니다.",
        age=45,
        gender="M",
        department="정형외과"
    )

    print(result)
    print(f"예측 KCD: {result.top_prediction['code']}")
    """)

from dataclasses import dataclass
from typing import TypeVar

from aion.codecs.catalog import CatalogCodec
from aion.codecs.image import ImageCodec
from aion.codecs.scalar import (
    GridScalarCodec,
    LogScalarCodec,
    MultiScalarCodec,
    ScalarCodec,
)
from aion.codecs.scalar_field import ScalarFieldCodec
from aion.codecs.spectrum import SpectrumCodec
from aion.modalities import (
    HSCAG,
    HSCAI,
    HSCAR,
    HSCAY,
    HSCAZ,
    Dec,
    DESISpectrum,
    GaiaFluxBp,
    GaiaFluxG,
    GaiaFluxRp,
    GaiaParallax,
    GaiaXpBp,
    GaiaXpRp,
    HSCImage,
    HSCMagG,
    HSCMagI,
    HSCMagR,
    HSCMagY,
    HSCMagZ,
    HSCShape11,
    HSCShape12,
    HSCShape22,
    Image,
    LegacySurveyCatalog,
    LegacySurveyEBV,
    LegacySurveyFluxG,
    LegacySurveyFluxI,
    LegacySurveyFluxR,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyFluxZ,
    LegacySurveyImage,
    LegacySurveySegmentationMap,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyShapeR,
    Ra,
    SDSSSpectrum,
    Spectrum,
    Z,
)

CodecType = TypeVar(
    "CodecModel",
    bound=type[
        CatalogCodec
        | GridScalarCodec
        | ImageCodec
        | LogScalarCodec
        | MultiScalarCodec
        | ScalarCodec
        | ScalarFieldCodec
        | SpectrumCodec
    ],
)


@dataclass
class CodecHFConfig:
    """Codec configuration for AION."""

    codec_class: CodecType
    repo_id: str


MODALITY_CODEC_MAPPING = {
    Dec: ScalarCodec,
    DESISpectrum: SpectrumCodec,
    GaiaFluxBp: LogScalarCodec,
    GaiaFluxG: LogScalarCodec,
    GaiaFluxRp: LogScalarCodec,
    GaiaParallax: LogScalarCodec,
    GaiaXpBp: MultiScalarCodec,
    GaiaXpRp: MultiScalarCodec,
    HSCAG: ScalarCodec,
    HSCAI: ScalarCodec,
    HSCAR: ScalarCodec,
    HSCAY: ScalarCodec,
    HSCAZ: ScalarCodec,
    HSCImage: ImageCodec,
    HSCMagG: ScalarCodec,
    HSCMagI: ScalarCodec,
    HSCMagR: ScalarCodec,
    HSCMagY: ScalarCodec,
    HSCMagZ: ScalarCodec,
    HSCShape11: ScalarCodec,
    HSCShape12: ScalarCodec,
    HSCShape22: ScalarCodec,
    Image: ImageCodec,
    LegacySurveyCatalog: CatalogCodec,
    LegacySurveyEBV: ScalarCodec,
    LegacySurveyFluxG: LogScalarCodec,
    LegacySurveyFluxI: LogScalarCodec,
    LegacySurveyFluxR: LogScalarCodec,
    LegacySurveyFluxW1: LogScalarCodec,
    LegacySurveyFluxW2: LogScalarCodec,
    LegacySurveyFluxW3: LogScalarCodec,
    LegacySurveyFluxW4: LogScalarCodec,
    LegacySurveyFluxZ: LogScalarCodec,
    LegacySurveyImage: ImageCodec,
    LegacySurveySegmentationMap: ScalarFieldCodec,
    LegacySurveyShapeE1: ScalarCodec,
    LegacySurveyShapeE2: ScalarCodec,
    LegacySurveyShapeR: LogScalarCodec,
    Ra: ScalarCodec,
    SDSSSpectrum: SpectrumCodec,
    Spectrum: SpectrumCodec,
    Z: GridScalarCodec,
}

HF_REPO_ID = "polymathic-ai/aion-base"

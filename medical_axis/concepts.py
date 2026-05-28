from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptSide:
    """One side of a contrastive medical concept axis."""

    name: str
    label: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    require_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConceptAxis:
    """A binary concept contrast used to construct a residual direction."""

    axis_id: str
    positive: ConceptSide
    negative: ConceptSide
    description: str


DEFAULT_AXES: tuple[ConceptAxis, ...] = (
    ConceptAxis(
        axis_id="diabetes_subtype",
        positive=ConceptSide(
            name="type1",
            label="Type 1 diabetes",
            include=(r"\btype (1|i) diabetes",),
            exclude=(r"\btype 2 diabetes", r"\btype ii diabetes"),
        ),
        negative=ConceptSide(
            name="type2",
            label="Type 2 diabetes",
            include=(r"\btype (2|ii) diabetes",),
            exclude=(r"\btype 1 diabetes", r"\btype i diabetes"),
        ),
        description="Diabetes subtype: Type 1 versus Type 2 diagnosis.",
    ),
    ConceptAxis(
        axis_id="complication_status",
        positive=ConceptSide(
            name="complicated",
            label="Diabetes with complications",
            include=(r"\bdiabetes",),
            exclude=(r"\bwithout complications", r"\bwithout complication"),
            require_any=(
                r"\bcomplication",
                r"\bneuropathy",
                r"\bnephropathy",
                r"\bretinopathy",
                r"\bangiopathy",
            ),
        ),
        negative=ConceptSide(
            name="uncomplicated",
            label="Diabetes without complications",
            include=(r"\bdiabetes", r"\bwithout complications", r"\bwithout complication"),
            require_any=(r"\btype 1 diabetes", r"\btype 2 diabetes", r"\bdiabetes mellitus"),
        ),
        description="Diabetes complication status: complicated versus uncomplicated coding.",
    ),
    ConceptAxis(
        axis_id="neoplasm_behavior",
        positive=ConceptSide(
            name="malignant",
            label="Malignant neoplasm",
            include=(r"\bmalignant", r"\bneoplasm"),
            exclude=(r"\bbenign",),
        ),
        negative=ConceptSide(
            name="benign",
            label="Benign neoplasm",
            include=(r"\bbenign", r"\bneoplasm"),
            exclude=(r"\bmalignant",),
        ),
        description="Neoplasm behavior: malignant versus benign.",
    ),
    ConceptAxis(
        axis_id="infectious_etiology",
        positive=ConceptSide(
            name="bacterial",
            label="Bacterial infection",
            include=(r"\bbacterial",),
            exclude=(r"\bviral",),
        ),
        negative=ConceptSide(
            name="viral",
            label="Viral infection",
            include=(r"\bviral",),
            exclude=(r"\bbacterial",),
        ),
        description="Infectious etiology: bacterial versus viral.",
    ),
    ConceptAxis(
        axis_id="disease_course",
        positive=ConceptSide(
            name="acute",
            label="Acute condition",
            include=(r"\bacute",),
            exclude=(r"\bchronic",),
        ),
        negative=ConceptSide(
            name="chronic",
            label="Chronic condition",
            include=(r"\bchronic",),
            exclude=(r"\bacute",),
        ),
        description="Disease course: acute versus chronic.",
    ),
)


PROMPT_TEMPLATES: tuple[str, ...] = (
    "Clinical coding note: ICD {icd_code} is described as {icd_description}. The contrastive medical concept is",
    "Medical ontology entry: {icd_description} ({icd_code}). The best concept label is",
    "A clinician reviews the diagnosis '{icd_description}' with ICD code {icd_code}. This diagnosis belongs to the concept",
    "In a diagnostic terminology table, {icd_code} means {icd_description}. The relevant concept category is",
    "Coding audit: diagnosis={icd_code}; description={icd_description}. Assign the medical concept label:",
    "EHR problem list item: {icd_description}. The high-level concept represented by this item is",
    "For ICD concept {icd_code}, '{icd_description}', the contrastive label should be",
    "The diagnosis text is '{icd_description}'. In this experiment, its medical concept side is",
    "Medical concept extraction task. Input diagnosis: {icd_description}. Output concept:",
    "Given the ICD description {icd_description} and code {icd_code}, classify the concept as",
)

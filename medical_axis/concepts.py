from __future__ import annotations

from dataclasses import dataclass


CCS_SOURCE = "appendix_a_single_dx"
ICD_DERIVED_SOURCE = "icd_description_regex"


@dataclass(frozen=True)
class ConceptSide:
    """One side of a contrastive medical concept axis."""

    name: str
    label: str
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    require_any: tuple[str, ...] = ()
    ccs_codes: tuple[str, ...] = ()
    exclude_ccs_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConceptAxis:
    """A binary concept contrast used to construct a residual direction."""

    axis_id: str
    positive: ConceptSide
    negative: ConceptSide
    description: str
    axis_family: str = "lexical"
    primary_axis: bool = False
    ccs_source: str = ""
    min_side_rows: int | None = None


PRIMARY_CCS_AXES: tuple[ConceptAxis, ...] = (
    ConceptAxis(
        axis_id="diabetes_complication_status_ccs",
        positive=ConceptSide(
            name="with_complications",
            label="Diabetes mellitus with complications",
            ccs_codes=("50",),
            exclude_ccs_codes=("186",),
        ),
        negative=ConceptSide(
            name="without_complication",
            label="Diabetes mellitus without complication",
            ccs_codes=("49",),
            exclude_ccs_codes=("186",),
        ),
        description="CCS diabetes complication status: CCS 50 versus CCS 49, excluding pregnancy-related diabetes CCS 186.",
        axis_family="diabetes_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
        min_side_rows=10,
    ),
    ConceptAxis(
        axis_id="infectious_etiology_ccs",
        positive=ConceptSide(
            name="bacterial_unspecified_site",
            label="Bacterial infection; unspecified site",
            ccs_codes=("3",),
        ),
        negative=ConceptSide(
            name="viral",
            label="Viral infection",
            ccs_codes=("7",),
        ),
        description="CCS infectious etiology contrast: bacterial infection versus viral infection.",
        axis_family="infectious_disease_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="eye_inflammation_vs_other_eye_ccs",
        positive=ConceptSide(
            name="eye_inflammation_infection",
            label="Inflammation or infection of eye",
            ccs_codes=("90",),
        ),
        negative=ConceptSide(
            name="other_eye_disorders",
            label="Other eye disorders",
            ccs_codes=("91",),
        ),
        description="CCS eye disorder contrast: inflammatory or infectious eye disease versus other eye disorders.",
        axis_family="eye_disorders_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="musculoskeletal_infective_vs_nontraumatic_joint_ccs",
        positive=ConceptSide(
            name="infective_arthritis_osteomyelitis",
            label="Infective arthritis and osteomyelitis",
            ccs_codes=("201",),
        ),
        negative=ConceptSide(
            name="other_nontraumatic_joint",
            label="Other non-traumatic joint disorders",
            ccs_codes=("204",),
        ),
        description="CCS musculoskeletal contrast: infective arthritis/osteomyelitis versus other non-traumatic joint disorders.",
        axis_family="musculoskeletal_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="intracranial_injury_vs_upper_limb_fracture_ccs",
        positive=ConceptSide(
            name="intracranial_injury",
            label="Intracranial injury",
            ccs_codes=("233",),
        ),
        negative=ConceptSide(
            name="upper_limb_fracture",
            label="Fracture of upper limb",
            ccs_codes=("229",),
        ),
        description="CCS trauma contrast: intracranial injury versus upper limb fracture.",
        axis_family="injury_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="pregnancy_complications_vs_birth_puerperium_ccs",
        positive=ConceptSide(
            name="other_pregnancy_complications",
            label="Other complications of pregnancy",
            ccs_codes=("181",),
            exclude_ccs_codes=("186",),
        ),
        negative=ConceptSide(
            name="birth_puerperium_complications",
            label="Other complications of birth and puerperium affecting management of mother",
            ccs_codes=("195",),
            exclude_ccs_codes=("186",),
        ),
        description="CCS obstetric contrast: other pregnancy complications versus birth/puerperium complications, excluding diabetes in pregnancy CCS 186.",
        axis_family="pregnancy_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="drug_poisoning_vs_nondrug_poisoning_ccs",
        positive=ConceptSide(
            name="drug_poisoning",
            label="Poisoning by other medications and drugs",
            ccs_codes=("242",),
        ),
        negative=ConceptSide(
            name="nondrug_poisoning",
            label="Poisoning by nonmedicinal substances",
            ccs_codes=("243",),
        ),
        description="CCS poisoning contrast: medication/drug poisoning versus nonmedicinal substance poisoning.",
        axis_family="injury_poisoning_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="congenital_cardiac_vs_other_congenital_ccs",
        positive=ConceptSide(
            name="cardiac_circulatory_congenital",
            label="Cardiac and circulatory congenital anomalies",
            ccs_codes=("213",),
        ),
        negative=ConceptSide(
            name="other_congenital",
            label="Other congenital anomalies",
            ccs_codes=("217",),
        ),
        description="CCS congenital anomaly contrast: cardiac/circulatory congenital anomalies versus other congenital anomalies.",
        axis_family="congenital_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="open_wounds_head_trunk_vs_extremity_ccs",
        positive=ConceptSide(
            name="open_wounds_head_neck_trunk",
            label="Open wounds of head, neck, and trunk",
            ccs_codes=("235",),
        ),
        negative=ConceptSide(
            name="open_wounds_extremities",
            label="Open wounds of extremities",
            ccs_codes=("236",),
        ),
        description="CCS injury contrast: open wounds of head/neck/trunk versus open wounds of extremities.",
        axis_family="injury_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
    ConceptAxis(
        axis_id="joint_dislocation_vs_sprain_ccs",
        positive=ConceptSide(
            name="joint_disorders_dislocations_trauma",
            label="Joint disorders and dislocations; trauma-related",
            ccs_codes=("225",),
        ),
        negative=ConceptSide(
            name="sprains_strains",
            label="Sprains and strains",
            ccs_codes=("232",),
        ),
        description="CCS injury contrast: trauma-related joint disorders/dislocations versus sprains and strains.",
        axis_family="injury_ccs",
        primary_axis=True,
        ccs_source=CCS_SOURCE,
    ),
)


EXPLORATORY_ICD_DERIVED_AXES: tuple[ConceptAxis, ...] = (
    ConceptAxis(
        axis_id="exploratory_icd_diabetes_subtype",
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
        description="Exploratory ICD-description-derived diabetes subtype contrast; not a CCS primary axis.",
        axis_family="diabetes_icd_derived",
        primary_axis=False,
        ccs_source=ICD_DERIVED_SOURCE,
    ),
    ConceptAxis(
        axis_id="exploratory_icd_neoplasm_behavior",
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
        description="Exploratory ICD-description-derived neoplasm behavior contrast.",
        axis_family="neoplasm_icd_derived",
        primary_axis=False,
        ccs_source=ICD_DERIVED_SOURCE,
    ),
    ConceptAxis(
        axis_id="exploratory_icd_disease_course",
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
        description="Exploratory ICD-description-derived disease course contrast.",
        axis_family="course_icd_derived",
        primary_axis=False,
        ccs_source=ICD_DERIVED_SOURCE,
    ),
)


DEFAULT_AXES: tuple[ConceptAxis, ...] = PRIMARY_CCS_AXES + EXPLORATORY_ICD_DERIVED_AXES


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

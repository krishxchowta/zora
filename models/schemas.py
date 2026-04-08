from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional
from datetime import datetime


class RunCreateRequest(BaseModel):
    problem_desc: Optional[str] = None
    target_column: Optional[str] = None
    enable_protein_analysis: bool = False
    protein_context_json: Optional[dict] = None


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    filename: str


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    status: str
    rows_count: Optional[int] = None
    cols_count: Optional[int] = None
    schema_json: Optional[dict] = None
    embedding_count: Optional[int] = None
    protein_context_json: Optional[dict] = None
    protein_summary_json: Optional[dict] = None
    feature_summary: Optional[dict] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentSSEEvent(BaseModel):
    type: str                    # "agent_update" | "pipeline_complete" | "error"
    agent: str
    status: str                  # "running" | "completed" | "failed"
    latency_ms: Optional[int] = None
    output_summary: Optional[str] = None
    error_message: Optional[str] = None
    data: Optional[dict] = None
    timestamp: str


class SchemaProfile(BaseModel):
    run_id: str
    filename: str
    rows: int
    cols: int
    columns: list[dict]          # [{name, dtype, null_count, null_pct, sample_values}]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    target_candidate: Optional[str]
    null_summary: dict           # {col: pct}
    duplicate_count: int
    memory_mb: float


class ProteinContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gene_symbol: Optional[str] = None
    protein_name: Optional[str] = None
    uniprot_id: Optional[str] = None
    variant_hgvs: Optional[str] = None
    disease_label: Optional[str] = None
    notes: Optional[str] = None


class MisfoldSummary(BaseModel):
    enabled: bool = False
    protein_name: Optional[str] = None
    uniprot_id: Optional[str] = None
    variant_hgvs: Optional[str] = None
    aggregation_propensity: Optional[float] = None
    stuck_score: Optional[float] = None
    energy_state: Optional[str] = None
    surface_exposure_score: Optional[float] = None
    disorder_score: Optional[float] = None
    variant_delta_score: Optional[float] = None
    residue_graph_risk: Optional[float] = None
    evidence: list[dict] = Field(default_factory=list)
    red_flags: list[dict] = Field(default_factory=list)
    viewer_stub: dict = Field(default_factory=dict)


class ProteinFoldingReport(BaseModel):
    """
    Structured output of alphafold_tool() — real biophysics from AlphaFold EBI
    API + UniProt REST API + BioPython ProtParam.
    Callers can validate the raw dict with ProteinFoldingReport.model_validate(result).
    """
    model_config = ConfigDict(protected_namespaces=())

    protein_name: str
    uniprot_id: str
    stability_score: float              # mean_pLDDT/100 or biopython-derived [0,1]
    mean_plddt: Optional[float] = None  # raw AlphaFold pLDDT score 0-100
    confidence: str                     # "high" | "medium" | "low"
    instability_index: Optional[float] = None   # Guruprasad 1990: <40 = stable
    isoelectric_point: Optional[float] = None   # pI
    molecular_weight: Optional[float] = None    # Daltons
    gravy_score: Optional[float] = None         # GRAVY hydropathy index
    aromaticity: Optional[float] = None         # fraction of aromatic residues
    secondary_structure: Optional[dict] = None  # {"helix": f, "turn": f, "sheet": f}
    sequence_length: int = 0
    sequence_source: str = "hardcoded_fallback"     # "uniprot_api" | "hardcoded_fallback"
    stability_source: str = "biopython_derived"     # "alphafold_api" | "biopython_derived"
    pdb_link: str = ""
    protein_function: Optional[str] = None          # from UniProt FUNCTION comment
    disease_associations: list[str] = Field(default_factory=list)  # from UniProt


class CleanReport(BaseModel):
    run_id: str
    rows_before: int
    rows_after: int
    dupes_removed: int
    same_visit_dupes_removed: int = 0
    nulls_imputed: dict           # {col: count_imputed}
    outliers_removed: dict        # legacy field kept for backward compatibility
    imputation_strategy: dict     # {col: "median"|"mode"|"skipped"}
    invalid_values_converted: dict = Field(default_factory=dict)
    capped_extremes: dict = Field(default_factory=dict)
    missingness_flags_added: list[str] = Field(default_factory=list)
    quality_score: Optional[int] = None
    critic_feedback: Optional[str] = None
    passed_critic: bool = False


class FeatureEngineeringReport(BaseModel):
    run_id: str
    source_rows: int
    feature_rows: int
    source_columns: int
    feature_columns: int
    derived_features_added: list[str] = Field(default_factory=list)
    dropped_from_model_columns: list[str] = Field(default_factory=list)
    rare_category_buckets: dict = Field(default_factory=dict)
    output_file: str = "featured.csv"


class PatientContactCreateRequest(BaseModel):
    run_id: str
    patient_name: str
    phone_e164: Optional[str] = None
    whatsapp_e164: Optional[str] = None
    preferred_channel: Literal["sms", "whatsapp"] = "whatsapp"
    request_message: Optional[str] = None


class PrescriptionUpsertRequest(BaseModel):
    doctor_name: str
    prescription_text: str
    notes: Optional[str] = None


class ReportRequestNotifyRequest(BaseModel):
    doctor_name: Optional[str] = None


class ReportApprovalRequest(BaseModel):
    doctor_name: str
    prescription_text: str
    notes: Optional[str] = None
    send_channel: Literal["preferred", "sms", "whatsapp"] = "preferred"


class ReportRejectRequest(BaseModel):
    doctor_name: str
    reason: str


class MessageSendRequest(BaseModel):
    doctor_name: str
    notes: Optional[str] = None


class BoardCaseSummary(BaseModel):
    run_id: str
    filename: str
    pipeline_status: str
    request_status: Optional[str] = None
    message_status: Optional[str] = None
    patient_name: Optional[str] = None
    doctor_review: bool = False
    doctor_report_ready: bool = False
    patient_report_ready: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class BoardCaseDetail(BaseModel):
    run: dict
    insight: Optional[dict] = None
    patient_contacts: list[dict] = Field(default_factory=list)
    report_requests: list[dict] = Field(default_factory=list)
    prescriptions: list[dict] = Field(default_factory=list)
    message_deliveries: list[dict] = Field(default_factory=list)
    doctor_report_text: Optional[str] = None
    patient_report_text: Optional[str] = None
    final_prescription_text: Optional[str] = None


class MessageDeliveryResult(BaseModel):
    ok: bool
    channel: Literal["sms", "whatsapp"]
    delivery_status: str
    provider_message_id: Optional[str] = None
    error_text: Optional[str] = None


class OpsReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: str
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OpsReadinessReport(BaseModel):
    overall_ready: bool
    database_ready: bool
    sms_ready: bool
    whatsapp_ready: bool
    doctor_notification_ready: bool
    board_delivery_ready: bool
    migration_bundle_path: str
    checks: list[OpsReadinessCheck] = Field(default_factory=list)
    required_manual_steps: list[str] = Field(default_factory=list)

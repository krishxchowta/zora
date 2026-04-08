"""
AlphaFold Tool — real protein biophysics via 3-layer pipeline:

  Layer 1  AlphaFold EBI Database API (free, no auth)
           GET https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}
           → meanPlddt → stability_score = plddt / 100
           Actual pLDDT confidence score (0-100, >70 = high confidence)

  Layer 2  UniProt REST API (free, no auth)
           GET https://rest.uniprot.org/uniprotkb/{uniprot_id}.json
           → real full-length sequence, protein function, disease associations

  Layer 3  BioPython ProtParam (local, no network, always runs)
           → instability_index, isoelectric_point, molecular_weight,
             gravy_score, aromaticity, secondary_structure_fraction

Fallback chain (silent, never raises):
  AlphaFold API down  → stability derived from instability_index
  UniProt API down    → use hardcoded PROTEIN_SEQUENCES
  BioPython fails     → biophysical fields set to None

Public function signature UNCHANGED:
  alphafold_tool(protein_name: str, uniprot_id: str) -> dict
All original 7 keys still present; 12 new keys added.
"""

import logging
import requests

log = logging.getLogger(__name__)

# ── API endpoints ──────────────────────────────────────────────────────────────
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
UNIPROT_API_URL   = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
REQUEST_TIMEOUT   = 8  # seconds per API call

# ── Fallback sequences (used only when UniProt API is unavailable) ────────────
PROTEIN_SEQUENCES: dict[str, str] = {
    "BNP":      "SPKMVQGSGCFGRKMDRISSSSGLGCKVLRRHSPKMVQGSGCFGRKMDRISSSSGLGCKVLRRH",
    "SERPINA1": (
        "EDPQGDAAQKTDTSHHDQDHPTFNKITPNLAEFAFSLYRQLAHQSNSTNIFFSPVSIATAFAMLSLGT"
        "KADTHDEILEGLNFNLTEIPEAQIHEGFQELLRTLNQPDSQLQLTTGNGLFLSEGLKLVDKFLEDVKKL"
        "YHSEAFTVNFGDTEEAKKQINDYVEKGTQGKIVDLVKELDRDTVFALVNYIFFKGQWIKFKGSPEDLPV"
        "AHVQGLNFGGHTIQSEQLPFKHSYFVSMQDQTTPLPQHLEQLKVFESLKSYLYPPEQKLISEEDL"
    ),
    "GCK":      (
        "MLDDRARMEAAKYTDPQFNSSMELQGLGRSQKKELYRQIFQKLGFSDENLWPAVLNDQGEDGDGADGEA"
        "MLASGRSIFNFSSLKQLGEKAHQNPEGRLRVSLPGFDDLRQESREHAKRQIHSIVDEETMEEFQAQRKK"
        "KELLDQQFEENFNRSEGNIYLNVAAGQERQVDPKTPFPVSRSQLHSFLQQKPQPKNIQQQLQNPHYHHAI"
        "AQAIEDFQKQHQGIHSRKKNIGRQISAHIQPVLQRTDTPVPQGAEKHSKHLSKGPQNISLKQLPRNTPK"
        "PFQHIFQPLFQDLKPFQEVFETALQGAQALWIQKRPQKGQSTLAHGTQAKELEQFEEPLKEFKQHLIP"
    ),
    "PLAT":     (
        "MDAMKRGLCCVLLLCGAVFVSPSQEIHARFRRGARSYQVICRDEKTQMIYQQHQSWLRPVLRSNRVEYC"
        "WCNSGRAQCHSVPVKSCSEPRCFNGGTCQQALYFSDFVCQCPEGFAGKCCEIDTRPC"
    ),
    "ACE":      (
        "MGAASGRRGPGLLLPLPLLLLLPPQESALGGGANETRQPVVELRPGETLHISMQPRTFLHYSGLEALRPA"
        "PQHLIRVEGSQLAQDALRVKMASVDPHQGPGDTLMESVHQSPNTQDLQTLRVDFSAEEGVYRVLNSSLL"
    ),
    "DEFB1":    "GNFLTGLGHRSDHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCK",
    "RANKL":    (
        "MMMQNSRSEVHHQITEYYQIVSQHQSQNKSILNELQKDLGRQEGFPGCVSGFIMQYSEKPTLCGLNQRI"
        "RQMVHIQNVYHYSSPKYSRSPEILTLQLPVSKNLHEIKHKK"
    ),
    "IL13":     "PGPVPPSTALRELLQEATGYVPLKEMKESFNVTAHVDKMLNEIHDKLNHLQLLRKQFHLDHALQFPCHKPCQYIK",
    "TLR4":     "MSSRNFLLMLFALCLFQAQGKPAPALCPQDLEFQKTYQVQPGQRAVSQGGKGLRFSCIFPEKFLNKRKKKLQFEYKNRMM",
    "ABCG8":    "MPCTADGQESKEAKALAQERRAAQKEKEKQEKDDDDEDEGKFKKKTKKKHEKHD",
    "SLC34A1":  "MEALVTEVSAGRGRRGPPEEEMEETPKGVAGEAAVREGGLGMAPK",
    "CRP":      (
        "MERWLLLALLGFAELCSGSTYKIRPLCQTLKPNMVHHEISFQDLTPMLKAFLVDMPYSLDQTIRQMSNI"
        "FYLWLDSSSGNQSRQGSGLAELFLDTEGRSKLNTYMGPNRQNAASVICDLNQPGGRDQGLRTYLSRGHEQ"
        "IMRAYVDGFSRESGVSYQTAIIDLHSQVNQTLPQTNAQIFNRLNQHQVEAQNLMQNIGAKLVQELTQSNN"
        "QTLYSQLANQLDREQVSMLQQLNLQ"
    ),
    # New proteins for expanded 30-disease map
    "EGFR":     (
        "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQ"
        "RNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQE"
    ),
    "BRCA1":    (
        "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKR"
        "SLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNACKQCTSKK"
    ),
    "APC":      (
        "MAAASYDQYLSESEEDKDNESDDESEGKKKTFKDAPGIPSTSQLSASGQYNLNSVPMDLHSTATGLNQNN"
        "SSPPDKKEIPVDSSFPSGSHSSHSTPGSEDINLQNLGQNLSSQQPEHESLDLSQRQGLMSAGQNPQHRPQ"
    ),
    "KLK3":     (
        "MWVPVVFLTLSVTWIGAAPLILSRIVGGWECEKHSQPWQVLVASRGRAVCGGVLVHPQWVLTAAHCIRNK"
        "SVILLGRHSLFHPEDTGQVFQVSHSFPHPLYDMSLLKNRFLRPGDDSSHDLMLLRLSEPAELTDAVKVMD"
    ),
    "APP":      (
        "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQ"
        "YCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMD"
    ),
    "SNCA":     (
        "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVV"
        "TGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
    ),
    "MBP":      (
        "MASQKRPSQRHGSKYLATASTMDHARHGFLPRHRDTGILDSLGRFFGSDRSGQKDGGHKALGRTQDEN"
        "PVVHFFKNIVTPRTPPPSQGKGRGLSLSRFSWGAEGQKPGFGYGGRASTQDGSTGRGAPKRGSGKDGHH"
        "AARTTHYGSLPQKSQRSQDENPVVHFFKNIVTPRTP"
    ),
    "SCN5A":    (
        "MSRAVGPAGPGPGPGPRARREQPPPAAGVQAAAGAAASGPAAGGPAGPSSRTLGPAAGGPARPAAAGTGTG"
        "PGARPGARARGPQAASGPARAGPGSGARPPPAGAAAGPRAGP"
    ),
    "PCSK9":    (
        "MGTVSSRRSWWPLPLCVLLLFLGAGSGEDDPQHITEHVSEQYKKVSEVSQPLNLSQPQVAELQNLHAQFKPG"
        "EFHISLKAFGEDQVQQFQELLNNSYGQPVTLNLTPEQTQAEELQLEEQQPQFLLTLPQNAQIQPKAHNFPLL"
    ),
    "LEP":      (
        "MHWGTLCGFLWLWPYLFYVQAVPIQKVQDDTKTLIKTIVTRINDISHTQSVSSKQKVTGLDFIPGLHPLL"
        "SLSKMDQTLAVYQQILTSMPSRNVIQISNDLENLRDLLHVLAFSKSCHLPWASGLETLDSLGGVLEASGYS"
        "TEVVALSRLQGSLQDMLWQLDLSPGC"
    ),
    "TSHR":     (
        "MRPADLLQLVLLLDLPRDLGGMGCSSPLAISVSTTSLASLPFPDSQSPVNLSDIGSLEQQDAEIQVLHQM"
        "TSSALQFLGHILLQYPDSQLCPELRDYSHFCQEFLFGTMYNLTVHCKEDGEELFCTVSSSYQDLKDFLM"
    ),
    "HPRT1":    (
        "MATIKSAIKDINLIAANRNSSTADAKLVNIYEKAIVSHHNSQKKLEHKLDALIEQMQVGCGCMTFYMNSSMQ"
        "LQDPQYMAFSSEEVYAACASGQTIYEGKMDFMNDRSISQACSYLHIQTEVPQLIEMQAFRHHKKDSLYGNN"
    ),
    "UMOD":     (
        "MRAVALSLLLAVLQVSRAEGSSASGLSANIFEFKPTQQNPNTNGEIIIIEYEASPLTFQLKRNTTQGQMVS"
        "SQFTDNITSLDLQQLDQLQTSCDFQDSEKLDTITPGTEPVAQKLSPVNHSIKHTSGKLSQNLNLVQRVSC"
    ),
    "NOD2":     (
        "MSSFLSAPQSHQLHSMPQRSPAQALQIPGQDFYNRTQSPGIQSPALSTPVSPFIPRLLLHQLSAQQISRLQ"
        "EAGLASPHSMLGLPQTLPAQQPETIRNQLQQCQTPQSASSSLQPTQFLTPSVSQTLSPLSAHPQHLTPHQ"
    ),
    "PRSS1":    (
        "MNPLLILTFVAAALAAPFDDDDKIVGGYNCEENSVPYQVSLNSGYHFCGGSLINEQWVVSAGHCYKSRIQV"
        "RLGEHNIEVLEGNEQFINAAKIIRHPQYDRKTLNNDIMLIKLSSPATLNSIVTPPAICSGEGDTPFGTTCL"
    ),
    "ALB":      (
        "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELT"
        "EFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPD"
    ),
    "TNF":      (
        "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLA"
        "QAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGMDLKDNQLVLNPSAHVSALKLQELNQKIIE"
        "EIDLKNTNNLAPLNQMISQKQSQDLRGDKLISSNPQIVNATNMTQTRQRMQRQFLQTTK"
    ),
    "TREX1":    (
        "MGAASRYRQGRELLAAWRGSVNFLRPESRFHLPVTPGFKAAFFLQRYMQHKQDHHCPEIHLFEQLPEEMRE"
        "SSNPDLQSLQEELEHLNKMAPDLLQFLEQERQRQQRQIHQKASQLSQEQNQLQAVFHQFMDNTQSIQLTRRTN"
    ),
    "INS":      "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "LBP":      (
        "MKLPVFLLLMAALVAPVAHAEDPFLGTLQGRLKALLPKNKLSNLHTLKKQENQGQIEFQTFLDSTPAFSDKD"
        "SDTLLMDNLQHTLKSSRLQKNLQGQLHKMQSEVSSMQLQSTQLLQQLQNLRQQLQSRQVIQKQQNFQQ"
    ),
    "MPT64":    "MAERKHRSASVTAASPVSAGGSPSDLSALTEVDPAASAFTYGGDAFNRHGMQLLMGIAAQLPQDADPDLESVKDVEGQLRTMDGGQLEYSIVLNVVQDLQNIQSSVDSAFQSALSEIESQLNRASAQFGSIDSTDDGRGLSQNWPELV",
    "SLC6A4":   (
        "METTPLNSQKQLSACEDGEDCQENGVLQKVVPTPGDKVESGQISNGYSAVPSPGAGDDRHSIPATTTTLVVN"
        "TVVNPSSDLSRLPELLKASLAEDQNPPTQKLLQELESLIQPGLQQLHPLSSSPQDPIVTQAQLGVQQTAM"
    ),
    "DTNBP1":   "MSALQAARGVLEARNKEIMRQAFSRLGHQLQRGREDTLQDPNQALQRYLQHMKNLQHFIQEAQELQSRLNALYQKLQNLHSLQEAFQNIHESLQQKLQAFQEKLQSLQEMIQKLQSQIAKLKAEQARLQHQIQAHLETYQS",
    "ACE2":     (
        "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFL"
        "KEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECL"
    ),
    "F2":       (
        "MAHVRGLQLPGCLALAALCSLVHSQHVFLAPQQARSLLQRVRRANTFLEEVRKGNLERECVEETCSYEEAFEDALE"
        "STRPFPPSTHVNKRTFLEKLRPGEERQNLFQSVDGQMAEYFCSSPSDCQSGKRLLEELHPIFSGFSKSADQ"
    ),
    "FBN1":     (
        "MGLRAGPGLLLLAVACASLGSASTQDVLVNNTPQLRTGIQSKLPQLKISQTPTIPAQYPPQEFYSFFSQGMQV"
        "RDPHGQELRTAFPAVGEFDSDILKGLEEEVLDSSNPSWVSRHLHERLQKDQCFHGPNGQHLAQIPDLQR"
    ),
    "TP53":     (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAP"
        "PVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLYA"
    ),
}

DEFAULT_PROTEIN_NAME = "TP53"


# ── Layer 3 — BioPython ProtParam (local, no network) ────────────────────────

def _run_biopython_protparam(sequence: str) -> dict:
    """
    Compute real biophysical properties from amino acid sequence.
    Strips non-standard residues (X, B, Z, U, O, J) before analysis.
    All fields are None on any failure — never raises.
    """
    empty = {
        "instability_index": None,
        "isoelectric_point": None,
        "molecular_weight":  None,
        "gravy_score":       None,
        "aromaticity":       None,
        "secondary_structure": None,
    }
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        clean = "".join(c for c in sequence.upper() if c in standard_aa)
        if len(clean) < 10:
            return empty
        pa = ProteinAnalysis(clean)
        helix, turn, sheet = pa.secondary_structure_fraction()
        return {
            "instability_index": round(pa.instability_index(), 4),
            "isoelectric_point": round(pa.isoelectric_point(), 4),
            "molecular_weight":  round(pa.molecular_weight(), 2),
            "gravy_score":       round(pa.gravy(), 4),
            "aromaticity":       round(pa.aromaticity(), 4),
            "secondary_structure": {
                "helix": round(helix, 4),
                "turn":  round(turn,  4),
                "sheet": round(sheet, 4),
            },
        }
    except Exception as exc:
        log.warning("BioPython ProtParam failed: %s", exc)
        return empty


# ── Layer 1 — AlphaFold EBI Database API ─────────────────────────────────────

def _fetch_alphafold_plddt(uniprot_id: str) -> float | None:
    """
    Query AlphaFold EBI Database API for mean pLDDT confidence score.
    Returns mean_plddt (float) or None on any failure.
    Note: bacterial proteins (e.g. MPT64) will return 404 — handled silently.
    """
    try:
        url  = ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                            headers={"Accept": "application/json"})
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                item = data[0]
                # EBI API v6+ uses globalMetricValue; older versions used meanPlddt
                plddt = item.get("globalMetricValue") or item.get("meanPlddt")
                if plddt is not None:
                    return float(plddt)
        # 404 is expected for bacterial/non-human proteins — no warning needed
        if resp.status_code not in (404, 400):
            log.warning("AlphaFold API: status=%s for %s", resp.status_code, uniprot_id)
    except Exception as exc:
        log.warning("AlphaFold API failed for %s: %s", uniprot_id, exc)
    return None


# ── Layer 2 — UniProt REST API ────────────────────────────────────────────────

def _fetch_uniprot_data(uniprot_id: str) -> dict:
    """
    Fetch real sequence + function + disease associations from UniProt REST API.
    Returns dict with all keys; values are None / [] on failure.
    """
    result: dict = {
        "sequence": None,
        "protein_function": None,
        "disease_associations": [],
    }
    try:
        url  = UNIPROT_API_URL.format(uniprot_id=uniprot_id)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                            headers={"Accept": "application/json"})
        if resp.status_code != 200:
            log.warning("UniProt API: status=%s for %s", resp.status_code, uniprot_id)
            return result
        data = resp.json()

        # Real full-length sequence
        seq_obj = data.get("sequence", {})
        result["sequence"] = seq_obj.get("value")

        # First FUNCTION comment
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    result["protein_function"] = texts[0].get("value")
                    break

        # Disease IDs
        disease_ids: list[str] = []
        for comment in data.get("comments", []):
            if comment.get("commentType") == "DISEASE":
                d = comment.get("disease", {})
                did = d.get("diseaseId") or d.get("diseaseAcronym")
                if did:
                    disease_ids.append(did)
        result["disease_associations"] = disease_ids

    except Exception as exc:
        log.warning("UniProt API failed for %s: %s", uniprot_id, exc)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plddt_to_confidence(mean_plddt: float) -> str:
    """AlphaFold2 official pLDDT confidence bands."""
    if mean_plddt > 70:
        return "high"
    if mean_plddt > 50:
        return "medium"
    return "low"


def _stability_from_instability_index(instability_index: float | None) -> float:
    """
    Derive stability_score from Guruprasad 1990 instability index.
    <40 = stable (0.75), 40-60 = borderline (0.55), >60 = unstable (0.35).
    """
    if instability_index is None:
        return 0.55
    if instability_index < 40:
        return 0.75
    if instability_index <= 60:
        return 0.55
    return 0.35


# ── Public API ────────────────────────────────────────────────────────────────

def alphafold_tool(protein_name: str, uniprot_id: str) -> dict:
    """
    Real protein stability analysis: AlphaFold EBI pLDDT + UniProt sequence
    + BioPython biophysical properties.

    Signature UNCHANGED from previous mock version — all call sites are valid.
    Original 7 keys preserved; 12 new biophysical keys added.

    Args:
        protein_name: human-readable name (e.g., "BNP")
        uniprot_id:   UniProt accession (e.g., "P16860")

    Returns dict with all fields documented in models/schemas.py ProteinFoldingReport.
    """
    pdb_link = ALPHAFOLD_PDB_URL.format(uniprot_id=uniprot_id)

    # Layer 2 — real sequence from UniProt
    uniprot_data         = _fetch_uniprot_data(uniprot_id)
    real_sequence        = uniprot_data["sequence"]
    protein_function     = uniprot_data["protein_function"]
    disease_associations = uniprot_data["disease_associations"]

    if real_sequence:
        sequence        = real_sequence
        sequence_source = "uniprot_api"
    else:
        sequence        = PROTEIN_SEQUENCES.get(
            protein_name, PROTEIN_SEQUENCES[DEFAULT_PROTEIN_NAME]
        )
        sequence_source = "hardcoded_fallback"

    # Layer 3 — biophysical properties (always runs locally)
    biophys = _run_biopython_protparam(sequence)

    # Layer 1 — real pLDDT from AlphaFold EBI
    mean_plddt = _fetch_alphafold_plddt(uniprot_id)

    # Assemble stability_score from best available source
    if mean_plddt is not None:
        stability_score  = round(mean_plddt / 100, 4)
        confidence       = _plddt_to_confidence(mean_plddt)
        stability_source = "alphafold_api"
    else:
        stability_score  = _stability_from_instability_index(biophys["instability_index"])
        confidence       = _plddt_to_confidence(stability_score * 100)
        stability_source = "biopython_derived"

    return {
        # ── Backward-compatible keys (original 7) ─────────────────────────
        "protein_name":    protein_name,
        "uniprot_id":      uniprot_id,
        "stability_score": stability_score,
        "confidence":      confidence,
        "sequence_length": len(sequence),
        "pdb_link":        pdb_link,
        # ── New biophysical keys ──────────────────────────────────────────
        "mean_plddt":           round(mean_plddt, 2) if mean_plddt is not None else None,
        "instability_index":    biophys["instability_index"],
        "isoelectric_point":    biophys["isoelectric_point"],
        "molecular_weight":     biophys["molecular_weight"],
        "gravy_score":          biophys["gravy_score"],
        "aromaticity":          biophys["aromaticity"],
        "secondary_structure":  biophys["secondary_structure"],
        "sequence_source":      sequence_source,
        "stability_source":     stability_source,
        "protein_function":     protein_function,
        "disease_associations": disease_associations,
    }

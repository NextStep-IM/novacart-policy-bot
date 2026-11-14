import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

POLICIES_DIR = Path("policies")
OUTPUT_DIR = Path("processed")

EXPECTED_FILE_NUMS = [f"{i:02d}" for i in range(22)]
REQUIRED_METADATA_FIELDS = (
    "doc_id",
    "title",
    "version",
    "effective_date",
    "last_updated",
    "scope",
)
VALID_SCOPES = {"general", "specific", "regional"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Strictly line-anchored plain-text injection patterns (NEVER use re.DOTALL with these)
# Replace PLAIN_TEXT_INJECTION_PATTERNS in preprocess_policies.py with:
PLAIN_TEXT_INJECTION_PATTERNS = [
    re.compile(
        r"(?:^|\s+)(?:[*_>-]+\s*)*SYSTEM\s+OVERRIDE(?:\s*[*_]+)?\s*:.*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s+)(?:[*_>-]+\s*)*Internal\s+note\s+for\s+(?:customer\s+(?:service|support)\s+)?(?:bots|ai|assistants)(?:\s*[*_]+)?\s*:.*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s+)(?:[*_>-]+\s*)*(?:NOTE|INSTRUCTION)\s+FOR\s+(?:AI|ASSISTANTS|BOTS)(?:\s*[*_]+)?\s*:.*$",
        flags=re.IGNORECASE,
    ),
]

# Poisoned Markdown table data row pattern (preserves header, separator, and legitimate rows)
POISONED_TABLE_ROW_RE = re.compile(
    r"^\s*\|\s*Instruction\s+for\s+(?:assistants|AI|bots)\s*\|.*\|\s*$",
    flags=re.IGNORECASE,
)

# Suspicious phrasing detector for flagging ambiguous content without auto-deleting
SUSPICIOUS_PHRASE_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous\s+instructions|reveal\s+your\s+system\s+prompt|"
    r"system\s+override|instruction\s+for\s+assistants|customer\s+service\s+bots)",
    flags=re.IGNORECASE,
)

# HTML comment patterns
HTML_COMMENT_RE = re.compile(r"<!--([\s\S]*?)-->")
MALICIOUS_COMMENT_RE = re.compile(
    r"(?:NOTE\s+FOR\s+(?:AI|ASSISTANTS|BOTS)|SYSTEM\s+OVERRIDE|"
    r"IGNORE\s+(?:ALL\s+)?PREVIOUS|ALWAYS\s+APPROVE|INSTRUCTION\s+FOR)",
    flags=re.IGNORECASE,
)

# Structural Markdown patterns
H1_TITLE_RE = re.compile(r"^#\s+(.+)$")
SECTION_HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
NUMBERED_ITEM_RE = re.compile(r"^\*\*(\d+\.\d+)\*\*\s+(.+)$")

# Cross-reference patterns
VALID_XREF_RE = re.compile(r"\[file\s+(\d{2})(?:,\s*Section\s+(\d+(?:\.\d+)?))?\]")
ANY_FILE_BRACKET_RE = re.compile(r"\[file[^\]]*\]", flags=re.IGNORECASE)

# Required citation format for numbered items: <filename>.md, Section <X.Y>
ITEM_CITATION_RE = re.compile(r"^\d{2}_[a-z0-9_]+\.md, Section \d+\.\d+$")


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of a file using read-only binary access."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _normalize_yaml_scalar(value: Any) -> str:
    """Preserves YAML scalar semantics cleanly without altering dates or scopes."""
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def parse_front_matter(raw_text: str, source_name: str) -> tuple[dict[str, str], str, str]:
    """
    Parses and validates YAML front matter using yaml.safe_load().
    Returns (metadata_dict, raw_front_matter_block, body_markdown).
    """
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"[{source_name}] Missing or malformed YAML front matter delimiters ('---').")

    raw_yaml_block = match.group(1)
    body = match.group(2)

    try:
        parsed = yaml.safe_load(raw_yaml_block)
    except yaml.YAMLError as exc:
        raise ValueError(f"[{source_name}] Malformed YAML front matter: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"[{source_name}] YAML front matter must parse to a key-value mapping.")

    metadata: dict[str, str] = {}
    for field in REQUIRED_METADATA_FIELDS:
        if field not in parsed or parsed[field] is None:
            raise ValueError(f"[{source_name}] Missing or null required metadata field: '{field}'.")
        normalized_val = _normalize_yaml_scalar(parsed[field])
        if not normalized_val:
            raise ValueError(f"[{source_name}] Empty required metadata field: '{field}'.")
        metadata[field] = normalized_val

    # Validate date formats (YYYY-MM-DD) without altering values
    for date_field in ("effective_date", "last_updated"):
        if not ISO_DATE_RE.match(metadata[date_field]):
            raise ValueError(
                f"[{source_name}] Field '{date_field}' must be YYYY-MM-DD, got '{metadata[date_field]}'."
            )

    # Validate scope against file 00 conventions
    if metadata["scope"] not in VALID_SCOPES:
        raise ValueError(
            f"[{source_name}] Invalid scope '{metadata['scope']}'. Expected one of {sorted(VALID_SCOPES)}."
        )

    # Validate doc_id aligns with NN file prefix if filename starts with NN_
    prefix_match = re.match(r"^(\d{2})_", source_name)
    if prefix_match:
        expected_doc_id = f"NC-POL-{prefix_match.group(1)}"
        if metadata["doc_id"] != expected_doc_id:
            raise ValueError(
                f"[{source_name}] Expected doc_id '{expected_doc_id}', got '{metadata['doc_id']}'."
            )

    return metadata, raw_yaml_block, body


def sanitize_injections(
    raw_body: str, source_name: str
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Safely removes confirmed adversarial prompt injections while preserving all
    legitimate policy paragraphs, bullet lists, and Markdown table rows.

    Returns (cleaned_body, audit_entries, warnings).
    """
    audit_entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    working_text = raw_body

    # 1. Detect unclosed or stray HTML comment markers BEFORE regex substitution
    open_count = working_text.count("<!--")
    close_count = working_text.count("-->")
    if open_count != close_count:
        raise ValueError(
            f"[{source_name}] Unclosed or mismatched HTML comment detected "
            f"(open={open_count}, close={close_count}). Refusing to consume trailing content."
        )

    # 2. Remove confirmed malicious HTML comments (handles multiple comments cleanly)
    def _replace_html_comment(match: re.Match) -> str:
        full_comment = match.group(0)
        inner_text = match.group(1)
        if MALICIOUS_COMMENT_RE.search(inner_text):
            audit_entries.append({
                "file_name": source_name,
                "category": "html_comment_injection",
                "removed_text": full_comment.strip(),
                "occurrences": 1,
            })
            return ""
        warnings.append(
            f"[{source_name}] Ambiguous HTML comment preserved for review: {full_comment.strip()[:120]}"
        )
        return full_comment

    working_text = HTML_COMMENT_RE.sub(_replace_html_comment, working_text)

    # Verify ordering of comment tags was valid (e.g., no '-->' preceding '<!--')
    if "<!--" in working_text and MALICIOUS_COMMENT_RE.search(working_text):
        raise ValueError(f"[{source_name}] Residual HTML comment marker remained after sanitization.")

    # 3. Line-by-line sanitization for plain-text override lines and poisoned table rows
    cleaned_lines: list[str] = []
    for line_num, line in enumerate(working_text.splitlines(), start=1):
        stripped_line = line.strip()

        # Check poisoned Markdown table rows
        if POISONED_TABLE_ROW_RE.match(line):
            audit_entries.append({
                "file_name": source_name,
                "category": "poisoned_table_row",
                "removed_text": stripped_line,
                "occurrences": 1,
            })
            continue

        # Check plain-text adversarial instruction lines
# Check plain-text adversarial instruction lines (supports standalone or inline-suffix)
        matched_plain = False
        current_line = line
        for pattern in PLAIN_TEXT_INJECTION_PATTERNS:
            match = pattern.search(current_line)
            if match:
                audit_entries.append({
                    "file_name": source_name,
                    "category": "plain_text_injection",
                    "removed_text": match.group(0).strip(),
                    "occurrences": 1,
                })
                current_line = pattern.sub("", current_line).rstrip()
                matched_plain = True
                break

        if matched_plain:
            if current_line.strip():
                cleaned_lines.append(current_line)
            continue

        # Flag any remaining suspicious phrasing without deleting legitimate content
        if SUSPICIOUS_PHRASE_RE.search(line):
            warnings.append(
                f"[{source_name}:{line_num}] Suspicious phrasing preserved for review: {stripped_line[:120]}"
            )

        cleaned_lines.append(line)

    cleaned_body = "\n".join(cleaned_lines)
    cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body).strip()

    # Consolidate duplicate audit entries per (file_name, category, removed_text)
    consolidated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in audit_entries:
        key = (entry["file_name"], entry["category"], entry["removed_text"])
        if key not in consolidated:
            consolidated[key] = dict(entry)
        else:
            consolidated[key]["occurrences"] += entry["occurrences"]

    return cleaned_body, list(consolidated.values()), warnings


def extract_cross_references(
    text: str, source_name: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Extracts [file NN], [file NN, Section X], and [file NN, Section X.Y] references,
    and reports any malformed [file ...] bracket patterns without altering text.
    """
    refs: list[dict[str, Any]] = []
    warnings: list[str] = []

    for match in ANY_FILE_BRACKET_RE.finditer(text):
        raw_token = match.group(0)
        valid_match = VALID_XREF_RE.fullmatch(raw_token)
        if not valid_match:
            warnings.append(f"[{source_name}] Malformed cross-reference syntax: '{raw_token}'")
            continue

        target_file_num = valid_match.group(1)
        target_section = valid_match.group(2)  # None, '2', or '2.2'
        refs.append({
            "raw": raw_token,
            "target_file_num": target_file_num,
            "target_section": target_section,
        })

    return refs, warnings


def parse_document_chunks(
    filename: str,
    metadata: dict[str, str],
    clean_body: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """
    Splits a sanitized policy Markdown body into:
    - Overview chunk (introductory content before ## 1.)
    - Section-level intro chunks (content between ## N. and **N.1**, if any)
    - Numbered item chunks (**X.Y** plus all attached tables, lists, and paragraphs)
    """
    file_num = filename.split("_", 1)[0]
    warnings: list[str] = []
    chunks: list[dict[str, Any]] = []

    sections_seen: set[str] = set()
    items_seen: set[str] = set()

    current_section_num = "0"
    current_section_title = "Overview"
    in_document_overview = True

    buffer_lines: list[str] = []
    active_item_id: str | None = None
    active_chunk_type: str = "overview"

    def _base_metadata(
        chunk_id: str,
        chunk_type: str,
        sec_num: str,
        sec_title: str,
        item_id: str | None,
        citation: str,
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "file_name": filename,
            "file_num": file_num,
            "doc_id": metadata["doc_id"],
            "doc_title": metadata["title"],
            "version": metadata["version"],
            "effective_date": metadata["effective_date"],
            "last_updated": metadata["last_updated"],
            "scope": metadata["scope"],
            "section_num": sec_num,
            "section_title": sec_title,
            "item_id": item_id,
            "citation": citation,
        }

    def _flush_buffer() -> None:
        nonlocal buffer_lines, active_item_id, active_chunk_type
        text = "\n".join(buffer_lines).strip()
        buffer_lines = []
        if not text:
            return

        if active_chunk_type == "overview":
            chunk_id = f"{file_num}_overview"
            citation = f"{filename}, Overview"
            base = _base_metadata(chunk_id, "overview", "0", "Overview", None, citation)
        elif active_chunk_type == "section_intro":
            chunk_id = f"{file_num}_sec_{current_section_num}_intro"
            citation = f"{filename}, Section {current_section_num} (Overview)"
            base = _base_metadata(
                chunk_id,
                "section_intro",
                current_section_num,
                current_section_title,
                None,
                citation,
            )
        else:
            assert active_item_id is not None
            chunk_id = f"{file_num}_{active_item_id}"
            citation = f"{filename}, Section {active_item_id}"
            base = _base_metadata(
                chunk_id,
                "item",
                current_section_num,
                current_section_title,
                active_item_id,
                citation,
            )

        xrefs, xref_warnings = extract_cross_references(text, filename)
        warnings.extend(xref_warnings)

        base["text"] = text
        base["has_table"] = bool(re.search(r"^\|.+\|\s*$", text, flags=re.MULTILINE))
        base["has_list"] = bool(re.search(r"^(?:[\*\-]|\d+\.)\s+", text, flags=re.MULTILINE))
        base["cross_references"] = xrefs
        chunks.append(base)

    for raw_line in clean_body.splitlines():
        stripped = raw_line.strip()

        # Top-level document title (# Title)
        if H1_TITLE_RE.match(stripped) and not stripped.startswith("## "):
            continue

        # Section heading (## 3. Refund method and timing)
        sec_match = SECTION_HEADING_RE.match(stripped)
        if sec_match:
            _flush_buffer()
            in_document_overview = False
            current_section_num = sec_match.group(1)
            current_section_title = sec_match.group(2).strip()
            if current_section_num in sections_seen:
                raise ValueError(
                    f"[{filename}] Duplicate section number '{current_section_num}'."
                )
            sections_seen.add(current_section_num)
            active_item_id = None
            active_chunk_type = "section_intro"
            continue

        # Numbered item (**3.2** The processing time starts...)
        item_match = NUMBERED_ITEM_RE.match(stripped)
        if item_match:
            _flush_buffer()
            in_document_overview = False
            item_id = item_match.group(1)
            item_text = item_match.group(2).strip()

            item_major = item_id.split(".", 1)[0]
            if item_major != current_section_num:
                raise ValueError(
                    f"[{filename}] Item '{item_id}' does not match enclosing section '{current_section_num}'."
                )
            if item_id in items_seen:
                raise ValueError(f"[{filename}] Duplicate numbered item '{item_id}'.")
            items_seen.add(item_id)

            active_item_id = item_id
            active_chunk_type = "item"
            buffer_lines = [item_text]
            continue

        # Continuation lines: overview text, section-level text, tables, lists, paragraphs
        if in_document_overview:
            active_chunk_type = "overview"
        buffer_lines.append(raw_line)

    _flush_buffer()

    doc_structure = {
        "sections": sorted(sections_seen, key=int),
        "items": sorted(items_seen, key=lambda s: tuple(int(p) for p in s.split("."))),
    }
    return chunks, doc_structure, warnings


def validate_corpus(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    doc_structures: dict[str, dict[str, list[str]]],
    sanitization_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Runs full repository, metadata, sanitization, chunking, citation, and xref checks."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Repository validation
    found_nums = [d["file_num"] for d in documents]
    repo_valid = len(documents) == 22 and sorted(found_nums) == EXPECTED_FILE_NUMS
    if not repo_valid:
        missing = sorted(set(EXPECTED_FILE_NUMS) - set(found_nums))
        extra = sorted(set(found_nums) - set(EXPECTED_FILE_NUMS))
        errors.append(
            f"Repository file count/numbering mismatch: count={len(documents)}, missing={missing}, extra={extra}"
        )

    # 2. Metadata validation
    metadata_valid = True
    doc_meta_by_num = {d["file_num"]: d["metadata"] for d in documents}
    for doc in documents:
        meta = doc["metadata"]
        for field in REQUIRED_METADATA_FIELDS:
            if not meta.get(field):
                metadata_valid = False
                errors.append(f"[{doc['file_name']}] Missing metadata field '{field}'.")

    for chunk in chunks:
        expected_meta = doc_meta_by_num.get(chunk["file_num"], {})
        if (
            chunk.get("doc_id") != expected_meta.get("doc_id")
            or chunk.get("effective_date") != expected_meta.get("effective_date")
            or chunk.get("scope") != expected_meta.get("scope")
            or chunk.get("version") != expected_meta.get("version")
        ):
            metadata_valid = False
            errors.append(f"[{chunk['chunk_id']}] Chunk metadata diverges from parent document.")

    # 3. Sanitization validation
    sanitization_valid = True
    for doc in documents:
        if SUSPICIOUS_PHRASE_RE.search(doc["clean_markdown"]) or "<!--" in doc["clean_markdown"]:
            sanitization_valid = False
            errors.append(f"[{doc['file_name']}] Residual injection signature found in cleaned text.")

    # 4. Chunking & citation validation
    chunking_valid = True
    citation_valid = True
    seen_chunk_ids: set[str] = set()
    seen_citations: set[str] = set()

    for chunk in chunks:
        cid = chunk["chunk_id"]
        cit = chunk["citation"]
        if not chunk.get("text", "").strip():
            chunking_valid = False
            errors.append(f"[{cid}] Empty chunk text.")
        if cid in seen_chunk_ids:
            chunking_valid = False
            errors.append(f"Duplicate chunk_id: '{cid}'.")
        seen_chunk_ids.add(cid)

        if cit in seen_citations:
            citation_valid = False
            errors.append(f"Duplicate citation: '{cit}'.")
        seen_citations.add(cit)

        if chunk["chunk_type"] == "item":
            if not ITEM_CITATION_RE.match(cit):
                citation_valid = False
                errors.append(f"[{cid}] Invalid item citation format: '{cit}'.")
            target_items = doc_structures.get(chunk["file_num"], {}).get("items", [])
            if chunk["item_id"] not in target_items:
                citation_valid = False
                errors.append(f"[{cid}] Cited item '{chunk['item_id']}' not registered in document.")

    # 5. Cross-reference validation
    total_xrefs = 0
    valid_xrefs = 0
    broken_xrefs: list[dict[str, Any]] = []

    for chunk in chunks:
        for xref in chunk["cross_references"]:
            total_xrefs += 1
            t_file = xref["target_file_num"]
            t_sec = xref["target_section"]
            if t_file not in doc_structures:
                xref["valid"] = False
                broken_xrefs.append({
                    "source_chunk": chunk["chunk_id"],
                    "raw": xref["raw"],
                    "reason": f"Target file '{t_file}' does not exist.",
                })
                errors.append(f"[{chunk['chunk_id']}] Broken xref {xref['raw']}: target file missing.")
                continue

            if t_sec is None:
                xref["valid"] = True
                valid_xrefs += 1
            elif "." in t_sec:
                exists = t_sec in doc_structures[t_file]["items"]
                xref["valid"] = exists
                if exists:
                    valid_xrefs += 1
                else:
                    broken_xrefs.append({
                        "source_chunk": chunk["chunk_id"],
                        "raw": xref["raw"],
                        "reason": f"Target item '{t_sec}' not found in file '{t_file}'.",
                    })
                    errors.append(f"[{chunk['chunk_id']}] Broken xref {xref['raw']}: target item missing.")
            else:
                exists = t_sec in doc_structures[t_file]["sections"]
                xref["valid"] = exists
                if exists:
                    valid_xrefs += 1
                else:
                    broken_xrefs.append({
                        "source_chunk": chunk["chunk_id"],
                        "raw": xref["raw"],
                        "reason": f"Target section '{t_sec}' not found in file '{t_file}'.",
                    })
                    errors.append(
                        f"[{chunk['chunk_id']}] Broken xref {xref['raw']}: target section missing."
                    )

    injections_by_category: dict[str, int] = {}
    injections_by_file: dict[str, list[dict[str, Any]]] = {}
    total_injections = 0
    for entry in sanitization_logs:
        count = entry["occurrences"]
        total_injections += count
        cat = entry["category"]
        injections_by_category[cat] = injections_by_category.get(cat, 0) + count
        injections_by_file.setdefault(entry["file_name"], []).append(entry)

    total_sections = sum(len(s["sections"]) for s in doc_structures.values())
    total_items = sum(len(s["items"]) for s in doc_structures.values())

    return {
        "status": "PASS" if not errors else "FAIL",
        "documents_processed": len(documents),
        "sections_discovered": total_sections,
        "numbered_items_discovered": total_items,
        "chunks_generated": len(chunks),
        "injections_removed": {
            "total": total_injections,
            "by_category": injections_by_category,
            "by_file": injections_by_file,
        },
        "cross_references": {
            "total_extracted": total_xrefs,
            "valid": valid_xrefs,
            "broken_count": len(broken_xrefs),
            "broken_details": broken_xrefs,
        },
        "validation_checks": {
            "repository_valid": repo_valid,
            "metadata_valid": metadata_valid,
            "sanitization_valid": sanitization_valid,
            "chunking_valid": chunking_valid,
            "citation_valid": citation_valid,
            "cross_references_valid": len(broken_xrefs) == 0,
        },
        "warnings": warnings,
        "errors": errors,
    }


def run_pipeline(
    policies_dir: Path = POLICIES_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Executes the full read-only ingestion, sanitization, chunking, and validation pipeline."""
    if not policies_dir.exists() or not policies_dir.is_dir():
        raise FileNotFoundError(f"Policies directory '{policies_dir}' does not exist.")

    policy_files = sorted(policies_dir.glob("*.md"))
    if len(policy_files) != 22:
        raise ValueError(f"Expected exactly 22 Markdown files in '{policies_dir}', found {len(policy_files)}.")

    # Record pre-run SHA-256 hashes of every source file to guarantee read-only integrity
    hashes_before = {f.name: compute_sha256(f) for f in policy_files}

    all_documents: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    doc_structures: dict[str, dict[str, list[str]]] = {}
    all_sanitization_logs: list[dict[str, Any]] = []
    pipeline_warnings: list[str] = []

    for filepath in policy_files:
        raw_text = filepath.read_text(encoding="utf-8")
        metadata, raw_front_matter, raw_body = parse_front_matter(raw_text, filepath.name)
        clean_body, file_sanitization_logs, san_warnings = sanitize_injections(
            raw_body, filepath.name
        )
        all_sanitization_logs.extend(file_sanitization_logs)
        pipeline_warnings.extend(san_warnings)

        chunks, structure, parse_warnings = parse_document_chunks(
            filepath.name, metadata, clean_body
        )
        pipeline_warnings.extend(parse_warnings)

        file_num = filepath.name.split("_", 1)[0]
        doc_structures[file_num] = structure
        all_chunks.extend(chunks)

        all_documents.append({
            "file_name": filepath.name,
            "file_num": file_num,
            "sha256": hashes_before[filepath.name],
            "metadata": metadata,
            "raw_front_matter": raw_front_matter,
            "clean_markdown": clean_body,
            "section_count": len(structure["sections"]),
            "item_count": len(structure["items"]),
            "chunk_count": len(chunks),
            "injections_removed": len(file_sanitization_logs),
        })

    report = validate_corpus(all_documents, all_chunks, doc_structures, all_sanitization_logs)
    report["warnings"] = pipeline_warnings + report["warnings"]

    # Verify post-run SHA-256 hashes of original policy files are 100% unchanged
    hashes_after = {f.name: compute_sha256(f) for f in sorted(policies_dir.glob("*.md"))}
    integrity_ok = hashes_before == hashes_after
    report["source_integrity_verified"] = integrity_ok
    if not integrity_ok:
        report["status"] = "FAIL"
        report["errors"].append("CRITICAL: One or more source files in policies/ changed SHA-256 hash!")

    # Write outputs exclusively to output_dir (processed/)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy_chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "policy_documents.json").write_text(
        json.dumps(all_documents, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if report["status"] != "PASS":
        raise RuntimeError(
            "Corpus validation failed:\n- " + "\n- ".join(report["errors"])
        )

    return report


if __name__ == "__main__":
    summary = run_pipeline()
    print(json.dumps(summary, indent=2))

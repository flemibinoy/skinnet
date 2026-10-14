def representation_audit(frame):
    return {
        "sample_count": len(frame),
        "disease_counts": frame.dx.value_counts().to_dict(),
        "identifiers": [c for c in ("patient_id", "lesion_id") if c in frame],
        "skin_tone_status": "unavailable",
        "reason": "HAM10000 does not supply verified skin-tone annotations. No image-based estimation is performed.",
        "severity_status": "disabled: no ground-truth severity annotations",
    }

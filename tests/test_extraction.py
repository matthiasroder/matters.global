from matters import extract_candidate_matters, extraction_proposal


def test_extraction_accepts_notes_source():
    candidates = extract_candidate_matters(
        "- [ ] Publish matters.global walkthrough\n", source_type="notes"
    )

    assert candidates[0]["id"] == "publish_matters_global_walkthrough"
    assert candidates[0]["source_type"] == "notes"
    assert candidates[0]["conditions"][0]["truth"] is False


def test_extraction_accepts_conversation_source():
    proposal = extraction_proposal(
        "Goal: Make matters.global multi-user\n"
        "Decision: Use a public/private split for state files\n",
        source_type="conversation",
        existing_matters={"make_matters_global_multi_user_system"},
    )

    assert [candidate["id"] for candidate in proposal["candidates"]] == [
        "make_matters_global_multi_user",
        "use_a_public_private_split_for_state_files",
    ]
    assert proposal["dependency_candidates"] == [
        {
            "prerequisite": "make_matters_global_multi_user_system",
            "dependent": "make_matters_global_multi_user",
            "reason": "name overlap",
        }
    ]
    assert proposal["requires_confirmation"] is True


def test_extraction_accepts_pdf_text_source():
    candidates = extract_candidate_matters(
        "Problem: Institutions cannot see shared blockers\n",
        source_type="pdf",
    )

    assert candidates[0]["description"] == "Problem extracted from source."

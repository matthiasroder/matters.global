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


def test_extraction_accepts_speaker_prefixed_conversation_markers():
    candidates = extract_candidate_matters(
        "Matthias: I want a graph that shows what creativity research can unlock.\n"
        "Agent: Goal: Map creativity interventions to measurable outcomes\n"
        "Agent: Responsibility: Confirm extracted creativity matters before saving\n",
        source_type="conversation",
    )

    assert [candidate["id"] for candidate in candidates] == [
        "map_creativity_interventions_to_measurable_outcomes",
        "confirm_extracted_creativity_matters_before_saving",
    ]


def test_creativity_research_conditions_are_domain_specific():
    candidates = extract_candidate_matters(
        "Matter: Build a review workflow for creativity research claims\n",
        source_type="paper",
    )

    labels = [condition["label"] for condition in candidates[0]["conditions"]]
    assert labels == [
        "Source context is captured for: Build a review workflow for creativity research claims",
        "Evidence quality and limits are reviewed for: Build a review workflow for creativity research claims",
        (
            "Human reviewer accepts this creativity research matter before persistence: "
            "Build a review workflow for creativity research claims"
        ),
    ]


def test_dependency_candidates_ignore_generic_matters_overlap():
    proposal = extraction_proposal(
        "Goal: Build a creativity research matters graph\n",
        source_type="notes",
        existing_matters={
            "build_general_purpose_web_research_agent",
            "build_mozart_junior",
            "develop_matters_global_extraction_skill",
            "setup_creativity_findings_slide_pipeline",
        },
    )

    assert proposal["dependency_candidates"] == [
        {
            "prerequisite": "build_general_purpose_web_research_agent",
            "dependent": "build_a_creativity_research_matters_graph",
            "reason": "name overlap",
        },
        {
            "prerequisite": "setup_creativity_findings_slide_pipeline",
            "dependent": "build_a_creativity_research_matters_graph",
            "reason": "name overlap",
        },
    ]

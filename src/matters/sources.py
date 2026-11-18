"""Read citations kept alongside a Matters state file."""

import json


def sources_for_matter(state_path, matter_id):
    """Return citations for one known matter from its graph's source file."""
    path = state_path.with_suffix(".sources.json")
    try:
        with path.open(encoding="utf-8") as source_file:
            document = json.load(source_file)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid sources file {path}. {error}") from error

    if not isinstance(document, dict):
        raise ValueError(f"Invalid sources file {path}. Expected a JSON object.")
    version = document.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError(f"Invalid sources file {path}. Expected schema_version 1.")
    matter_sources = document.get("matter_sources")
    if not isinstance(matter_sources, dict):
        raise ValueError(f"Invalid sources file {path}. Expected a matter_sources object.")
    sources = matter_sources.get(matter_id, [])
    if not isinstance(sources, list):
        raise ValueError(f"Invalid sources file {path}. Expected a list for {matter_id}.")

    for number, source in enumerate(sources, start=1):
        try:
            validate_source(source)
        except ValueError as error:
            raise ValueError(
                f"Invalid source {number} for {matter_id} in {path}. {error}"
            ) from error
    return sources


def validate_source(source):
    if not isinstance(source, dict):
        raise ValueError("Expected an object.")
    for field in ("title", "url"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Expected a nonempty {field}.")

    authors = source.get("authors", [])
    if not isinstance(authors, list) or any(
        not isinstance(author, str) or not author.strip() for author in authors
    ):
        raise ValueError("Expected authors to be a list of nonempty names.")

    pages = source.get("pdf_pages", [])
    if not isinstance(pages, list):
        raise ValueError("Expected pdf_pages to be a list of inclusive page ranges.")
    for page_range in pages:
        if (
            not isinstance(page_range, list)
            or len(page_range) != 2
            or any(type(page) is not int for page in page_range)
            or not 1 <= page_range[0] <= page_range[1]
        ):
            raise ValueError(
                "Expected each PDF page range to be [first, last] with 1 <= first <= last."
            )

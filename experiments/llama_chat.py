import ollama

def get_prompt(text):
    template = '''
    [INST] <<SYS>>
    You are a precise information extraction system for geological and geoscience literature. 
    Your task is to extract events and their temporal relations, plus event argument structure and time bounds, from multi-sentence passages.

    Critical requirements:
    - Be literal and evidence-based. Extract only what the passage explicitly supports.
    - Resolve cross-sentence references and coreference (e.g., "this uplift", "the eruption", "it", "these flows").
    - Use domain-appropriate wording (e.g., "uplift", "intrusion", "orogeny", "basalt flows", "marine transgression", "rift initiation", "glaciation", "deposition", "metamorphism").
    - If an item is not stated or cannot be inferred with high confidence, output null/None for that field.
    - Do NOT invent events, times, or relations.

    Output format must be **valid JSON** matching the schema below, and nothing else (no commentary, no markdown).

    Temporal relations (allowed values ONLY):
    BEFORE, AFTER, DURING, CONTAINS, IDENTITY, EQUALS, OVERLAPS

    Semantics (use these precisely):
    - BEFORE: event1 ends strictly before event2 begins.
    - AFTER:  event1 begins after event2 ends.
    - DURING: event1 occurs entirely within the bounds of event2.
    - CONTAINS: event1 fully contains event2.
    - IDENTITY/EQUALS: event1 and event2 refer to the same event or same time span (treat as synonyms).
    - OVERLAPS: events intersect in time but neither fully contains the other.

    Time normalization:
    - Prefer explicit ISO if present (e.g., "1998-06" or a year range like "-0120/0010" for BCE/CE when stated).
    - If geological periods/epochs are given (e.g., "Late Cretaceous"), keep them as canonical strings (e.g., "Late Cretaceous") and do not guess numeric Ma unless explicitly provided.
    - If only ordering is known (before/after), set start_time/end_time to null/None.
    - If a bounded interval is stated (e.g., "between 72-66 Ma"), set start_time="72 Ma", end_time="66 Ma" (preserve units/labels exactly as written).

    Event arguments:
    - subject: the agent/undergoer or grammatical subject of the event if stated (e.g., "arc magmatism", "ice sheets", "fault block"), else null.
    - object: the patient/theme/goal if stated (e.g., "basalt flows", "sediments", "crust"), else null.
    - event: a short verb-centric phrase naming the event (e.g., "uplift occurred", "basalt flows erupted", "marine sediments were deposited").
    - start_time/end_time: normalized per rules above (string or null).
    - Keep spans concise and faithful to the text.

    Event identification:
    - Treat each distinct geologic process/change as an event (e.g., initiation, peak, cessation can be separate events if text distinguishes them).
    - Merge exact duplicates (different mentions of the same event/time span) and prefer the most informative wording.

    JSON SCHEMA (must follow exactly):
    {
    "events": [
        {
        "id": "E1",
        "event": "string",
        "subject": "string or null",
        "object": "string or null",
        "start_time": "string or null",
        "end_time": "string or null",
        "evidence_span": "verbatim snippet from the passage indicating this event"
        },
        ...
    ],
    "temporal_triples": [
        {
        "event1_id": "E#",
        "temp_relation": "BEFORE|AFTER|DURING|CONTAINS|IDENTITY|EQUALS|OVERLAPS",
        "event2_id": "E#",
        "evidence_span": "verbatim snippet that justifies the relation"
        },
        ...
    ]
    }

    Validation rules:
    - Every temporal_triple must reference event ids present in "events".
    - Use stable ids "E1", "E2", ... in order of first appearance.
    - evidence_span must be a short quote (<= 30 words) from the passage.
    - Return at least the events explicitly present; it is acceptable to return an empty temporal_triples array if no justified relation is stated.
    - Output MUST be valid JSON (no trailing commas, no comments).
    <</SYS>>

    Extract events and temporal relations from the following passage:

    {TEXT}
    [/INST]
    '''
    return template.replace("{TEXT}", text)

extract = "Multiple drill lines along 20km of the Prairie Downs Fault (PDF) were completed in the 2017-2018 exploration season. A total of 6276.6m was drilled for 54 drill holes. The aim of the program was to test 20km of the PDF for base metal mineralisation in tenements E52 and E52. Numerous drill holes intersected significant base metal, vanadium and gold mineralisation including 19m @ 5.9% Pb, 0.1% Zn, 0.1% Cu and 40 g/t Ag from 87m in hole PDP456, at the Husky South prospect. Down hole total electro magnetics was completed on the two diamond drill holes PDD504 and PDD506 at Husky South. No significant off hole responses were detected. "
prompt = get_prompt(extract)

resp = ollama.chat(
    model='llama2:13b-chat',
    messages=[{'role':'user','content': prompt}],
    options={'temperature': 0.2, 'top_p': 0.9, 'seed': 42}
)
print(resp['message']['content'])
import random


def toy_agent_log() -> str:
    """
    Small toy log (kept for earlier tests).
    """
    return (
        "Project: USC\n"
        "Goal: build a unified state codec.\n"
        "Decision: start with USC-MEM v0.\n"
        "Next: implement skeleton + witnesses + residual patches.\n"
        "Note: always keep ROADMAP, FILEMAP, MASTER_HANDOFF, CHANGES updated.\n"
        "\n"
        "Session recap:\n"
        "- Created repo scaffold.\n"
        "- Fixed pyenv python version.\n"
        "- Installed usc editable.\n"
        "- Installed pytest.\n"
        "- Ran tests successfully.\n"
        "\n"
        "Reminder: compress memories safely using anchors and residuals.\n"
        "Reminder: compress memories safely using anchors and residuals.\n"
        "Reminder: compress memories safely using anchors and residuals.\n"
    )


def toy_big_agent_log(repeats: int = 30) -> str:
    """
    Big toy log (repetition-heavy). Good for gzip showcase, not fair for USC.
    """
    base = toy_agent_log()
    out = []
    for i in range(repeats):
        out.append(f"--- LOOP {i} ---\n")
        out.append(base)
        out.append("\n")
    return "".join(out)


def toy_big_agent_log_varied(loops: int = 30, seed: int = 7) -> str:
    """
    Big toy log (variation-heavy). Fairer for USC.

    Properties:
    - meaning stays similar, but wording changes each loop
    - unique tokens appear each loop (IDs, timestamps, small random words)
    - occasional new decisions appear
    """
    rng = random.Random(seed)

    project_titles = [
        "Project: USC",
        "Project: Unified State Codec",
        "Project: USC System",
        "Project: USC Memory Layer",
    ]

    goal_lines = [
        "Goal: build a unified state codec.",
        "Goal: compress agent memory safely.",
        "Goal: create tiered compression with verification.",
        "Goal: reduce memory bottlenecks in AI systems.",
    ]

    next_lines = [
        "Next: implement skeleton + witnesses + residual patches.",
        "Next: add chunking and prioritize important chunks.",
        "Next: tune probes and confidence gates for stability.",
        "Next: replace JSON packets with binary format.",
    ]

    notes = [
        "Note: always keep ROADMAP, FILEMAP, MASTER_HANDOFF, CHANGES updated.",
        "Note: refuse silent hallucination; upgrade tiers when uncertain.",
        "Note: commit known-good decodes to prevent drift.",
        "Note: prefer utility compression over perfect recall when safe.",
    ]

    recap_templates = [
        [
            "Session recap:\n",
            "- Created scaffold and wired CLI.\n",
            "- Added tiering with auto-upgrade.\n",
            "- Added commit loop output.\n",
        ],
        [
            "Session recap:\n",
            "- Bench harness prints ratios and confidence.\n",
            "- Probes validate chunks safely.\n",
            "- Priority rules select important chunks.\n",
        ],
        [
            "Session recap:\n",
            "- Added ECC checksum over truth spine.\n",
            "- Added fingerprint behavior ID.\n",
            "- Added fallback decode pathway.\n",
        ],
    ]

    decisions = [
        "Decision: keep Tier 0 tiny for boring chunks.",
        "Decision: Tier 3 must stay lossless and verified.",
        "Decision: auto-tier escalation is mandatory for safety.",
        "Decision: commit loop is required after safe decode.",
    ]

    reminders = [
        "Reminder: anchors + residuals beat raw text dumps.",
        "Reminder: chunk priority prevents wasting bits.",
        "Reminder: probes gate unsafe decode attempts.",
        "Reminder: store truth spine first; details second.",
    ]

    out = []
    for i in range(loops):
        # Unique tokens each loop (hurts gzip, helps USC meaning structure)
        loop_id = f"LOOP={i} ID={rng.randint(100000, 999999)}"

        proj = rng.choice(project_titles)
        goal = rng.choice(goal_lines)
        nxt = rng.choice(next_lines)
        note = rng.choice(notes)

        # Sometimes add a new decision line
        dec = rng.choice(decisions)
        if rng.random() < 0.25:
            dec = dec + f" (variant={rng.choice(['A','B','C'])})"

        # Recap changes slightly
        recap = rng.choice(recap_templates)
        recap_block = "".join(recap)

        # Reminders vary and include unique token
        rem = rng.choice(reminders) + f" [{loop_id}]"

        # Construct loop
        out.append(f"--- {loop_id} ---\n")
        out.append(proj + "\n")
        out.append(goal + "\n")
        out.append(dec + "\n")
        out.append(nxt + "\n")
        out.append(note + "\n")
        out.append("\n")
        out.append(recap_block)
        out.append("\n")
        out.append(rem + "\n")
        out.append("\n")

    return "".join(out)

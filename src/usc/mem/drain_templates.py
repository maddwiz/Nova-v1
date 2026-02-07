from typing import List, Tuple

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


def build_drain_miner() -> TemplateMiner:
    """
    Drain3 v0.9.x wants: TemplateMiner(persistence_handler, config)
    We disable persistence for now: persistence_handler = None
    """
    cfg = TemplateMinerConfig()

    # Safe mild tuning
    cfg.profiling_enabled = False
    cfg.drain_depth = 4
    cfg.max_children = 100
    cfg.sim_th = 0.4

    # IMPORTANT: arg order is (persistence_handler, config)
    return TemplateMiner(None, cfg)


def drain_extract_templates(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Returns:
      templates_per_line: Drain template per line
      params_per_line: Drain extracted params per line
    """
    miner = build_drain_miner()

    templates: List[str] = []
    params_per_line: List[List[str]] = []

    for line in lines:
        r = miner.add_log_message(line)

        t = r.get("cluster_template", line)
        params = r.get("parameter_list", []) or []

        templates.append(t)
        params_per_line.append(params)

    return templates, params_per_line


def convert_drain_to_numeric(params_per_line: List[List[str]]) -> List[List[int]]:
    """
    v0 encoding:
    - numbers -> int
    - strings -> [len, byte1, byte2, ...] up to 32 bytes
    Later: replace with persistent string dictionary.
    """
    out: List[List[int]] = []

    for row in params_per_line:
        new_row: List[int] = []

        for v in row:
            v = (v or "").strip()

            if v.isdigit():
                new_row.append(int(v))
            else:
                b = v.encode("utf-8")
                new_row.append(len(b))
                for ch in b[:32]:
                    new_row.append(int(ch))

        out.append(new_row)

    return out

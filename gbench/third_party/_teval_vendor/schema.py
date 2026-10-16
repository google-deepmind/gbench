# -*- coding: utf-8 -*-
# Vendored from T-Eval (github.com/open-compass/T-Eval, schema.py) for leaderboard-faithful
# scoring. Verbatim except: mmengine.load -> stdlib json shim; teval.* imports -> relative;
# unused `import evaluate`/`termcolor` dropped. Do NOT edit the scoring logic.
# T-Eval: Chen et al., ACL 2024 (arXiv:2312.14033).

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ResponseDataSample:
    """
    Args:
        template(str): Format string with keyword-only arguments. For
            example '{who} like {what}'
        pred(Any): Parsed data from LLM generating response.
        gt(Any): Ground truth data
        meta_data(dict, optional): Meta information will be used to evaluate
             LLM's response
    """
    template: str
    pred: Any
    gt: Any
    meta_data: dict = None
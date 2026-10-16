# -*- coding: utf-8 -*-
# Vendored from T-Eval (github.com/open-compass/T-Eval, __init__.py) for leaderboard-faithful
# scoring. Verbatim except: mmengine.load -> stdlib json shim; teval.* imports -> relative;
# unused `import evaluate`/`termcolor` dropped; and the `from sentence_transformers import ...`
# lines wrapped in try/except so `import eval_suites` does not crash the whole package when
# sentence-transformers is absent (t_eval hard-errors via infra_required at run time instead).
# Do NOT edit the scoring logic.
# T-Eval: Chen et al., ACL 2024 (arXiv:2312.14033).

from .instruct_evaluator import InstructEvaluator
from .planning_evaluator import PlanningEvaluator
from .reason_retrieve_understand_evaluator import ReasonRetrieveUnderstandEvaluator
from .review_evaluator import ReviewEvaluator
from .schema import ResponseDataSample

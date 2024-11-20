from dataclasses import dataclass, field
from typing import Union, List, Dict, Any

Message = dict[str, Any]  # keys role, content
MessageList = List[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: Union[float, None]  # top-line metric
    metrics: Union[Dict[str, float], None]  # other metrics
    htmls: List[str]  # strings of valid HTML
    convos: List[MessageList]  # sampled conversations


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: Union[float, None]
    metrics: Dict[str, float] = field(default_factory=dict)
    html: Union[str, None] = None
    convo: Union[MessageList, None] = None  # sampled conversation


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError

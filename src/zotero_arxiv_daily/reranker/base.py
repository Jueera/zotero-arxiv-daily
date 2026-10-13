from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score([c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) # [n_candidate]
        interests = self.config.get("research", {}) if self.config is not None else {}
        topics = interests.get("topics", [])
        if topics:
            weight = float(interests.get("weight", 0.7))
            if not 0 <= weight <= 1:
                raise ValueError("research.weight must be between 0 and 1")
            # Match title + abstract against complete topic descriptions, not isolated words.
            topic_sim = self.get_similarity_score(
                [f"{c.title}\n{c.abstract}" for c in candidates], list(topics)
            )
            assert topic_sim.shape == (len(candidates), len(topics))
            # A paper can be relevant to any one research direction.
            topic_scores = topic_sim.max(axis=1)
            scores = (1 - weight) * scores + weight * topic_scores
        scores = scores * 10
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]

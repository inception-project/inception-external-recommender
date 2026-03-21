# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_span_prediction
from ariadne.protocol import PredictionRequest

from typing import Any, List, Dict, Iterable, Optional, Tuple, Set, Union, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


@runtime_checkable
class GLiNER2LikeExtractor(Protocol):
    def extract_entities(
        self, text: str, entity_types, include_spans: bool = True, include_confidence: bool = False
    ) -> Dict[str, Any]: ...


@dataclass
class ExtractedEntity:
    start: int
    end: int
    label: str
    score: Optional[float] = None


@dataclass
class GLiNERSchema:
    """Canonical representation of a GLiNER schema returned from tagsets.

    Use `mapping` when richer descriptions are available (label -> description).
    `labels` always holds the list of label names (in lower-case as used
    when passed to GLiNER).
    """

    mapping: Optional[Dict[str, str]] = None
    labels: Optional[List[str]] = None

    def has_descriptions(self) -> bool:
        return self.mapping is not None

    def as_extractor_input(self) -> Union[Dict[str, str], List[str]]:
        return self.mapping if self.mapping is not None else (self.labels or [])

    def __str__(self) -> str:  # pragma: no cover - readable logging representation
        if self.mapping:
            entries = ", ".join(f"{k}: {v!r}" for k, v in self.mapping.items())
            return f"GLiNERSchema(dict: {entries})"
        return f"GLiNERSchema(list: {self.labels})"


def _split_text(t: str, w: int, o: int) -> List[Tuple[int, str]]:
    """Split text into (start_offset, chunk_text) tuples using character windows.

    Returns a list of (start, text) tuples. If window size `w` is non-positive,
    returns the whole text as a single chunk starting at 0.
    """
    if w <= 0:
        return [(0, t)]
    if o < 0:
        o = 0
    step = w - o
    if step <= 0:
        step = w

    chunks: List[Tuple[int, str]] = []
    start = 0
    length = len(t)
    while start < length:
        end = min(start + w, length)
        chunks.append((start, t[start:end]))
        if end == length:
            break
        start += step

    return chunks


def _span_length(p: Dict[str, Any]) -> int:
    return p["end"] - p["start"]


def _score_val(p: Dict[str, Any]) -> float:
    s = p.get("score")
    if s is None:
        return float("-inf")
    try:
        return float(s)
    except Exception:
        return float("-inf")


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


class Gliner2SpanRecommender(Classifier):
    """Span recommender adapter for GLiNER2."""

    # Class-level defaults and documented attributes set by the constructor
    _entity_types: Optional[List[str]] = None
    _model: Any = None
    window_size: int = 1000
    overlap: int = 150

    def __init__(
        self,
        model: Union[str, "GLiNER2LikeExtractor"] = "fastino/gliner2-base-v1",
        entity_types: Optional[Iterable[str]] = None,
    ):
        super().__init__()

        self._entity_types = list(entity_types) if entity_types is not None else None

        # If the caller provided a model name (string) load GLiNER2 lazily.
        if isinstance(model, str):
            try:
                from gliner2 import GLiNER2  # type: ignore
            except Exception as e:  # pragma: no cover - environment dependent
                raise ImportError(
                    "gliner2 is required for Gliner2SpanRecommender when loading by name."
                    " Install it in your environment, e.g. `pip install gliner2`,"
                    " or add `gliner2` to your project's dependencies and install via your environment tool."
                ) from e

            try:
                self._model = GLiNER2.from_pretrained(model)
            except Exception as e:
                raise RuntimeError(f"Could not load GLiNER2 model '{model}': {e}") from e
        else:
            # Accept any object that follows the Extractor Protocol (duck-typed).
            if isinstance(model, GLiNER2LikeExtractor) or hasattr(model, "extract_entities"):
                self._model = model
            else:
                raise TypeError(
                    "When passing a model instance it must implement extract_entities(text, entity_types, ...)"
                )

    def on_prediction_request(self, req: PredictionRequest):
        cas = req.cas
        layer = req.layer
        feature = req.feature
        document_id = req.document_id
        text = cas.sofa_string

        # Prefer schema derived from tagsets, fall back to a default schema.
        schema = self._build_schema_from_tagsets(req) or self._build_default_schema()

        # Build a case-insensitive map from lowercased tag name -> original tag name
        tagname_map = self._build_tagname_map(req)

        # Extractor input is derived from the canonical schema object
        entity_types_or_schema = schema.as_extractor_input()

        # Log schema details: pass the canonical `GLiNERSchema` object so it
        # decides its own representation via `__str__`.
        logger.debug(
            "Running GLiNER2SpanRecommender on document '%s' with schema: %s",
            document_id,
            schema,
        )

        # Keep track of already emitted spans (start,end,label) to avoid
        # duplicates across overlapping windows. Exclude score from the key
        # because floating-point noise can prevent proper deduplication.
        seen: Set[Tuple[int, int, str]] = set()

        # Collect raw predictions from all chunks first, then filter overlaps
        all_preds: List[Dict[str, Any]] = []

        # Call the extractor on sliding windows to collect predictions.
        chunks = _split_text(text, self.window_size, self.overlap)
        for chunk_start, chunk_text in chunks:
            self._predict_on_chunk(
                cas,
                tagname_map,
                entity_types_or_schema,
                seen,
                all_preds,
                chunk_start,
                chunk_text,
            )

        kept = self._select_best_predictions(all_preds)

        # Add kept predictions to CAS
        for p in kept:
            pred = create_span_prediction(
                cas, layer, feature, int(p["start"]), int(p["end"]), p["label"], score=p.get("score")
            )
            cas.add(pred)

    def _select_best_predictions(self, all_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter overlapping spans: keep longest, then by score, then earliest start
        sort by length desc, score desc, start asc
        """
        all_preds.sort(key=lambda p: (_span_length(p) * -1, -_score_val(p), p["start"]))

        kept: List[Dict[str, Any]] = []

        for p in all_preds:
            s = p["start"]
            e = p["end"]
            conflict = False
            for k in kept:
                if _overlaps(s, e, k["start"], k["end"]):
                    conflict = True
                    break
            if not conflict:
                kept.append(p)
        return kept

    def _normalize_extractor_result(self, raw: Any, chunk_start: int) -> List[ExtractedEntity]:
        """Validate and normalize extractor output into ExtractedEntity list.

        This helper is defensive: it accepts the common dict shape documented
        above and returns an empty list for unexpected shapes while logging a
        debug message.
        """
        out: List[ExtractedEntity] = []
        if not isinstance(raw, dict):
            logger.debug("extract_entities returned unexpected type: %s", type(raw))
            return out

        entities = raw.get("entities")
        if not isinstance(entities, dict):
            logger.debug("extract_entities returned non-dict 'entities' field: %r", type(entities))
            return out

        for label, items in entities.items():
            if not isinstance(items, (list, tuple)):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                start = item.get("start")
                end = item.get("end")
                if start is None or end is None:
                    continue
                try:
                    s = int(start) + int(chunk_start)
                    e = int(end) + int(chunk_start)
                except Exception:
                    continue

                score = item.get("confidence") or item.get("score")
                if score is not None:
                    try:
                        score = float(score)
                    except Exception:
                        score = None

                out.append(ExtractedEntity(start=s, end=e, label=label, score=score))

        return out

    def _predict_on_chunk(
        self,
        cas,
        tagname_map,
        entity_types_or_schema,
        seen,
        all_preds,
        chunk_start,
        chunk_text,
    ) -> None:
        raw = self._model.extract_entities(
            chunk_text, entity_types_or_schema, include_spans=True, include_confidence=True
        )

        extracted_entities = self._normalize_extractor_result(raw, chunk_start)

        for ent in extracted_entities:
            mapped_label = tagname_map.get(ent.label.lower(), ent.label)
            # use (start,end,label) as dedupe key; ignore score to avoid
            # floating-point differences causing duplicate entries
            key = (ent.start, ent.end, mapped_label)
            if key in seen:
                continue
            seen.add(key)

            all_preds.append({"start": int(ent.start), "end": int(ent.end), "label": mapped_label, "score": ent.score})
            try:
                doc_text = cas.sofa_string
                excerpt = doc_text[int(ent.start) : int(ent.end)]
            except Exception:
                excerpt = ""
            logger.debug(
                "GLiNER2 extracted entity: '%s' '%s' [%d, %d] score=%s",
                excerpt,
                mapped_label,
                ent.start,
                ent.end,
                ent.score,
            )

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Backwards-compatible wrapper: construct a PredictionRequest and delegate
        req = PredictionRequest(cas, layer, feature, project_id, document_id, user_id, tagsets=None)
        return self.on_prediction_request(req)

    def _build_schema_from_tagsets(self, req: PredictionRequest) -> Optional[GLiNERSchema]:
        """Convert `req.tagsets` into a GLiNER schema (dict type->description or list of type names).

        Returns None when there are no applicable tagsets or no tags.
        """
        sel = self._select_applicable_tagsets(req)
        if sel is None:
            return None
        applicable, used_fallback = sel

        if used_fallback:
            logger.debug(
                "No tagsets matched layer=%s feature=%s; falling back to all provided tagsets",
                req.layer,
                req.feature,
            )
        else:
            logger.debug(
                "Selected tagsets matching layer=%s feature=%s",
                req.layer,
                req.feature,
            )

        # Log which tagsets and targets were selected for visibility
        try:
            selected_info = []
            for ts in applicable:
                desc = ts.description
                targets = ts.targets or []
                tlist = [f"{t.type}:{t.feature}" for t in targets]
                tags = [tag.tag for tag in (ts.tags or [])]
                selected_info.append({"description": desc, "targets": tlist, "tags": tags})
            logger.debug("Applicable tagsets: %s", selected_info)
        except Exception:
            logger.debug("Could not serialize applicable tagsets for debug output")

        schema: Dict[str, str] = {}
        for ts in applicable:
            desc = ts.description or ""
            for tag in ts.tags or []:
                name = tag.tag
                if not name:
                    continue
                tag_desc = tag.description or desc or ""
                # Use lower-cased tag names when handing schema to GLiNER2
                schema[name.lower()] = tag_desc

        if not schema:
            return None

        # If any description is non-empty, return the dict for richer context,
        # otherwise return the simple list of names wrapped in GLiNERSchema.
        if any(v for v in schema.values()):
            return GLiNERSchema(mapping=schema, labels=list(schema.keys()))
        return GLiNERSchema(mapping=None, labels=list(schema.keys()))

    def _build_default_schema(self) -> GLiNERSchema:
        """Return a default GLiNERSchema based on configured entity types or
        a reasonable fallback list.
        """
        if self._entity_types is None:
            labels = ["person", "company", "product", "location"]
            logger.info(
                "No tagsets or entity types configured; using default GLiNER2 schema: %s",
                labels,
            )
        else:
            labels = list(self._entity_types)
            logger.info("Using configured entity types as schema: %s", labels)

        labels = [l.lower() for l in labels]
        return GLiNERSchema(mapping=None, labels=labels)

    def _build_tagname_map(self, req: PredictionRequest) -> Dict[str, str]:
        """Return a mapping from lowercased tag name -> original tag name as present in the tagsets.

        This is used to translate GLiNER2's predicted labels back to the original
        tag names accepted by INCEpTION (case-preserving), matching case-insensitively.
        """
        sel = self._select_applicable_tagsets(req)
        if sel is None:
            return {}
        applicable, _used_fallback = sel

        mapping: Dict[str, str] = {}
        for ts in applicable:
            for tag in ts.tags or []:
                name = tag.tag
                if not name:
                    continue
                lower = name.lower()
                # prefer the first-seen original casing
                mapping.setdefault(lower, name)

        return mapping

    def _select_applicable_tagsets(self, req: PredictionRequest) -> Optional[Tuple[List[Any], bool]]:
        """Return (applicable_tagsets, used_fallback).

        - If `req.tagsets` is falsy, return None to indicate no tagsets.
        - If tagsets exist, return a list of tagsets that target the
          requested layer/feature. If none match, return the full list and
          set `used_fallback=True`.
        """
        if not req.tagsets:
            return None

        applicable: List[Any] = []
        for ts in req.tagsets:
            targets = ts.targets or []
            for t in targets:
                if t.type == req.layer and t.feature == req.feature:
                    applicable.append(ts)
                    break

        if not applicable:
            return list(req.tagsets), True
        return applicable, False

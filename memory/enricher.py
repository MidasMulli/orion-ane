"""
Phantom Enricher — Always-on background intelligence for your knowledge vault
===============================================================================

While your LLM is idle, the enricher continuously scans and improves your
knowledge base. It reclassifies mistyped facts, discovers entity relationships,
flags stale information, detects cross-entity patterns, and consolidates
entity profiles.

Runs on CPU by default (zero GPU impact). Designed for ANE swap-in when a
CoreML classifier is available.

Architecture:
    PhantomEnricher (orchestrator)
      └── SweepEngine (5 sweep types)
            ├── RECLASSIFY — fix mistyped "general" facts
            ├── RELATE — build entity relationship graph
            ├── STALE — flag outdated time-sensitive facts
            ├── PATTERNS — cross-entity analysis and anomaly detection
            └── CONSOLIDATE — auto-generate entity profile summaries
      └── EnrichmentTracker (state persistence)

Usage:
    enricher = PhantomEnricher(store=memory_store, vault=vault_writer)
    enricher.start()   # runs forever in background thread
    enricher.stop()

    # Or one-shot:
    enricher.run_once()
"""

import os
import re
import json
import time
import threading
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger("phantom.enricher")


# ── Protocols for ANE swap-in ──────────────────────────────────

class Classifier(Protocol):
    """Protocol for fact classifier. Swap CPU regex for ANE CoreML."""
    def classify(self, text: str) -> str: ...


class Embedder(Protocol):
    """Protocol for text embedder. Swap CPU sentence-transformers for ANE CoreML."""
    def encode(self, texts: list[str]) -> np.ndarray: ...


# ── Default CPU implementations ────────────────────────────────

class RegexClassifier:
    """Default classifier using the same keyword heuristics as FactExtractor."""

    def __init__(self):
        try:
            from phantom_memory.daemon import FactExtractor
        except ImportError:
            from daemon import FactExtractor
        self._classify = FactExtractor.classify_type

    def classify(self, text: str) -> str:
        return self._classify(text)


class ANEClassifier:
    """Hybrid classifier: regex for type classification + ANE for analysis.

    The 1.7B model is excellent at analysis/extraction/summarization but
    unreliable for constrained single-word classification (always says "decision").
    So we keep regex for classify() and add analyze() for richer tasks.
    """

    VALID_TYPES = {"decision", "task", "preference", "quantitative", "general"}

    def __init__(self, socket_path: str = "/tmp/orion-ane-server.sock"):
        self._socket_path = socket_path
        self._regex = RegexClassifier()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from phantom_memory.ane_server import ANEClient
            except ImportError:
                from ane_server import ANEClient
            self._client = ANEClient(self._socket_path)
        return self._client

    def classify(self, text: str) -> str:
        """Classify using regex (fast, reliable for type detection)."""
        return self._regex.classify(text)

    def analyze(self, prompt: str, max_tokens: int = 100) -> str:
        """Run generative analysis on ANE (entity profiles, relationships, summaries)."""
        try:
            client = self._get_client()
            return client.analyze(prompt, max_tokens=max_tokens)
        except Exception as e:
            return f"[ANE unavailable: {e}]"

    @property
    def has_ane(self) -> bool:
        """Check if ANE server is available for analysis."""
        try:
            self._get_client().ping()
            return True
        except Exception:
            return False


class CPUEmbedder:
    """Default embedder using sentence-transformers on CPU."""

    def __init__(self, model=None):
        self._model = model  # Reuse existing MemoryStore model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32,
        )


# ── Enrichment State Tracker ──────────────────────────────────

class EnrichmentTracker:
    """Tracks what has been processed to avoid redundant work.

    Persists to a JSON file in the vault. Each sweep type maintains a
    set of processed fact IDs and a last-run timestamp.
    """

    SWEEP_TYPES = ["RECLASSIFY", "RELATE", "STALE", "PATTERNS", "CONSOLIDATE"]
    MAX_TRACKED_IDS = 10000  # Rotate after this many to keep state small

    def __init__(self, vault_path: str):
        self._path = os.path.join(vault_path, "memory", ".enricher_state.json")
        self._state = self._load()

    def is_processed(self, sweep_type: str, fact_id: str) -> bool:
        return fact_id in self._state.get("processed_ids", {}).get(sweep_type, [])

    def mark_processed(self, sweep_type: str, fact_ids: list[str]):
        if sweep_type not in self._state["processed_ids"]:
            self._state["processed_ids"][sweep_type] = []
        self._state["processed_ids"][sweep_type].extend(fact_ids)
        # Rotate if too large
        if len(self._state["processed_ids"][sweep_type]) > self.MAX_TRACKED_IDS:
            self._state["processed_ids"][sweep_type] = self._state["processed_ids"][sweep_type][-5000:]
        self._state["last_sweep"][sweep_type] = datetime.now().isoformat()

    def last_sweep_time(self, sweep_type: str) -> Optional[datetime]:
        ts = self._state.get("last_sweep", {}).get(sweep_type)
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return None

    def save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            log.warning("Failed to save enricher state: %s", e)

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": 1,
            "last_sweep": {},
            "processed_ids": {},
            "stats": {},
        }


# ── Sweep Engine ──────────────────────────────────────────────

class SweepEngine:
    """Implements the five enrichment sweeps.

    Each sweep reads from ChromaDB and/or vault files, processes facts,
    and writes enrichments back to the vault. All sweeps are idempotent.
    """

    # Time-sensitive markers for staleness detection
    TEMPORAL_MARKERS = [
        "currently", "right now", "as of", "today", "this week",
        "this month", "for now", "at the moment", "latest", "recent",
        "next week", "next month", "tomorrow", "by end of",
        "next tuesday", "next monday", "next wednesday", "next thursday",
        "next friday",
    ]

    def __init__(self, store, vault, tracker: EnrichmentTracker,
                 classifier: Optional[Classifier] = None,
                 embedder: Optional[Embedder] = None):
        self.store = store
        self.vault = vault
        self.tracker = tracker
        self.classifier = classifier or RegexClassifier()
        self.embedder = embedder or CPUEmbedder(model=store.emb_model)

    # ── SWEEP 1: Reclassify ──────────────────────────────────

    def sweep_reclassify(self, batch_size: int = 100) -> list[dict]:
        """Find 'general' facts that should be decision/task/preference/quantitative.

        Re-runs classification on every general fact. If the classifier (regex or ANE)
        now returns a different type, the fact is reclassified in ChromaDB and the
        vault files are updated.
        """
        results = []
        offset = 0

        while True:
            batch = self.store.get_by_type("general", limit=batch_size, offset=offset)
            if not batch["ids"]:
                break

            reclassified_ids = []
            for i, fact_id in enumerate(batch["ids"]):
                if self.tracker.is_processed("RECLASSIFY", fact_id):
                    continue

                text = batch["documents"][i]
                meta = batch["metadatas"][i]
                new_type = self.classifier.classify(text)

                if new_type != "general":
                    # Update ChromaDB
                    meta["type"] = new_type
                    meta["reclassified_at"] = datetime.now().isoformat()
                    meta["reclassified_from"] = "general"
                    self.store.collection.update(ids=[fact_id], metadatas=[meta])

                    # Write to correct vault type file
                    entities = []
                    try:
                        entities = json.loads(meta.get("entities", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        pass

                    fact_dict = {
                        "text": text, "type": new_type, "entities": entities,
                        "timestamp": meta.get("timestamp", datetime.now().isoformat()),
                    }
                    self.vault.write_fact(fact_dict, category=new_type)

                    # Remove from general.md
                    self._remove_from_vault_file("memory/facts/general.md", text)

                    results.append({
                        "action": "reclassified",
                        "fact_id": fact_id,
                        "text": text[:80],
                        "old_type": "general",
                        "new_type": new_type,
                    })
                    log.info("Reclassified: [general→%s] %s", new_type, text[:60])

                reclassified_ids.append(fact_id)

            self.tracker.mark_processed("RECLASSIFY", reclassified_ids)
            offset += batch_size

            # Don't process forever in one sweep
            if offset > 1000:
                break

        return results

    # ── SWEEP 2: Relate ──────────────────────────────────────

    def sweep_relate(self, batch_size: int = 500) -> list[dict]:
        """Build entity relationship graph from co-occurring entities.

        Two entities are related if they appear in the same fact. The relationship
        strength is the number of shared facts. Written to memory/relationships.md
        with Obsidian wikilinks.
        """
        # Build entity → fact_ids index
        entity_facts = defaultdict(list)
        entity_texts = defaultdict(list)
        offset = 0

        while True:
            batch = self.store.get_all(limit=batch_size, offset=offset)
            if not batch["ids"]:
                break

            for i, fact_id in enumerate(batch["ids"]):
                meta = batch["metadatas"][i]
                try:
                    entities = json.loads(meta.get("entities", "[]"))
                except (json.JSONDecodeError, TypeError):
                    entities = []
                for entity in entities:
                    entity_facts[entity].append(fact_id)
                    entity_texts[entity].append(batch["documents"][i])

            offset += batch_size
            if offset > 5000:
                break

        # Find co-occurring entity pairs
        relationships = defaultdict(lambda: {"count": 0, "sample_facts": []})
        entities_list = list(entity_facts.keys())

        for i, ent_a in enumerate(entities_list):
            facts_a = set(entity_facts[ent_a])
            for ent_b in entities_list[i + 1:]:
                facts_b = set(entity_facts[ent_b])
                shared = facts_a & facts_b
                if shared:
                    key = tuple(sorted([ent_a, ent_b]))
                    relationships[key]["count"] = len(shared)
                    relationships[key]["entities"] = list(key)

        # Write relationships.md
        if relationships:
            self._write_relationships(relationships, entity_facts)

        self.tracker.mark_processed("RELATE", ["full_sweep"])
        return [{"action": "relationships_updated", "entity_count": len(entity_facts),
                 "relationship_count": len(relationships)}]

    # ── SWEEP 3: Stale ───────────────────────────────────────

    def sweep_stale(self, stale_days: int = 14, batch_size: int = 500) -> list[dict]:
        """Flag time-sensitive facts that are likely outdated.

        Tasks older than stale_days are flagged. Facts with temporal markers
        ("currently", "this week", "next Tuesday") are also flagged if old.
        Does NOT delete — only flags for human review.
        """
        results = []
        cutoff = datetime.now() - timedelta(days=stale_days)
        offset = 0

        while True:
            batch = self.store.get_all(limit=batch_size, offset=offset)
            if not batch["ids"]:
                break

            stale_ids = []
            for i, fact_id in enumerate(batch["ids"]):
                if self.tracker.is_processed("STALE", fact_id):
                    continue

                meta = batch["metadatas"][i]
                text = batch["documents"][i]

                try:
                    fact_time = datetime.fromisoformat(meta.get("timestamp", ""))
                except (ValueError, TypeError):
                    stale_ids.append(fact_id)
                    continue

                if fact_time > cutoff:
                    stale_ids.append(fact_id)
                    continue  # Not old enough

                # Check if time-sensitive
                is_temporal = False
                text_lower = text.lower()

                # Tasks are inherently time-sensitive
                if meta.get("type") == "task":
                    is_temporal = True

                # Check temporal markers
                if any(m in text_lower for m in self.TEMPORAL_MARKERS):
                    is_temporal = True

                if is_temporal:
                    age_days = (datetime.now() - fact_time).days
                    results.append({
                        "action": "stale_flagged",
                        "fact_id": fact_id,
                        "text": text[:80],
                        "type": meta.get("type", "general"),
                        "age_days": age_days,
                    })

                stale_ids.append(fact_id)

            self.tracker.mark_processed("STALE", stale_ids)
            offset += batch_size
            if offset > 5000:
                break

        # Write stale insights
        if results:
            self._write_stale_insights(results)

        return results

    # ── SWEEP 4: Patterns ────────────────────────────────────

    def sweep_patterns(self, batch_size: int = 500) -> list[dict]:
        """Cross-entity pattern analysis.

        Finds:
        - Quantity outliers (one entity's threshold far from group average)
        - Similar entity profiles (two counterparties with matching attributes)
        - Recurring patterns (same provision appearing across many entities)
        """
        # Gather all facts grouped by entity
        entity_data = defaultdict(lambda: {"facts": [], "quantities": [], "types": defaultdict(int)})
        offset = 0

        while True:
            batch = self.store.get_all(limit=batch_size, offset=offset)
            if not batch["ids"]:
                break

            for i in range(len(batch["ids"])):
                meta = batch["metadatas"][i]
                text = batch["documents"][i]
                try:
                    entities = json.loads(meta.get("entities", "[]"))
                    quantities = json.loads(meta.get("quantities", "[]"))
                except (json.JSONDecodeError, TypeError):
                    entities, quantities = [], []

                for entity in entities:
                    entity_data[entity]["facts"].append(text)
                    entity_data[entity]["quantities"].extend(quantities)
                    entity_data[entity]["types"][meta.get("type", "general")] += 1

            offset += batch_size
            if offset > 5000:
                break

        insights = []

        # ── Quantity comparison across entities ──
        # Parse dollar amounts for comparison
        entity_amounts = {}
        for entity, data in entity_data.items():
            amounts = []
            for q in data["quantities"]:
                parsed = self._parse_amount(q)
                if parsed is not None:
                    amounts.append(parsed)
            if amounts:
                entity_amounts[entity] = amounts

        # Find outliers (entities whose amounts differ >2x from median)
        if len(entity_amounts) >= 3:
            all_amounts = []
            for amounts in entity_amounts.values():
                all_amounts.extend(amounts)
            if all_amounts:
                median = sorted(all_amounts)[len(all_amounts) // 2]
                for entity, amounts in entity_amounts.items():
                    for amt in amounts:
                        if median > 0 and (amt / median > 3 or amt / median < 0.33):
                            insights.append({
                                "type": "quantity_outlier",
                                "entity": entity,
                                "amount": amt,
                                "median": median,
                                "ratio": round(amt / median, 2),
                            })

        # ── Entity similarity (by embedding) ──
        entities_with_facts = [(e, d) for e, d in entity_data.items() if len(d["facts"]) >= 3]
        if len(entities_with_facts) >= 2:
            # Embed concatenated facts for each entity
            entity_names = [e for e, _ in entities_with_facts]
            entity_summaries = [" ".join(d["facts"][:10]) for _, d in entities_with_facts]

            embeddings = self.embedder.encode(entity_summaries)

            # Find similar pairs
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim > 0.75:
                        insights.append({
                            "type": "similar_profiles",
                            "entity_a": entity_names[i],
                            "entity_b": entity_names[j],
                            "similarity": round(sim, 3),
                            "facts_a": len(entities_with_facts[i][1]["facts"]),
                            "facts_b": len(entities_with_facts[j][1]["facts"]),
                        })

        # ── Recurring provisions ──
        provision_entities = defaultdict(set)
        provision_pattern = re.compile(
            r'\b(cross[- ]default|close[- ]out\s+netting|automatic\s+early\s+termination'
            r'|eligible\s+collateral|minimum\s+transfer\s+amount|threshold\s+amount'
            r'|variation\s+margin|initial\s+margin|credit\s+support)\b',
            re.IGNORECASE
        )
        for entity, data in entity_data.items():
            for fact in data["facts"]:
                for match in provision_pattern.finditer(fact):
                    provision_entities[match.group(0).lower()].add(entity)

        for provision, entities_set in provision_entities.items():
            if len(entities_set) >= 3:
                insights.append({
                    "type": "recurring_provision",
                    "provision": provision,
                    "entity_count": len(entities_set),
                    "entities": sorted(entities_set)[:10],
                })

        # Write pattern insights
        if insights:
            self._write_pattern_insights(insights)

        self.tracker.mark_processed("PATTERNS", ["full_sweep"])
        return insights

    # ── SWEEP 5: Consolidate ─────────────────────────────────

    def sweep_consolidate(self, min_facts: int = 8) -> list[dict]:
        """Generate auto-summaries for entity pages with many facts.

        Reads each entity markdown file, groups facts by type, extracts key
        quantities, and writes a profile block at the top of the file.
        """
        results = []
        entities_dir = os.path.join(self.vault.vault_path, "memory", "entities")
        if not os.path.exists(entities_dir):
            return results

        for fname in os.listdir(entities_dir):
            if not fname.endswith(".md"):
                continue

            filepath = os.path.join(entities_dir, fname)
            entity_name = fname.replace(".md", "").replace("-", " ")

            with open(filepath, "r") as f:
                content = f.read()

            lines = content.split("\n")

            # Count fact lines (lines starting with "- [")
            fact_lines = [l for l in lines if l.startswith("- [")]
            if len(fact_lines) < min_facts:
                continue

            # Check if already consolidated recently
            if "> **Profile**" in content:
                # Check if fact count changed
                existing_match = re.search(r'> - (\d+) facts', content)
                if existing_match and int(existing_match.group(1)) == len(fact_lines):
                    continue  # No new facts since last consolidation

            # Parse facts
            type_counts = defaultdict(int)
            quantities = []
            date_range = [None, None]
            related_entities = set()

            for line in fact_lines:
                # Extract date
                date_match = re.match(r'- \[(\d{4}-\d{2}-\d{2})\]', line)
                if date_match:
                    d = date_match.group(1)
                    if date_range[0] is None or d < date_range[0]:
                        date_range[0] = d
                    if date_range[1] is None or d > date_range[1]:
                        date_range[1] = d

                # Extract wikilinks (related entities)
                for match in re.finditer(r'\[\[([^\]|]+)\|([^\]]+)\]\]', line):
                    related_entities.add(match.group(2))

                # Classify the fact text
                text = line.split("] ", 1)[1] if "] " in line else line
                clean_text = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', text)
                fact_type = self.classifier.classify(clean_text)
                type_counts[fact_type] += 1

                # Extract quantities
                try:
                    from phantom_memory.daemon import FactExtractor
                except ImportError:
                    from daemon import FactExtractor
                for q in FactExtractor.QUANTITY_PATTERN.findall(clean_text):
                    if q not in quantities:
                        quantities.append(q)

            # Remove self from related entities
            related_entities.discard(entity_name)

            # Build profile block
            profile_lines = [
                "",
                f"> **Profile** (auto-generated {datetime.now().strftime('%Y-%m-%d')})",
                f"> - {len(fact_lines)} facts ({date_range[0] or '?'} to {date_range[1] or '?'})",
            ]

            type_summary = ", ".join(f"{c} {t}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]) if c > 0)
            if type_summary:
                profile_lines.append(f"> - {type_summary}")

            if quantities:
                profile_lines.append(f"> - Key quantities: {', '.join(quantities[:8])}")

            if related_entities:
                links = ", ".join(f"[[{self.vault._entity_filename(e)}|{e}]]" for e in sorted(related_entities)[:8]
                                 if self.vault._entity_filename(e))
                if links:
                    profile_lines.append(f"> - Related: {links}")

            # ANE-enhanced summary: use 1.7B model for intelligent profile
            if hasattr(self.classifier, 'has_ane') and self.classifier.has_ane:
                try:
                    facts_text = "\n".join(f"- {l.split('] ', 1)[1] if '] ' in l else l}" for l in fact_lines[:20])
                    summary = self.classifier.analyze(
                        f"Write a 2-sentence summary of this entity based on these facts:\n"
                        f"Entity: {entity_name}\n{facts_text}",
                        max_tokens=80,
                    )
                    if summary and not summary.startswith("[ANE"):
                        profile_lines.append(f"> ")
                        profile_lines.append(f"> *{summary.strip()}*")
                except Exception:
                    pass  # Silently skip ANE summary on error

            profile_lines.append("")

            profile_block = "\n".join(profile_lines)

            # Insert profile after header, replacing old profile if exists
            new_lines = []
            skip_old_profile = False
            header_found = False

            for line in lines:
                if line.startswith("# ") and not header_found:
                    new_lines.append(line)
                    new_lines.append(profile_block)
                    header_found = True
                    continue

                # Skip old profile block
                if line.startswith("> **Profile**"):
                    skip_old_profile = True
                    continue
                if skip_old_profile:
                    if line.startswith(">") or line.strip() == "":
                        continue
                    else:
                        skip_old_profile = False

                new_lines.append(line)

            # Write back atomically
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w") as f:
                f.write("\n".join(new_lines))
            os.replace(tmp_path, filepath)

            results.append({
                "action": "consolidated",
                "entity": entity_name,
                "fact_count": len(fact_lines),
                "types": dict(type_counts),
                "quantities": len(quantities),
            })
            log.info("Consolidated: %s (%d facts)", entity_name, len(fact_lines))

        self.tracker.mark_processed("CONSOLIDATE", ["full_sweep"])
        return results

    # ── Vault write helpers ──────────────────────────────────

    def _remove_from_vault_file(self, rel_path: str, text: str):
        """Remove a line containing `text` from a vault markdown file."""
        filepath = os.path.join(self.vault.vault_path, rel_path)
        if not os.path.exists(filepath):
            return

        # Clean wikilinks from search text for matching
        clean_text = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', text)
        match_prefix = clean_text[:60]

        with open(filepath, "r") as f:
            lines = f.readlines()

        new_lines = []
        removed = False
        for line in lines:
            line_clean = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', line)
            if match_prefix in line_clean and not removed:
                removed = True
                continue  # Skip this line
            new_lines.append(line)

        if removed:
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w") as f:
                f.writelines(new_lines)
            os.replace(tmp_path, filepath)

    def _write_relationships(self, relationships: dict, entity_facts: dict):
        """Write entity relationship graph to vault."""
        filepath = os.path.join(self.vault.vault_path, "memory", "relationships.md")

        lines = [
            f"# Entity Relationships",
            f"*Auto-generated by Phantom Enricher — {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        # Group by entity
        entity_relations = defaultdict(list)
        for (ent_a, ent_b), data in sorted(relationships.items(), key=lambda x: -x[1]["count"]):
            entity_relations[ent_a].append((ent_b, data["count"]))
            entity_relations[ent_b].append((ent_a, data["count"]))

        for entity in sorted(entity_relations.keys()):
            safe_name = self.vault._entity_filename(entity)
            if not safe_name:
                continue
            lines.append(f"## [[{safe_name}|{entity}]]")
            lines.append(f"*{len(entity_facts.get(entity, []))} facts*")
            relations = sorted(entity_relations[entity], key=lambda x: -x[1])
            for related, count in relations[:10]:
                rel_safe = self.vault._entity_filename(related)
                if rel_safe:
                    lines.append(f"- [[{rel_safe}|{related}]] ({count} shared facts)")
            lines.append("")

        # Also write graph.json for dashboard
        graph = {
            "nodes": [{"id": e, "fact_count": len(entity_facts.get(e, []))} for e in entity_relations],
            "edges": [{"source": a, "target": b, "weight": d["count"]}
                      for (a, b), d in relationships.items()],
        }

        tmp = filepath + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines))
        os.replace(tmp, filepath)

        graph_path = os.path.join(self.vault.vault_path, "memory", "graph.json")
        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)

    def _write_stale_insights(self, stale_items: list[dict]):
        """Write stale fact report to insights directory."""
        insights_dir = os.path.join(self.vault.vault_path, "memory", "insights")
        os.makedirs(insights_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(insights_dir, f"stale-{date_str}.md")

        lines = [
            f"# Stale Items — {date_str}",
            f"*{len(stale_items)} potentially outdated facts flagged by Phantom Enricher*",
            "",
        ]

        # Group by type
        by_type = defaultdict(list)
        for item in stale_items:
            by_type[item.get("type", "general")].append(item)

        for fact_type, items in sorted(by_type.items()):
            lines.append(f"## {fact_type.title()} ({len(items)})")
            for item in items:
                lines.append(f"- ⏳ {item['text']} *({item['age_days']} days old)*")
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def _write_pattern_insights(self, insights: list[dict]):
        """Write cross-entity pattern analysis to insights directory."""
        insights_dir = os.path.join(self.vault.vault_path, "memory", "insights")
        os.makedirs(insights_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(insights_dir, f"patterns-{date_str}.md")

        lines = [
            f"# Pattern Insights — {date_str}",
            f"*Cross-entity analysis by Phantom Enricher*",
            "",
        ]

        # Quantity outliers
        outliers = [i for i in insights if i["type"] == "quantity_outlier"]
        if outliers:
            lines.append("## Quantity Outliers")
            for o in outliers:
                direction = "above" if o["ratio"] > 1 else "below"
                lines.append(f"- **{o['entity']}**: ${o['amount']:,.0f} ({o['ratio']}x {direction} median ${o['median']:,.0f})")
            lines.append("")

        # Similar profiles
        similar = [i for i in insights if i["type"] == "similar_profiles"]
        if similar:
            lines.append("## Similar Entity Profiles")
            for s in sorted(similar, key=lambda x: -x["similarity"]):
                lines.append(f"- **{s['entity_a']}** ↔ **{s['entity_b']}** "
                            f"(similarity: {s['similarity']:.0%}, {s['facts_a']} vs {s['facts_b']} facts)")
            lines.append("")

        # Recurring provisions
        recurring = [i for i in insights if i["type"] == "recurring_provision"]
        if recurring:
            lines.append("## Recurring Provisions")
            for r in sorted(recurring, key=lambda x: -x["entity_count"]):
                entities_str = ", ".join(r["entities"][:5])
                more = f" +{r['entity_count'] - 5} more" if r["entity_count"] > 5 else ""
                lines.append(f"- **{r['provision']}** appears across {r['entity_count']} entities: {entities_str}{more}")
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def _parse_amount(self, quantity_str: str) -> Optional[float]:
        """Parse a dollar amount string into a float. Only actual currency amounts."""
        try:
            # Only parse strings that start with a currency symbol or code
            if not re.match(r'^[\$£€]|^(?:USD|GBP|EUR)\s', quantity_str):
                return None
            clean = re.sub(r'[\$£€,]', '', quantity_str)
            clean = re.sub(r'^(USD|GBP|EUR)\s*', '', clean)
            match = re.match(r'([\d.]+)\s*(million|billion|M|B|K)?', clean, re.IGNORECASE)
            if not match:
                return None
            value = float(match.group(1))
            multiplier = match.group(2)
            if multiplier:
                m = multiplier.lower()
                if m in ('million', 'm'):
                    value *= 1_000_000
                elif m in ('billion', 'b'):
                    value *= 1_000_000_000
                elif m == 'k':
                    value *= 1_000
            # Skip tiny values (likely not real dollar amounts)
            elif value < 1000:
                return None
            return value
        except (ValueError, AttributeError):
            return None


# ── Phantom Enricher (orchestrator) ───────────────────────────

class PhantomEnricher:
    """Always-on background intelligence for your knowledge vault.

    Runs five sweeps in a continuous loop:
    1. RECLASSIFY — fix mistyped facts
    2. RELATE — build entity relationship graph
    3. STALE — flag outdated information
    4. PATTERNS — cross-entity analysis
    5. CONSOLIDATE — auto-generate entity summaries

    While your LLM sleeps, the enricher makes your vault smarter.

    Args:
        store: MemoryStore instance (shared with daemon)
        vault: VaultWriter instance (shared with daemon)
        interval: Seconds between full sweep cycles (default: 300 = 5 min)
        classifier: Optional custom classifier (default: regex). Swap for ANE.
        embedder: Optional custom embedder (default: CPU). Swap for ANE.

    Usage:
        enricher = PhantomEnricher(store=store, vault=vault)
        enricher.start()
        # ... runs forever ...
        enricher.stop()
    """

    DEFAULT_SWEEP_ORDER = ["RECLASSIFY", "RELATE", "STALE", "PATTERNS", "CONSOLIDATE"]

    def __init__(self, store, vault, interval: int = 300,
                 sweep_order: list[str] = None,
                 classifier: Optional[Classifier] = None,
                 embedder: Optional[Embedder] = None):
        self.store = store
        self.vault = vault
        self.interval = interval
        self.sweep_order = sweep_order or self.DEFAULT_SWEEP_ORDER

        self.tracker = EnrichmentTracker(vault.vault_path)
        self.engine = SweepEngine(
            store=store, vault=vault, tracker=self.tracker,
            classifier=classifier, embedder=embedder,
        )

        self._running = False
        self._thread = None
        self._stats = {
            "cycles": 0,
            "reclassified": 0,
            "relationships": 0,
            "stale_flagged": 0,
            "patterns": 0,
            "consolidated": 0,
            "errors": 0,
            "last_cycle": None,
        }

    def start(self):
        """Start the enricher background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="phantom-enricher")
        self._thread.start()
        log.info("Enricher started (interval=%ds, sweeps=%s)", self.interval, self.sweep_order)

    def stop(self):
        """Stop the enricher and save state."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self.tracker.save()
        log.info("Enricher stopped. Stats: %s", self._stats)

    def run_once(self):
        """Run all sweeps once (for testing or CLI one-shot mode)."""
        for sweep_type in self.sweep_order:
            self._run_sweep(sweep_type)
        self.tracker.save()

    @property
    def stats(self) -> dict:
        return {**self._stats}

    def _run_loop(self):
        """Main loop: cycle through sweeps, sleep between cycles."""
        while self._running:
            for sweep_type in self.sweep_order:
                if not self._running:
                    break
                self._run_sweep(sweep_type)

            self._stats["cycles"] += 1
            self._stats["last_cycle"] = datetime.now().isoformat()
            self.tracker.save()

            # Sleep in small increments so stop() is responsive
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)

    def _run_sweep(self, sweep_type: str):
        """Execute a single sweep type."""
        sweep_map = {
            "RECLASSIFY": self.engine.sweep_reclassify,
            "RELATE": self.engine.sweep_relate,
            "STALE": self.engine.sweep_stale,
            "PATTERNS": self.engine.sweep_patterns,
            "CONSOLIDATE": self.engine.sweep_consolidate,
        }

        sweep_fn = sweep_map.get(sweep_type)
        if not sweep_fn:
            log.warning("Unknown sweep type: %s", sweep_type)
            return

        try:
            t0 = time.monotonic()
            results = sweep_fn()
            elapsed = time.monotonic() - t0

            # Update stats
            stat_key = {
                "RECLASSIFY": "reclassified",
                "RELATE": "relationships",
                "STALE": "stale_flagged",
                "PATTERNS": "patterns",
                "CONSOLIDATE": "consolidated",
            }.get(sweep_type, sweep_type.lower())
            self._stats[stat_key] += len(results)

            if results:
                log.info("Sweep %s: %d results in %.1fs", sweep_type, len(results), elapsed)
            else:
                log.debug("Sweep %s: no changes (%.1fs)", sweep_type, elapsed)

        except Exception as e:
            self._stats["errors"] += 1
            log.error("Sweep %s failed: %s", sweep_type, e, exc_info=True)

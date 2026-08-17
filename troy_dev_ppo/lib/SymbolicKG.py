import torch
import torch.nn.functional as F

from lib.Perception import EMBEDDING_DIM, RELATION_NAMES

SYMBOL_DIM = 4
EMA_ALPHA = 0.1


class SymbolicKG:
    """Concept-level knowledge graph grounded in visual embeddings.

    Each node holds a 512-dim visual embedding (EMA over merged detections)
    and a random 4-dim symbol. Relation types are a fixed small vocabulary,
    each with its own random 4-dim symbol. Edges record which relations were
    observed between which concepts during phase 1 (with counts).

    Phase 2 freezes the structure: detections are matched against existing
    nodes only, and the symbols are handed to the policy, where they receive
    PPO gradients.
    """

    def __init__(
        self, embedding_dim=EMBEDDING_DIM, similarity_threshold=0.50, device="cuda", seed=0
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.embeddings = torch.empty(0, embedding_dim, device=device)
        self.symbols = torch.empty(0, SYMBOL_DIM)
        self.counts = []
        self.relation_names = list(RELATION_NAMES)
        self.relation_symbols = torch.randn(
            len(self.relation_names), SYMBOL_DIM, generator=gen
        )
        self.edges = {}  # (src, rel_idx, dst) -> count
        self._gen = gen

    @property
    def num_nodes(self):
        return self.embeddings.shape[0]

    # ------------------------------------------------------------------ match

    def _best_match(self, embedding):
        """embedding: [512] normalized tensor.

        Returns (node_id or None, best_sim, all_sims). all_sims is the
        similarity to every node *at this moment*; diagnostics need that
        snapshot because node embeddings drift under the EMA update, so
        recomputing a similarity even one detection later gives a different
        number.
        """
        if self.num_nodes == 0:
            return None, -1.0, torch.empty(0, device=self.device)
        sims = self.embeddings @ embedding.to(self.device)
        best = int(torch.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= self.similarity_threshold:
            return best, best_sim, sims
        return None, best_sim, sims

    def match_or_add(self, embedding):
        """Phase 1: merge into the best node above threshold (EMA update) or
        create a new node with a fresh random symbol.

        Returns (node_id, created, best_sim, all_sims), all measured *before*
        this detection was absorbed. On a creation, best_sim is the near-miss
        that fell below the threshold."""
        embedding = embedding.to(self.device)
        node_id, best_sim, sims = self._best_match(embedding)
        if node_id is not None:
            updated = (1 - EMA_ALPHA) * self.embeddings[node_id] + EMA_ALPHA * embedding
            self.embeddings[node_id] = F.normalize(updated, dim=0)
            self.counts[node_id] += 1
            return node_id, False, best_sim, sims
        self.embeddings = torch.cat([self.embeddings, embedding.unsqueeze(0)])
        self.symbols = torch.cat(
            [self.symbols, torch.randn(1, SYMBOL_DIM, generator=self._gen)]
        )
        self.counts.append(1)
        return self.num_nodes - 1, True, best_sim, sims

    def match_only(self, embedding):
        """Phase 2: match against frozen structure. Returns node_id or None."""
        node_id, _, _ = self._best_match(embedding.to(self.device))
        return node_id

    def match_batch(self, embeddings, allow_add, with_info=False):
        """Match [N,512] embeddings. Returns list of node_id or None.

        with_info=True additionally returns a per-detection list of
        {"created", "best_sim", "sims"} dicts, for oracle diagnostics.
        """
        ids, info = [], []
        for emb in embeddings:
            if allow_add:
                node_id, created, best_sim, sims = self.match_or_add(emb)
            else:
                node_id = self.match_only(emb)
                created, best_sim, sims = False, -1.0, torch.empty(0)
            ids.append(node_id)
            if with_info:
                info.append({"created": created, "best_sim": best_sim, "sims": sims})
        return (ids, info) if with_info else ids

    def match_batch_sims(self, embeddings):
        """Phase 2: match [N,E] embeddings against the frozen node set.
        Returns (ids, sims): node_id or None per detection, plus the best
        cosine similarity (reported even when below threshold)."""
        ids, sims = [], []
        for emb in embeddings:
            node_id, sim, _ = self._best_match(emb.to(self.device))
            ids.append(node_id)
            sims.append(sim)
        return ids, sims

    # ------------------------------------------------------------------ edges

    def relation_index(self, name):
        return self.relation_names.index(name)

    def add_edge(self, src, relation_name, dst):
        key = (src, self.relation_index(relation_name), dst)
        self.edges[key] = self.edges.get(key, 0) + 1

    # ------------------------------------------------------------ consolidate

    def consolidate(self, merge_threshold=0.85, min_count=5):
        """Post-phase-1 cleanup for online-clustering splinter.

        Repeatedly merges the most-similar node pair above merge_threshold
        (count-weighted mean embedding; the higher-count node's symbol wins),
        then prunes nodes seen fewer than min_count times. Edges are remapped;
        self-edges and edges touching pruned nodes are dropped.
        Returns {old_id: new_id or None} for logging.
        """
        remap = {i: i for i in range(self.num_nodes)}

        def roots():
            return sorted(set(remap[i] for i in remap if remap[i] is not None))

        while True:
            ids = roots()
            best_pair, best_sim = None, merge_threshold
            for ai, a in enumerate(ids):
                for b in ids[ai + 1 :]:
                    sim = float(self.embeddings[a] @ self.embeddings[b])
                    if sim >= best_sim:
                        best_pair, best_sim = (a, b), sim
            if best_pair is None:
                break
            a, b = best_pair
            keep, drop = (a, b) if self.counts[a] >= self.counts[b] else (b, a)
            total = self.counts[keep] + self.counts[drop]
            merged = (
                self.counts[keep] * self.embeddings[keep]
                + self.counts[drop] * self.embeddings[drop]
            ) / total
            self.embeddings[keep] = F.normalize(merged, dim=0)
            self.counts[keep] = total
            for i in remap:
                if remap[i] == drop:
                    remap[i] = keep

        for i in remap:
            if remap[i] is not None and self.counts[remap[i]] < min_count:
                remap[i] = None

        kept = [i for i in roots() if self.counts[i] >= min_count]
        new_index = {old: new for new, old in enumerate(kept)}
        for i in remap:
            remap[i] = new_index.get(remap[i]) if remap[i] is not None else None

        self.embeddings = self.embeddings[kept]
        self.symbols = self.symbols[kept]
        self.counts = [self.counts[i] for i in kept]

        new_edges = {}
        for (src, rel, dst), count in self.edges.items():
            ns, nd = remap.get(src), remap.get(dst)
            if ns is None or nd is None or ns == nd:
                continue
            key = (ns, rel, nd)
            new_edges[key] = new_edges.get(key, 0) + count
        self.edges = new_edges
        return remap

    # ------------------------------------------------------------------ io

    def save(self, path):
        torch.save(
            {
                "embeddings": self.embeddings.cpu(),
                "symbols": self.symbols.cpu(),
                "counts": self.counts,
                "relation_names": self.relation_names,
                "relation_symbols": self.relation_symbols.cpu(),
                "edges": self.edges,
                "similarity_threshold": self.similarity_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cuda"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        kg = cls(
            embedding_dim=blob["embeddings"].shape[1],
            similarity_threshold=blob["similarity_threshold"],
            device=device,
        )
        kg.embeddings = blob["embeddings"].to(device)
        kg.symbols = blob["symbols"]
        kg.counts = blob["counts"]
        kg.relation_names = blob["relation_names"]
        kg.relation_symbols = blob["relation_symbols"]
        kg.edges = blob["edges"]
        return kg

    # ------------------------------------------------------------------ debug

    def summary(self):
        lines = [f"SymbolicKG: {self.num_nodes} nodes, {len(self.edges)} distinct edges"]
        for i in range(self.num_nodes):
            lines.append(
                f"  node {i}: merged {self.counts[i]} detections, "
                f"symbol {self.symbols[i].tolist()}"
            )
        if self.num_nodes > 1:
            sims = (self.embeddings @ self.embeddings.T).cpu()
            lines.append("  pairwise visual cosine similarity:")
            for i in range(self.num_nodes):
                row = " ".join(f"{sims[i, j]:+.3f}" for j in range(self.num_nodes))
                lines.append(f"    node {i}: {row}")
        for (src, rel, dst), count in sorted(self.edges.items(), key=lambda kv: -kv[1]):
            lines.append(f"  edge: {src} --{self.relation_names[rel]}--> {dst}  (seen {count}x)")
        return "\n".join(lines)

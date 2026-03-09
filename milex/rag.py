"""Local RAG (Retrieval-Augmented Generation) for MILEX."""
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ollama
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

from .ui import console, print_info, print_success, print_warning, ThinkingSpinner

# Try to import tree-sitter for advanced context
try:
    import tree_sitter_languages
    from tree_sitter import Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

class RagManager:
    """Handles file indexing, chunking, and vector search using Ollama embeddings."""

    def __init__(self, config: dict):
        self.config = config
        self.rag_config = config.get("rag", {})
        self.client = ollama.Client(host=config["ollama_host"])
        self.storage_dir = Path.home() / ".milex" / "rag_index"
        self.index_file = self.storage_dir / "index.json"

        # In-memory index
        self.chunks: List[Dict] = []  # List of {text, path, start_line}
        self.embeddings: Optional[np.ndarray] = None

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self):
        """Load the persisted index if it exists."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    # We don't store embeddings in JSON usually because of size,
                    # but for small projects we could.
                    # For now, we'll re-embed or expect persistence if small.
                    if "embeddings" in data and data["embeddings"]:
                        self.embeddings = np.array(data["embeddings"])
            except Exception as e:
                print_warning(f"Failed to load RAG index: {e}")

    async def _generate_embeddings_async(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings asynchronously using asyncio."""
        batch_size = self.rag_config.get("batch_size", 32)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Process batch concurrently
            tasks = [
                asyncio.to_thread(self.client.embeddings, model=model, prompt=text)
                for text in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print_warning(f"Embedding error: {result}")
                    all_embeddings.append([0.0] * 768)  # Fallback
                else:
                    all_embeddings.append(result["embedding"])

        return all_embeddings

    def _save_index(self):
        """Save the current index to disk."""
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
        with open(self.index_file, 'w') as f:
            json.dump(data, f)

    def index_directory(self, root_path: str = "."):
        """Scan, chunk, and embed all relevant files in a directory."""
        root = Path(root_path).resolve()
        exclude_dirs = self.rag_config.get("exclude_dirs", [])
        chunk_size = self.rag_config.get("chunk_size", 1000)
        chunk_overlap = self.rag_config.get("chunk_overlap", 100)

        # Respect .gitignore
        ignore_spec = self._get_ignore_spec(root)

        new_chunks = []
        files_to_index = []

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            # Skip excluded dirs
            if any(part in exclude_dirs for part in p.parts):
                continue

            # Skip matches in gitignore
            rel_p = p.relative_to(root)
            if ignore_spec and ignore_spec.match_file(str(rel_p)):
                continue

            # Only index text-like files
            if p.suffix.lower() not in ('.py', '.js', '.md', '.txt', '.go', '.html', '.css', '.c', '.cpp', '.h', '.sh', '.yml', '.yaml', '.toml', '.json'):
                continue

            files_to_index.append(p)

        if not files_to_index:
            print_warning("No files found to index.")
            return

        with ThinkingSpinner(f"Indexing {len(files_to_index)} files..."):
            for p in files_to_index:
                try:
                    content = p.read_text(errors="replace")
                    file_chunks = self._chunk_text(content, str(p.relative_to(root)), chunk_size, chunk_overlap)
                    new_chunks.extend(file_chunks)
                except Exception:
                    continue

        if not new_chunks:
            return

        # Generate embeddings using async method
        model = self.config.get("roles", {}).get("embeddings", "nomic-embed-text:latest")

        # Run async embedding generation in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    all_embeddings = []
                    texts = [c["text"] for c in new_chunks]
                    batch_size = 32
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        for text in batch:
                            resp = self.client.embeddings(model=model, prompt=text)
                            all_embeddings.append(resp["embedding"])
            else:
                all_embeddings = loop.run_until_complete(
                    self._generate_embeddings_async([c["text"] for c in new_chunks], model)
                )
        except RuntimeError:
            # No event loop, run synchronously
            all_embeddings = []
            texts = [c["text"] for c in new_chunks]
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                for text in batch:
                    resp = self.client.embeddings(model=model, prompt=text)
                    all_embeddings.append(resp["embedding"])

        self.chunks = new_chunks
        self.embeddings = np.array(all_embeddings)
        self._save_index()
        print_success(f"Indexed {len(self.chunks)} chunks from {len(files_to_index)} files.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for the most relevant chunks for a query."""
        if self.embeddings is None or not self.chunks:
            return []

        model = self.config.get("roles", {}).get("embeddings", "nomic-embed-text:latest")
        try:
            resp = self.client.embeddings(model=model, prompt=query)
            query_embedding = np.array(resp["embedding"])
            
            # Cosine similarity
            # Since Ollama embeddings are typically normalized or we can just use dot product if normalized
            norms = np.linalg.norm(self.embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            
            if query_norm == 0 or np.any(norms == 0):
                return []
                
            similarities = np.dot(self.embeddings, query_embedding) / (norms * query_norm)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            threshold = self.config.get("rag", {}).get("similarity_threshold", 0.3)
            for idx in top_indices:
                if similarities[idx] > threshold: # Threshold
                    chunk = self.chunks[idx].copy()
                    chunk["score"] = float(similarities[idx])
                    results.append(chunk)
            return results
        except Exception as e:
            print_error(f"Search failed: {e}")
            return []

    def _chunk_text(self, text: str, path_str: str, size: int, overlap: int) -> List[Dict]:
        """Split text into overlapping chunks, using tree-sitter if available."""
        path = Path(path_str)
        ext = path.suffix.lower().lstrip(".")
        
        # Mapping extension to tree-sitter language names
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "go": "go",
            "rb": "ruby",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "rs": "rust"
        }

        if TREE_SITTER_AVAILABLE and ext in lang_map:
            try:
                return self._tree_sitter_chunk(text, path_str, lang_map[ext])
            except Exception as e:
                # Fallback to simple chunking if it fails
                pass

        # Simple line-based chunking for code to preserve context
        chunks = []
        lines = text.splitlines()
        current_chunk = []
        current_size = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_size += len(line) + 1
            
            if current_size >= size:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "path": path_str,
                    "start_line": start_line
                })
                # Overlap: keep last few lines
                overlap_lines = lines[max(0, i-2):i+1] # basic overlap
                current_chunk = overlap_lines
                current_size = sum(len(l) for l in current_chunk)
                start_line = i + 1
                
        if current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "path": path_str,
                "start_line": start_line
            })
        return chunks

    def _tree_sitter_chunk(self, text: str, path: str, lang_name: str) -> List[Dict]:
        """Use tree-sitter to chunk by function/class/method."""
        language = tree_sitter_languages.get_language(lang_name)
        parser = Parser()
        parser.set_language(language)
        
        tree = parser.parse(bytes(text, "utf8"))
        
        # Simple extraction of top-level blocks
        chunks = []
        query_str = """
            (function_definition) @func
            (class_definition) @class
            (module) @mod
            (method_definition) @meth
        """
        if lang_name == "javascript" or lang_name == "typescript":
            query_str = "(function_declaration) @func (class_declaration) @class"
            
        try:
            query = language.query(query_str)
            captures = query.captures(tree.root_node)
            
            for node, tag in captures:
                # Only take nodes with significant size but not the whole file if multiple
                if node.end_byte - node.start_byte < 50: continue
                
                chunk_text = text[node.start_byte:node.end_byte]
                chunks.append({
                    "text": chunk_text,
                    "path": path,
                    "start_line": node.start_point[0] + 1,
                    "type": tag
                })
        except Exception:
            # If query fails, just chunk by top-level nodes
            for child in tree.root_node.children:
                if child.end_byte - child.start_byte > 100:
                    chunks.append({
                        "text": text[child.start_byte:child.end_byte],
                        "path": path,
                        "start_line": child.start_point[0] + 1
                    })
        
        # If no semantic chunks found, return empty to trigger fallback
        return chunks

    def _get_ignore_spec(self, root: Path) -> Optional[PathSpec]:
        """Load .gitignore if it exists."""
        gitignore = root / ".gitignore"
        if gitignore.exists():
            try:
                patterns = gitignore.read_text().splitlines()
                return PathSpec.from_lines(GitWildMatchPattern, patterns)
            except Exception:
                pass
        return None

def print_error(msg: str):
    from .ui import console
    console.print(f"[bold red]✗ Error:[/] {msg}")

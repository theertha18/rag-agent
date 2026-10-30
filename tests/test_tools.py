"""
Tests for app/tools.py. Run from the project root with:

    python -m unittest tests.test_tools -v

(pytest runs these too.) No network, no API key and no ChromaDB are needed:
the document search gets a fake collection, GitHub gets a fake response, and
the database is built in a temporary folder.
"""

import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest import mock

from app import seed_db, tools


class FakeCollection:
    """Mimics the one ChromaDB call search_documents makes."""

    def __init__(self, rows):
        self.rows = rows  # list of (text, source, distance), closest first
        self.last_n_results = None

    def query(self, query_texts, n_results):
        self.last_n_results = n_results
        rows = self.rows[:n_results]
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[{"source": r[1]} for r in rows]],
            "distances": [[r[2] for r in rows]],
        }


class SearchDocumentsTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection([
            ("close chunk", "a.txt", 0.6),
            ("medium chunk", "b.txt", 1.2),
            ("far chunk", "c.txt", 1.7),
        ])

    def test_drops_chunks_beyond_max_distance(self):
        hits = tools.search_documents("q", collection=self.collection, max_distance=1.5)
        self.assertEqual([h["source"] for h in hits], ["a.txt", "b.txt"])

    def test_returns_empty_list_when_nothing_is_close(self):
        hits = tools.search_documents("q", collection=self.collection, max_distance=0.1)
        self.assertEqual(hits, [])

    def test_result_has_source_text_and_distance(self):
        hit = tools.search_documents("q", collection=self.collection)[0]
        self.assertEqual(hit, {"source": "a.txt", "text": "close chunk", "distance": 0.6})

    def test_passes_k_to_the_collection(self):
        tools.search_documents("q", k=2, collection=self.collection)
        self.assertEqual(self.collection.last_n_results, 2)


class FindProjectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = pathlib.Path(cls.tmp.name) / "portfolio.db"
        seed_db.seed(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def names(self, keyword, **kw):
        return [p["name"] for p in tools.find_projects(keyword, db_path=self.db, **kw)]

    def test_empty_keyword_lists_everything_up_to_the_limit(self):
        self.assertEqual(len(tools.find_projects("", limit=100, db_path=self.db)),
                         len(seed_db.PROJECTS))
        self.assertEqual(len(tools.find_projects("", limit=3, db_path=self.db)), 3)

    def test_matches_technologies_case_insensitively(self):
        found = self.names("FASTAPI")
        self.assertTrue(any("Clinical study data portal" in n for n in found))

    def test_matches_several_projects(self):
        self.assertGreaterEqual(len(self.names("ultrasonic")), 3)

    def test_no_match_returns_empty_list(self):
        self.assertEqual(self.names("cobol-mainframe"), [])

    def test_matches_whole_words_not_fragments(self):
        # "go" must find the Go project but not Django ("Service Management Portal")
        found = self.names("go")
        self.assertEqual(found, ["Rexroth Control Unit (RCU) microservices at Bosch Rexroth"])
        self.assertNotIn("Service Management Portal", found)

    def test_plural_forms_match(self):
        self.assertTrue(any("microservices" in n for n in self.names("microservice")))
        self.assertGreaterEqual(len(self.names("sensor")), 3)

    def test_multi_word_phrase_and_symbols(self):
        self.assertTrue(any("Mercedes" in n for n in self.names("90%")))
        self.assertTrue(any("thesis" in n.lower() for n in self.names("Isolation Forest")))

    def test_special_characters_are_plain_text(self):
        for keyword in ["%", "_", ".*", "(", "[a-z]+"]:
            self.assertEqual(self.names(keyword), [], keyword)

    def test_sql_injection_attempt_is_just_text(self):
        self.assertEqual(self.names("x'; DROP TABLE projects; --"), [])
        self.assertEqual(len(tools.find_projects("", limit=100, db_path=self.db)),
                         len(seed_db.PROJECTS))

    def test_result_fields(self):
        project = tools.find_projects("Mercedes", db_path=self.db)[0]
        self.assertEqual(set(project), {"name", "kind", "description", "technologies"})

    def test_missing_database_gives_a_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            tools.find_projects("x", db_path=pathlib.Path(self.tmp.name) / "nope.db")
        self.assertIn("seed_db", str(ctx.exception))


def fake_response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


REPOS = [
    {"name": "rag-agent", "description": "RAG demo", "language": "Python",
     "stargazers_count": 2, "updated_at": "2026-10-30T10:00:00Z",
     "html_url": "https://github.com/u/rag-agent", "fork": False},
    {"name": "someone-elses-fork", "description": None, "language": "Go",
     "stargazers_count": 0, "updated_at": "2026-01-01T00:00:00Z",
     "html_url": "https://github.com/u/someone-elses-fork", "fork": True},
    {"name": "portfolio", "description": None, "language": None,
     "stargazers_count": 0, "updated_at": "2025-05-05T00:00:00Z",
     "html_url": "https://github.com/u/portfolio", "fork": False},
]


class GithubReposTests(unittest.TestCase):
    def test_skips_forks_and_maps_fields(self):
        with mock.patch("urllib.request.urlopen", return_value=fake_response(REPOS)):
            repos = tools.github_repos("u")
        self.assertEqual([r["name"] for r in repos], ["rag-agent", "portfolio"])
        self.assertEqual(repos[0], {
            "name": "rag-agent", "description": "RAG demo", "language": "Python",
            "stars": 2, "updated": "2026-10-30", "url": "https://github.com/u/rag-agent",
        })

    def test_respects_limit(self):
        with mock.patch("urllib.request.urlopen", return_value=fake_response(REPOS)):
            self.assertEqual(len(tools.github_repos("u", limit=1)), 1)

    def test_unknown_user_raises_value_error(self):
        err = urllib.error.HTTPError("https://api.github.com", 404, "Not Found", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(ValueError):
                tools.github_repos("nobody")

    def test_rate_limit_raises_runtime_error(self):
        err = urllib.error.HTTPError("https://api.github.com", 403, "Forbidden", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(RuntimeError):
                tools.github_repos("u")

    def test_network_failure_raises_runtime_error(self):
        err = urllib.error.URLError("no route to host")
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(RuntimeError):
                tools.github_repos("u")


if __name__ == "__main__":
    unittest.main()

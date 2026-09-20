"""Smoke tests — verify the repo layout is intact.

These run on every push/PR via .github/workflows/python-test.yml.
Kept dependency-free (stdlib only) so the CI matrix stays fast.
"""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_readme_en_exists():
    assert (ROOT / 'README.md').is_file()


def test_readme_zh_cn_exists():
    """中文 README."""
    assert (ROOT / 'README.zh-CN.md').is_file()


def test_license_exists():
    assert (ROOT / 'LICENSE').is_file()


def test_gitignore_exists():
    assert (ROOT / '.gitignore').is_file()


def test_diagrams_exist():
    for f in ['diagram_1_arch.png', 'diagram_2_sim2real_flow.png', 'diagram_3_mindmap.png']:
        path = ROOT / f
        assert path.is_file(), f'missing diagram: {f}'
        assert path.stat().st_size > 1000, f'diagram too small: {f} ({path.stat().st_size} B)'


def test_tech_doc_exists():
    """工业视觉_sim2real_技术方案.md."""
    assert (ROOT / '工业视觉_sim2real_技术方案.md').is_file()
    assert (ROOT / '工业视觉_sim2real_技术方案.md').stat().st_size > 5000


def test_readme_en_not_empty():
    assert (ROOT / 'README.md').stat().st_size > 5000, "README.md is too short"


def test_readme_zh_cn_not_empty():
    assert (ROOT / 'README.zh-CN.md').stat().st_size > 5000, "README.zh-CN.md is too short"


def test_readme_en_mentions_key_concepts():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for kw in ['sim-to-real', 'MuJoCo', 'YOLOv8', 'Anomalib', 'Jev-Like', 'BlenderProc']:
        assert kw in text, f"README.md missing keyword: {kw}"


def test_readme_zh_mentions_key_concepts():
    text = (ROOT / 'README.zh-CN.md').read_text(encoding='utf-8')
    for kw in ['MuJoCo', 'YOLOv8', 'Anomalib', 'JevLikeLocal', 'BlenderProc', '缺陷']:
        assert kw in text, f"README.zh-CN.md missing keyword: {kw}"


def test_readme_zh_has_language_toggle():
    """中文 README 顶部应有英文链接."""
    text = (ROOT / 'README.zh-CN.md').read_text(encoding='utf-8')
    assert 'README.md' in text[:500], "README.zh-CN.md header missing English link"


def test_readme_en_has_language_toggle():
    """英文 README 顶部应有中文链接."""
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'README.zh-CN.md' in text[:500], "README.md header missing Chinese link"


def test_license_is_mit():
    text = (ROOT / 'LICENSE').read_text()
    assert 'MIT License' in text
    assert 'Copyright (c) 2026 Mavis' in text


def test_gitignore_covers_pycache():
    text = (ROOT / '.gitignore').read_text()
    assert '__pycache__' in text
    assert '.py[cod]' in text or '*.pyc' in text